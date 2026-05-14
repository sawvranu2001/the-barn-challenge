import heapq
import math
import numpy as np

class Node:
    def __init__(self, pos, hcost=0, gcost=0, parent=None, depth=0):
        self.pos = pos
        self.gcost = gcost
        self.hcost = hcost
        self.parent = parent
        self.depth = depth

    @property
    def fcost(self):
        return self.gcost + self.hcost

    def __lt__(self, other):
        return self.fcost < other.fcost


class GlobalPlanner:
    '''Tangent Intersection Guidance (TIG) Algorithm'''
    def __init__(self, r_safe=0.1, r_vir=0.0, wg=1.0, wh=1.0, max_depth=float('inf')):
        self.r_safe = r_safe
        self.r_vir = r_vir
        self.wg = wg
        self.wh = wh
        self.max_depth = max_depth

    def __call__(self, start, target, obstacles):
        # Extract lines and circles based on input structure
        Fl, Idl, BPl = obstacles.get('lines', ([], [], []))
        Fc, Idc = obstacles.get('circles', ([], []))
        self._all_obstacles = BPl + Fc

        tgt_key = self._pos_key(target)

        open_list = []
        closed_list = set()

        start_node = Node(pos=start, hcost=distance(start, target))
        heapq.heappush(open_list, start_node)

        while open_list:
            current = heapq.heappop(open_list)
            # print(f'{current.pos}, {current.depth}')
            closed_list.add(self._pos_key(current.pos))

            if self._pos_key(current.pos) == tgt_key or current.depth > self.max_depth:
                path = []
                while current:
                    path.append(current.pos)
                    current = current.parent
                return path[::-1] # Reverse to get Start -> Target

            waypoints = self._generate_waypoints(current.pos, target)

            for w_pos in waypoints:
                if self._pos_key(w_pos) in closed_list:
                    continue

                hcost = self.wh * distance(w_pos, target)
                gcost = self.wg * distance(current.pos, w_pos) + current.gcost
                depth = current.depth + 1
                next_node = Node(pos=w_pos, hcost=hcost, gcost=gcost, parent=current, depth=depth)
                heapq.heappush(open_list, next_node)
        return None 

    def _generate_waypoints(self, src, tgt):
        waypoints = []
        to_explore = [tgt]
        explored = set() 
        while to_explore:
            next_pt = to_explore.pop(0)
            # print(next_pt, end=': ')
            explored.add(self._pos_key(next_pt))
            no_ids = self._intersected_obstacles(src, next_pt, self._all_obstacles, True)
            # waypoints.append(next_pt)
            if not no_ids:
                # print("no intersection")
                waypoints.append(next_pt)
            else:
                obs_f = self._all_obstacles[no_ids[0][1]]
                wp = self._compute_waypoints(src, obs_f)
                # print(f"o:{no_ids[0][1]}, {wp}")
                for q in wp:
                    if self._pos_key(q) not in explored:
                        to_explore.append(q)
        # print(len(explored))
        return waypoints    
    
    def _compute_waypoints(self, src, obs):
        if self._is_circle(obs):
            wp = circle_waypoints(src, obs, self.r_safe, self.r_vir)
        else:
            wp = box_waypoints(src, obs, self.r_safe, self.r_vir)
        return wp

    def _intersected_obstacles(self, src, tgt, obstacles, get_all=False):
        ids = []
        for i, O in enumerate(obstacles):
            if self._is_circle(O):
                is_col, t = segment_circle_collision(src, tgt, O, self.r_safe)
            else:
                is_col, t = segment_box_collision(src, tgt, O, self.r_safe)
            # print(f"oi:{i}, {flag}, {src}, {tgt}, {O}")
            if is_col and t > 0:
                ids.append((t, i))
                if not get_all: return ids
        ids.sort()
        return ids
    

    def _is_circle(self, O):
        return len(O) == 3

    def _pos_key(self, pos, eps=1e-3):
        return (round(pos[0] / eps), round(pos[1] / eps))

def segment_box_collision(p0, p1, segment, r_safe, eps=1e-8):
    (x0, y0), (x1, y1) = segment

    xm = (x0 + x1) * 0.5
    ym = (y0 + y1) * 0.5
    dxs, dys = x1 - x0, y1 - y0
    L = math.hypot(dxs, dys)

    if L == 0:
        u0 = (1.0, 0.0)
    else:
        u0 = (dxs/L, dys/L)
    u1 = (-u0[1], u0[0])
    e = (0.5*L + r_safe - eps, r_safe-eps)

    dp = p1[0] - p0[0], p1[1] - p0[1]
    # d = math.hypot(dp[0], dp[1])
    # dp = (dp[0]/d, dp[1]/d)
    pl = ((p0[0] - xm)*u0[0] + (p0[1] - ym)*u0[1], 
          (p0[0] - xm)*u1[0] + (p0[1] - ym)*u1[1])

    dpl = (dp[0]*u0[0] + dp[1]*u0[1], 
           dp[0]*u1[0] + dp[1]*u1[1])

    t_min = 0.0
    t_max = 1
    for i in range(2):
        if abs(dpl[i]) < 1e-8:
            if pl[i] < -e[i] or pl[i] > e[i]: return False, 0
        else:
            ood = 1.0 / dpl[i]
            t1 = (-e[i] - pl[i]) * ood
            t2 = ( e[i] - pl[i]) * ood
            if t1 > t2: t1, t2 = t2, t1            
            t_min = max(t_min, t1)
            t_max = min(t_max, t2)
            if t_min > t_max: return False, 0
    return True, t_min


def box_waypoints(p, segment, r_safe, r_vir, eps=1e-6):
    (x0, y0), (x1, y1) = segment
    R = max(r_safe, eps) + r_vir

    dx = x1 - x0
    dy = y1 - y0
    L = math.hypot(dx, dy)
    
    ux, uy = (dx/L)*R, (dy/L)*R
    vx, vy = -uy, ux
    all_wp = [
        (x0 - ux + vx, y0 - uy + vy), 
        (x0 - ux - vx, y0 - uy - vy),
        (x1 + ux - vx, y1 + uy - vy), 
        (x1 + ux + vx, y1 + uy + vy)
    ]
    # return all_wp
    all_dwp = [(wp[0] - p[0], wp[1] - p[1]) for wp in all_wp]
    wp = []
    for i in range(4):
        dx1, dy1 = all_dwp[i]
        d = math.hypot(dx1, dy1)
        # print('d', d, d < 1e-8)
        if d < eps:
            # print((i-1)%4, (i+1)%4)
            wp.append(all_wp[(i-1)%4])
            wp.append(all_wp[(i+1)%4])
            return wp
    
    for i in range(4):
        dx1, dy1 = all_dwp[i]
        n = 0
        for j in range(4):
            if i == j: continue
            dx2, dy2 = all_dwp[j]
            cross = dx1*dy2 - dx2*dy1
            if cross >= 0:
                n += 1
        if n == 0 or n == 3:
            wp.append(all_wp[i])
        if len(wp) == 2:
            break
    return wp


def segment_circle_collision(p0, p1, circle, r_safe=0, eps=1e-8):
    cx, cy, r = circle
    x0, y0 = p0[0], p0[1]
    x1, y1 = p1[0], p1[1]
    R = r + r_safe - eps

    mx, my = x0 - cx, y0 - cy
    dx, dy = x1 - x0, y1 - y0
    d = math.hypot(dx, dy)
    dx /= d
    dy /= d
    
    b = mx*dx + my*dy
    c = mx**2 + my**2 - R**2
    if c > 0 and b > 0: return False, 0
    
    discr = b**2 - c
    if discr < 0: return False, 0
    
    t = max(-b - math.sqrt(discr), 0)
    return t <= d, min(t, d)/d

def circle_waypoints(p, circle, r_safe, r_vir, dv=0.1):
    c, r = circle[:2], circle[2]
    r = r + r_safe
    
    dx, dy = c[0] - p[0], c[1] - p[1]
    d = math.hypot(dx, dy)
    if d < r: return []

    sth = min(r/d, 1.0)
    cth = math.sqrt(1 - sth**2)

    dx /= d
    dy /= d
    dt1 = (cth*dx + sth*dy, -sth*dx + cth*dy)
    dt2 = (cth*dx - sth*dy, sth*dx + cth*dy)

    R = r*(1 + dv) + r_vir
    b = -d*cth
    c = d**2- R**2
    discr = b**2 - c
    h = -b + math.sqrt(discr)
    wp1 = (p[0] + h*dt1[0], p[1] + h*dt1[1])
    wp2 = (p[0] + h*dt2[0], p[1] + h*dt2[1])
    return wp1, wp2

def distance(p0, p1):
    return math.hypot(p1[0] - p0[0], p1[1] - p0[1])
