import heapq
import math

class Node:
    def __init__(self, pos, hcost=0, gcost=0, pcost=0, parent=None, depth=0):
        self.pos = pos
        self.gcost = gcost
        self.hcost = hcost
        self.pcost = pcost
        self.parent = parent
        self.depth = depth

    @property
    def fcost(self):
        return self.gcost + self.hcost + self.pcost

    def __lt__(self, other):
        return self.fcost < other.fcost
    


class SmoothPath:
    def __init__(self, smoothing_distance=1.0, num_points=5):
        self.smoothing_distance = smoothing_distance
        self.num_points = num_points

    def __call__(self, waypoints):
        if len(waypoints) < 3:
            return waypoints # No turns to smooth

        s_path = [waypoints[0]] # Add the start node
        for i in range(1, len(waypoints) - 1):
            prev_wp = waypoints[i-1]
            curr_wp = waypoints[i]
            next_wp = waypoints[i+1]

            dir_back = (prev_wp[0] - curr_wp[0], prev_wp[1] - curr_wp[1])
            dist_prev = math.hypot(*dir_back)
            dir_back = (dir_back[0]/dist_prev, dir_back[1]/dist_prev)
            
            dir_forward = (next_wp[0] - curr_wp[0], next_wp[1] - curr_wp[1])
            dist_next = math.hypot(*dir_forward)
            dir_forward = (dir_forward[0]/dist_next, dir_forward[1]/dist_next)

            safe_dist = min(self.smoothing_distance, dist_prev * 0.4, dist_next * 0.4)
            p0 = (curr_wp[0] + dir_back[0]*safe_dist, 
                curr_wp[1] + dir_back[1]*safe_dist)
            p1 = (curr_wp[0] + dir_forward[0]*safe_dist, 
                curr_wp[1] + dir_forward[1]*safe_dist)
            
            turn_curve = self._quadratic_bezier(p0, curr_wp, p1)

            # s_path.append(p0)
            s_path.extend(turn_curve)

        s_path.append(waypoints[-1])
        return s_path

    def _quadratic_bezier(self, p0, p1, p2):
        """
        Formula: B(t) = (1-t)^2 * P0 + 2(1-t)t * P1 + t^2 * P2
        """
        curve = []
        for i in range(self.num_points+1):
            t = i/self.num_points
            x = (1 - t)**2 * p0[0] + 2*(1 - t)*t*p1[0] + t**2*p2[0]
            y = (1 - t)**2 * p0[1] + 2*(1 - t)*t*p1[1] + t**2*p2[1]
            curve.append((x,y))
        return curve

class GlobalPlanner:
    def __init__(self, r_safe=0.1, dv=0.1, wg=1.0, wh=1.0, wp=1.0, max_depth=float('inf')):
        self.r_safe = r_safe
        self.dv = dv
        self.wg = wg
        self.wh = wh
        self.wp = wp
        self.max_depth = max_depth
        
        self.prev_path_v0 = None

    def __call__(self, start, target, obstacles):
        start = self._resolve_start(start, target, obstacles)
        tgt_key = _pos_key(target)

        open_list = []
        closed_list = set()

        start_node = Node(pos=start, hcost=distance(start, target))
        heapq.heappush(open_list, start_node)

        while open_list:
            current = heapq.heappop(open_list)
            # print(f'{current.pos}, {current.depth}')
            closed_list.add(_pos_key(current.pos))

            if _pos_key(current.pos) == tgt_key or current.depth > self.max_depth:
                path = []
                while current:
                    path.append(current.pos)
                    current = current.parent
                path = path[::-1]
                
                v0 = (path[1][0] - path[0][0], path[1][1] - path[0][1])
                d0 = math.hypot(*v0)
                # print('v0', math.degrees(math.atan2(v0[1], v0[0])))
                self.prev_path_v0 = (v0[0]/d0, v0[1]/d0)
                return path

            waypoints = self._generate_waypoints(current.pos, target, obstacles)
            
            for w_pos in waypoints:
                if _pos_key(w_pos) in closed_list:
                    continue

                hcost = self.wh * distance(w_pos, target)
                gcost = self.wg * distance(current.pos, w_pos) + current.gcost
                pcost = self._turn_penalty(w_pos, current) + current.pcost

                depth = current.depth + 1
                next_node = Node(pos=w_pos, hcost=hcost, gcost=gcost, pcost=pcost, parent=current, depth=depth)
                # print(next_node.fcost)
                heapq.heappush(open_list, next_node)
        return None 
    
    def _turn_penalty(self, waypoint, current):
        v_cw = (waypoint[0] - current.pos[0], waypoint[1] - current.pos[1])
        d_cw = math.hypot(*v_cw)
        
        if current.depth == 0:
            if self.prev_path_v0 is not None:
                dot = (self.prev_path_v0[0]*v_cw[0] + self.prev_path_v0[1]*v_cw[1]) / d_cw
                pcost = self.wp * (1 - dot)
                return pcost
            else:
                return 0
        
        v_pc = (current.pos[0] - current.parent.pos[0], current.pos[1] - current.parent.pos[1])
        d_pc = math.hypot(*v_pc)        
        if d_pc < 1e-6 or d_cw < 1e-6: return 0

        dot = (v_pc[0]*v_cw[0] + v_pc[1]*v_cw[1]) / (d_pc * d_cw)
        pcost = self.wp * (1 - dot)

        # angle = math.acos(max(-1.0, min(1.0, dot))        
        # print('wc', current.parent.pos, current.pos, waypoint, d2, math.degrees(angle), pcost)
        return pcost

    def _generate_waypoints(self, src, tgt, obstacles):
        waypoints = []
        to_explore = [tgt]
        explored = set() 
        while to_explore:
            next_pt = to_explore.pop(0)
            # print(next_pt, end=': ')
            explored.add(_pos_key(next_pt))
            no_ids = self._intersected_obstacles(src, next_pt, obstacles, True)
            # print(src, no_ids)
            # waypoints.append(next_pt)
            if not no_ids:
                # print("no intersection")
                waypoints.append(next_pt)
            else:
                obs_f = obstacles[no_ids[0][1]]
                wp = self._compute_waypoints(src, obs_f)
                # print(f"o:{no_ids}, {wp}")
                for q in wp:
                    if _pos_key(q) not in explored:
                        to_explore.append(q)
        # print(len(explored))
        return waypoints    
    
    def _compute_waypoints(self, src, obs):
        if _is_circle(obs):
            wp = circle_waypoints(src, obs, self.r_safe, self.dv)
        else:
            wp = capsule_waypoints(src, obs, self.r_safe, self.dv)
        return wp

    def _intersected_obstacles(self, src, tgt, obstacles, get_all=False):
        ids = []
        for i, O in enumerate(obstacles):
            if _is_circle(O):
                is_col, t = segment_circle_collision(src, tgt, O, self.r_safe)
            else:
                is_col, t = segment_capsule_collision(src, tgt, O, self.r_safe)
            # print(f"oi:{i}, {is_col, t}, {src}, {tgt}, {O}")
            if is_col:# and t > 0:
                ids.append((t, i))
                if not get_all: return ids
        ids.sort()
        return ids
    
    def _resolve_start(self, start, target, obstacles):
        ids1 = self._intersected_obstacles(start, target, obstacles, True)
        # print(ids1)
        if ids1 and ids1[0][0] < 1e-8:
            print('warning! resolving start')
            d = (target[0] - start[0], target[1] - start[1])
            t0 = -1
            p = (start[0] + t0*d[0],  start[1] + t0*d[1])
            ids2 = self._intersected_obstacles(p, target, obstacles, True)
            t = 0.0
            for i in range(len(ids2)-1):
                if ids2[i][1] == ids1[0][1]:
                    # print(ids2[i], ids2[i+1])
                    t = ids2[i][0] - 1e-6
                    break
            # print(t, (t0 + t))
            start = (p[0] + t*(target[0] - p[0]),  p[1] + t*(target[1] - p[1]))
            
        return start


def _pos_key(pos, eps=1e-3):
    return (round(pos[0] / eps), round(pos[1] / eps))

def _is_circle(O):
    return len(O) == 3

def segment_capsule_collision(p0, p1, segment, r_safe, eps=1e-8):
    s0, s1 = segment
    c0 = (s0[0], s0[1], 0.0)
    c1 = (s1[0], s1[1], 0.0)
    R = r_safe - eps

    d = (s1[0] - s0[0], s1[1] - s0[1])
    m = (p0[0] - s0[0], p0[1] - s0[1])
    n = (p1[0] - p0[0], p1[1] - p0[1])
    md = m[0]*d[0] + m[1]*d[1]
    nd = n[0]*d[0] + n[1]*d[1]
    dd = d[0]**2 + d[1]**2
    
    if md < 0.0 and md + nd < 0.0: return segment_circle_collision(p0, p1, c0, r_safe)
    if md > dd and md + nd > dd: return segment_circle_collision(p0, p1, c1, r_safe)
    
    nn = n[0]**2 + n[1]**2
    mn = m[0]*n[0] + m[1]*n[1]
    mm = m[0]**2 + m[1]**2
    a = dd*nn - nd**2
    k = mm - R**2
    c = dd*k - md**2
    
    if abs(a) < 1e-8:
        if c > 0.0: return False, 0
        if md < 0.0: return segment_circle_collision(p0, p1, c0, r_safe)
        elif md > dd: return segment_circle_collision(p0, p1, c1, r_safe)
        else: return True, 0.0
    
    b = dd*mn - nd*md
    discr = b**2 - a*c
    if discr < 0.0: return False, 0
    
    t = (-b - math.sqrt(discr))/a
    # print(md + t*nd, dd, t)
    if md + t*nd < 0.0: return segment_circle_collision(p0, p1, c0, r_safe)
    elif md + t*nd > dd: return segment_circle_collision(p0, p1, c1, r_safe)
    if t < 0.0 or t > 1.0: return False, 0
    return True, t

def capsule_waypoints(p, segment, r_safe, dv=0.1):
    (x0, y0), (x1, y1) = segment
    all_wp = []
    all_wp.extend(circle_waypoints(p, (x0, y0, 0.0), r_safe, dv))
    all_wp.extend(circle_waypoints(p, (x1, y1, 0.0), r_safe, dv))
    if len(all_wp) != 4: return []
    # return all_wp
    wp = []
    for i in range(4):
        is_col, t = segment_capsule_collision(p, all_wp[i], segment, r_safe)
        # print(all_wp[i], is_col, t)
        if not is_col:
            wp.append(all_wp[i])
    return wp
    # lh = all_wp[0]
    # rh = all_wp[0]
    # for wp in all_wp[1:]:
    #     vx, vy = wp[0] - p[0], wp[1] - p[1]
    #     lx, ly = lh[0] - p[0], lh[1] - p[1]
    #     rx, ry = rh[0] - p[0], rh[1] - p[1]
        
    #     l_cross = lx*vy - ly*vx
    #     r_cross = rx*vy - ry*vx
    #     if l_cross > 0: lh = wp
    #     if r_cross < 0: rh = wp
    # return [rh, lh]


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

def circle_waypoints(p, circle, r_safe, dv=0.1):
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

    R = r*(1 + dv)
    b = -d*cth
    c = d**2- R**2
    discr = b**2 - c
    h = -b + math.sqrt(discr)
    wp1 = (p[0] + h*dt1[0], p[1] + h*dt1[1])
    wp2 = (p[0] + h*dt2[0], p[1] + h*dt2[1])
    return [wp1, wp2]

def distance(p0, p1):
    return math.hypot(p1[0] - p0[0], p1[1] - p0[1])



# def segment_box_collision(p0, p1, segment, r_safe, eps=1e-8):
#     (x0, y0), (x1, y1) = segment

#     xm = (x0 + x1) * 0.5
#     ym = (y0 + y1) * 0.5
#     dxs, dys = x1 - x0, y1 - y0
#     L = math.hypot(dxs, dys)

#     if L == 0:
#         u0 = (1.0, 0.0)
#     else:
#         u0 = (dxs/L, dys/L)
#     u1 = (-u0[1], u0[0])
#     e = (0.5*L + r_safe - eps, r_safe-eps)

#     dp = p1[0] - p0[0], p1[1] - p0[1]
#     # d = math.hypot(dp[0], dp[1])
#     # dp = (dp[0]/d, dp[1]/d)
#     pl = ((p0[0] - xm)*u0[0] + (p0[1] - ym)*u0[1], 
#           (p0[0] - xm)*u1[0] + (p0[1] - ym)*u1[1])

#     dpl = (dp[0]*u0[0] + dp[1]*u0[1], 
#            dp[0]*u1[0] + dp[1]*u1[1])

#     t_min = 0.0
#     t_max = 1
#     for i in range(2):
#         if abs(dpl[i]) < 1e-8:
#             if pl[i] < -e[i] or pl[i] > e[i]: return False, 0
#         else:
#             ood = 1.0 / dpl[i]
#             t1 = (-e[i] - pl[i]) * ood
#             t2 = ( e[i] - pl[i]) * ood
#             if t1 > t2: t1, t2 = t2, t1            
#             t_min = max(t_min, t1)
#             t_max = min(t_max, t2)
#             if t_min > t_max: return False, 0
#     return True, t_min

# def box_waypoints(p, segment, r_safe, eps=1e-6):
#     (x0, y0), (x1, y1) = segment
#     R = max(r_safe, eps)

#     dx = x1 - x0
#     dy = y1 - y0
#     L = math.hypot(dx, dy)
    
#     ux, uy = (dx/L)*R, (dy/L)*R
#     vx, vy = -uy, ux
#     all_wp = [
#         (x0 - ux + vx, y0 - uy + vy), 
#         (x0 - ux - vx, y0 - uy - vy),
#         (x1 + ux - vx, y1 + uy - vy), 
#         (x1 + ux + vx, y1 + uy + vy)
#     ]
#     # return all_wp
#     all_dwp = [(wp[0] - p[0], wp[1] - p[1]) for wp in all_wp]
#     wp = []
#     for i in range(4):
#         dx1, dy1 = all_dwp[i]
#         d = math.hypot(dx1, dy1)
#         # print('d', d, d < 1e-8)
#         if d < eps:
#             # print((i-1)%4, (i+1)%4)
#             wp.append(all_wp[(i-1)%4])
#             wp.append(all_wp[(i+1)%4])
#             return wp
    
#     for i in range(4):
#         dx1, dy1 = all_dwp[i]
#         n = 0
#         for j in range(4):
#             if i == j: continue
#             dx2, dy2 = all_dwp[j]
#             cross = dx1*dy2 - dx2*dy1
#             if cross >= 0:
#                 n += 1
#         if n == 0 or n == 3:
#             wp.append(all_wp[i])
#         if len(wp) == 2:
#             break
#     return wp