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
    def __init__(self, r_safe=0.1, wg=1.0, wh=1.0, max_depth=float('inf')):
        self.r_safe = r_safe
        self.wg = wg
        self.wh = wh
        self.max_depth = max_depth

    def __call__(self, start, target, obstacles):
        # Extract lines and circles based on input structure
        Fl, Idl, BPl = obstacles.get('lines', ([], [], []))
        Fc, Idc = obstacles.get('circles', ([], []))

        # Combine base primitives
        all_obstacles = BPl + Fc

        tgt_key = self._pos_key(target)

        open_list = []
        closed_list = set()

        start_node = Node(pos=start, hcost=distance(start, target))
        heapq.heappush(open_list, start_node)

        while open_list:
            current = heapq.heappop(open_list)
            closed_list.add(self._pos_key(current.pos))

            if self._pos_key(current.pos) == tgt_key or current.depth > self.max_depth:
                # print(f'{current.pos}, {current.depth}, {len(obstacles)}')
                path = []
                while current:
                    path.append(current.pos)
                    current = current.parent
                return path[::-1] # Reverse to get Start -> Target

            neighbours = self._compute_neighbours(current.pos, target, all_obstacles)

            for pos_n in neighbours:
                if self._pos_key(pos_n) in closed_list:
                    continue

                hcost = distance(pos_n, target) * self.wh
                gcost = distance(current.pos, pos_n) * self.wg + current.gcost
                depth = current.depth + 1
                next_node = Node(pos=pos_n, hcost=hcost, gcost=gcost, parent=current, depth=depth)
                heapq.heappush(open_list, next_node)

        return None

    def _compute_neighbours(self, src, tgt, obstacles):
        if self._is_segment_clear(src, tgt, obstacles):
            return [tgt]

        No = len(obstacles)
        S_Ft = []
        for i in range(No):
            O = obstacles[i]
            if len(O) == 3:
                Ft = circle_tangent(src, O, self.r_safe)
            else:
                Ft = segment_tangent(src, O, self.r_safe)
            S_Ft.append(Ft)

        neighbors = []
        for i, s_Ft in enumerate(S_Ft):
            Oi = obstacles[i]
            if len(Oi) == 3:
                t_Ft = circle_tangent(tgt, Oi, self.r_safe)
            else:
                t_Ft = segment_tangent(tgt, Oi, self.r_safe)
            
            # print(f'\nobs:{i}')
            for k1, F1 in enumerate(s_Ft):
                t_max = 0
                w_pt = None
                for k2, F2 in enumerate(t_Ft):
                    pt = line_intersect(F1, F2)
                    if pt is not None:
                        t = valid_intersect(src, tgt, pt)
                        # print(f"t:{(k1, k2)}: {t:.4f}")
                        if (0 < t <= 1) and (t > t_max):
                            t_max = t
                            w_pt = pt
                # print(f"t:{k1}: {t_max:.4f} {w_pt}", end=' ')
                if (w_pt is not None and 
                    self._is_segment_clear(src, w_pt, obstacles, exclude=i)
                    ):
                        # print(f"intersect")
                        neighbors.append(w_pt)
        return neighbors

    def _is_segment_clear(self, src, tgt, obstacles, exclude=None):
        for i, O in enumerate(obstacles):
            if exclude is not None and i == exclude:
                continue

            if len(O) == 3:
                flag = segment_circle_collision(src, tgt, O, self.r_safe)
            else:
                flag = segment_segment_collision(src, tgt, O, self.r_safe)
            # print(f"oi:{i}, {flag}, {src}, {tgt}, {O}")
            if flag:
                return False
        return True

    def _pos_key(self, pos, eps=1e-3):
        return (round(pos[0] / eps), round(pos[1] / eps))

def valid_intersect(src, tgt, intersect):
    v_st = (tgt[0] - src[0], tgt[1] - src[1])
    v_si = (intersect[0] - src[0], intersect[1] - src[1])
    dot = v_st[0]*v_si[0] + v_st[1]*v_si[1]
    d_sq = v_st[0]**2 + v_st[1]**2
    t = dot / d_sq 
    # print(t, dot, 0 < t <= 1, dot > 0)
    return t #0 < t <= 1 #dot > 0

def line_intersect(F1, F2):
    a1, b1, c1 = F1
    a2, b2, c2 = F2
    det = a1*b2 - a2*b1

    if abs(det) < 1e-8:
        return None

    x = (b1*c2 - b2*c1) / det
    y = (a2*c1 - a1*c2) / det
    return (x, y)

def segment_segment_collision(p0, p1, segment, r_safe, eps=1e-6):
    (cx0, cy0), (cx1, cy1) = segment
    R = max(r_safe, eps)

    dx, dy = p1[0] - p0[0], p1[1] - p0[1]
    dcx, dcy = cx1 - cx0, cy1 - cy0
    rx, ry = p0[0] - cx0, p0[1] - cy0
    
    a = dx**2 + dy**2
    e = dcx**2 + dcy**2
    f = dcx*rx + dcy*ry

    if a <= eps and e <= eps: 
        s, t = 0.0, 0.0
    elif a<=eps:
        s = 0.0
        t = max(0.0, min(1.0, f/e))
    
    else:
        c = dx*rx + dy*ry
        if e <= eps:
            t = 0.0
            s = max(0.0, min(1.0, -c/a))
        else:
            b = dx*dcx + dy*dcy
            denom = a*e - b**2
            if denom != 0:
                s = max(0.0, min(1.0, (b*f - c*e)/denom))
            else:
                s = 0
            t = (b*s + f)/e
            if t < 0:
                t = 0.0
                s = max(0.0, min(1.0, -c/a))
            elif t > 1:
                t = 1.0
                s = max(0.0, min(1.0, (b - c)/a))
            
    c1 = (p0[0] + dx*s, p0[1] + dy*s)
    c2 = (cx0 + dcx*t, cy0 + dcy*t)

    dc = (c1[0] - c2[0])**2 + (c1[1] - c2[1])**2
    return dc < R**2

def segment_tangent(p, segment, r_safe, eps=1e-6):
    (cx0, cy0), (cx1, cy1) = segment

    all_Ft = []
    all_Ft.extend(circle_tangent(p, (cx0, cy0, eps), r_safe))
    all_Ft.extend(circle_tangent(p, (cx1, cy1, eps), r_safe))
    if len(all_Ft) < 4: return []
    
    Ft = []
    for i in range(4):
        a1, b1, c1 = all_Ft[i]
        n = 0
        for j in range(4):
            if i == j: continue
            a2, b2, c2 = all_Ft[j]
            cross = a1*b2 - b1*a2
            if cross > 0:
                n += 1
        if n == 0 or n == 3:
            Ft.append((a1, b1, c1))
    return Ft

def segment_circle_collision(p0, p1, circle, r_safe):
    xc, yc, r = circle
    x0, y0 = p0[0], p0[1]
    x1, y1 = p1[0], p1[1]
    R = r + r_safe

    dxc0 = xc - x0
    dyc0 = yc - y0
    dx = x1 - x0
    dy = y1 - y0
    d_sq = dx**2 + dy**2

    if d_sq == 0:
        return dxc0**2+ dyc0**2 < R**2

    t = (dxc0*dx + dyc0*dy)/d_sq
    t = max(0.0, min(1.0, t))
    
    xp = x0 + t * dx
    yp = y0 + t * dy
    dr = (xp - xc)**2 + (yp - yc)**2
    return dr < R**2

def circle_tangent(p, circle, r_safe):
    c, r = circle[:2], circle[2]
    R = r + r_safe

    dx = c[0] - p[0]
    dy = c[1] - p[1]
    d = math.hypot(dx, dy)
    if d <= R: return []
    
    sth = min(R/d, 1.0)
    cth = math.sqrt(1 - sth**2)

    dx /= d
    dy /= d
    t1 = (cth*dx + sth*dy, -sth*dx + cth*dy)
    t2 = (cth*dx - sth*dy, sth*dx + cth*dy)

    Ft1 = (-t1[1], t1[0], t1[1]*p[0] - t1[0]*p[1])
    Ft2 = (-t2[1], t2[0], t2[1]*p[0] - t2[0]*p[1])
    
    return Ft1, Ft2

def distance(p0, p1):
    return math.hypot(p1[0] - p0[0], p1[1] - p0[1])
