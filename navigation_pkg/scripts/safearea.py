import math
import numpy as np
import scipy.optimize as opt

class Polytope:
    def __init__(self, A=[], b=[]):
        self.A = np.asarray(A)
        self.b = np.asarray(b)
        
    def __len__(self):
        return len(self.A)

class SafeCorridor:
    def __init__(self, r_safe=0.2, Lmax=0.5, ds=0.1, h=0.5):
        self.r_safe = r_safe
        self.Lmax = Lmax
        self.ds = ds
        self.h = h

    def __call__(self, path, obstacles):
        P = self._resample_path(path)
        
        P_poly = []
        for segment in zip(P[:-1], P[1:]):
            A, b = bounding_rectangle(segment, self.h, self.ds)
            Ao, bo = self._obstable_boundary(segment, obstacles)
            A.extend(Ao)
            b.extend(bo)
            poly = Polytope(A, b)
            P_poly.append(poly)
        return P, P_poly

    def _obstable_boundary(self, segment, obstacles):
        A, b = [], []
        for O in obstacles:
            if _is_circle(O):
                c1, c2 = closest_pt_segment_circle(segment, O)
            else:
                c1, c2 = closest_pt_segment_segment(segment, O)
            _A, _b = point_point_boundary(c1, c2, k=1)
            A.append(_A)
            b.append(_b - self.r_safe)
        return A, b

    def _resample_path(self, path):
        resampled_path = []
        for p0, p1 in zip(path[:-1], path[1:]):
            dp = (p1[0] - p0[0], p1[1] - p0[1])
            L = math.hypot(*dp)
            n = math.floor(L/self.Lmax)
            # print(n, L, L//self.Lmax)
            for i in range(n+1):
                l = i*self.Lmax/L
                p = (p0[0] + l*dp[0], p0[1] + l*dp[1])
                resampled_path.append(p)
        resampled_path.append(path[-1])
        return resampled_path
    
    # def _resample_path(self, path):
    #     distances = [0.0]
    #     for i in range(1, len(path)):
    #         dx = path[i][0] - path[i-1][0]
    #         dy = path[i][1] - path[i-1][1]
    #         distances.append(distances[-1] + math.hypot(dx, dy))

    #     total_dist = distances[-1]
    #     N = math.ceil(total_dist/self.Lmax)
    #     resampled_path = [path[0]]
        
    #     curr_dist = self.Lmax
    #     idx = 1        
    #     for _ in range(1, N):
    #         while idx < len(path) and distances[idx] < curr_dist:
    #             idx += 1
            
    #         prev_d = distances[idx-1]
    #         next_d = distances[idx]
    #         p0 = path[idx-1]
    #         p1 = path[idx]

    #         ratio = (curr_dist - prev_d) / (next_d - prev_d) if next_d != prev_d else 0
            
    #         x = p0[0] + ratio * (p1[0] - p0[0])
    #         y = p0[1] + ratio * (p1[1] - p0[1])
    #         resampled_path.append((x, y))
            
    #         curr_dist += self.Lmax
    #     resampled_path.append(path[-1])
    #     return resampled_path

def closest_pt_segment_segment(segment1, segment2, eps=1e-6):
    p1, q1 = segment1
    p2, q2 = segment2
    d1 = (q1[0] - p1[0], q1[1] - p1[1])
    d2 = (q2[0] - p2[0], q2[1] - p2[1])
    r = (p1[0] - p2[0], p1[1] - p2[1])
    a = d1[0]**2 + d1[1]**2
    e = d2[0]**2 + d2[1]**2
    f = d2[0]*r[0] + d2[1]*r[1]
    
    if a <= eps and e <= eps:
        s = t = 0.0
        c1 = p1
        c2 = p2 
        return c1, c2
    if a < eps:
        s = 0.0
        t = max(min(f/e, 1.0), 0.0)
    else:
        c = d1[0]*r[0] + d1[1]*r[1]
        if e < eps:
            t = 0.0
            s = max(min(-c/a, 1.0), 0.0)
        else:
            b = d1[0]*d2[0] + d1[1]*d2[1]
            denom = a*e - b*b
            if denom != 0.0:
                s = max(min((b*f - c*e)/denom, 1.0), 0.0)
            else:
                s = 0.0
            t = (b*s + f)/e
            if t < 0.0:
                t = 0.0
                s = max(min(-c/a, 1.0), 0.0)
            elif t > 1.0:
                t = 1.0
                s = max(min((b-c)/a, 1.0), 0.0)
    c1 = (p1[0] + d1[0]*s, p1[1] + d1[1]*s)
    c2 = (p2[0] + d2[0]*t, p2[1] + d2[1]*t) 
    return c1, c2

def closest_pt_segment_circle(segment, circle, eps=1e-6):
    c, r = circle[:2], circle[2]
    p, q = segment
    
    d1 = (q[0] - p[0], q[1] - p[1])
    a = d1[0]**2 + d1[1]**2
    
    if a <= eps and r <= eps:
        s = t = 0.0
        c1 = p
        c2 = q
        return c1, c2
    if a < eps:
        s = 0.0
    else:
        s = ((c[0] - p[0])*d1[0] + (c[1] - p[1])*d1[1]) / a
        s = max(min(s, 1.0), 0.0)

    c1 = (p[0] + d1[0]*s, p[1] + d1[1]*s)
    
    d2 = (c1[0] - c[0], c1[1] - c[1])
    t = math.atan2(d2[1], d2[0])
    c2 = (r*math.cos(t) + c[0], r*math.sin(t) + c[1])
    return c1, c2

def bounding_rectangle(segment, h, ds):
    p0, p1 = segment
    dx, dy = p1[0] - p0[0], p1[1] - p0[1]
    l = math.hypot(dx, dy)
    dx /= l
    dy /= l

    A = [
        (-dy, dx),  #t
        (dy, -dx),  #b
        (dx, dy),   #r
        (-dx, -dy), #l
    ]
    b = [
        A[0][0]*p0[0] + A[0][1]*p0[1] + h,
        A[1][0]*p0[0] + A[1][1]*p0[1] + h,
        A[2][0]*p1[0] + A[2][1]*p1[1] + ds,
        A[3][0]*p0[0] + A[3][1]*p0[1] + ds,
    ]
    return A, b

def point_point_boundary(p1, p2, k=0):
    d1 = (p2[0] - p1[0], p2[1] - p1[1])
    d2 = (p2[0] + p1[0], p2[1] + p1[1])
    l = math.hypot(*d1)
    A = (d1[0]/l, d1[1]/l)
    b = (d1[0]*d2[0] + d1[1]*d2[1])*0.5/l + k*l*0.5
    return A, b

def _is_circle(O):
    return len(O) == 3


class SafeArea:
    def __init__(self, r_safe=0.0, xlim=[-5,5], ylim=[-5,5]):
        self.xlim, self.ylim = xlim, ylim
        self.r_safe = r_safe

    def __call__(self, pos, obstacles):
        A = [(-1,0), (1,0), (0,-1),(0,1)]
        b = [-self.xlim[0] -self.r_safe, 
             self.xlim[1] - self.r_safe, 
             -self.ylim[0] - self.r_safe, 
             self.ylim[1] - self.r_safe]
        
        for O in obstacles:
            if _is_circle(O):
                q = closest_pt_point_circle(pos, O)
            else:
                q = closest_pt_point_segment(pos, O)
            _A, _b = point_point_boundary(pos, q, k=1)
            A.append(_A)
            b.append(_b - self.r_safe)
        return Polytope(A, b)
        

def closest_pt_point_segment(p, segment):
    a, b = segment
    ab = (b[0] - a[0], b[1] - a[1])
    d = ab[0]**2 + ab[1]**2
    t = ((p[0] - a[0])*ab[0] + (p[1] - a[1])*ab[1])/d
    t = max(min(t, 1.0), 0.0)
    q = (a[0] + ab[0]*t, a[1] + ab[1]*t)
    return q

def closest_pt_point_circle(p, circle):
    c, r = circle[:2], circle[2]
    cp = (p[0] - c[0], p[1] - c[1])
    t = math.atan2(cp[1], cp[0])
    q = (r*math.cos(t) + c[0], r*math.sin(t) + c[1])
    return q