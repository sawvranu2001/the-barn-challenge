import math

class ObstacleDetector:
    '''
    A line segment extraction algorithm using laser data based on seeded region growing
    (with circle extraction)
    '''
    def __init__(
        self, 
        delta_angle, min_angle, 

        eps_l=0.03, 
        dk=5, 
        s_num=6, 
        s_min=3, 
        p_min=10, 
        l_min=0.6, 
        si_max_l=0.005,
        
        eps_c=0.03,
        r_max=2.5,
        si_max_c=0.005,
    ):
        self.delta_angle = delta_angle
        self.min_angle = min_angle

        self.eps_l = eps_l
        self.dk = dk
        self.s_num = s_num
        self.s_min = min(s_num, s_min)
        self.p_min = p_min
        self.l_min = l_min
        self.si_max_l = si_max_l
        
        self.eps_c = min(eps_l, eps_c)
        self.r_max = r_max
        self.si_max_c = si_max_c

    def _cartesian(self, all_R):
        P, R, a = [], [], []
        for i, r in enumerate(all_R):
            if math.isfinite(r):
                R.append(r)
                a.append(i*self.delta_angle + self.min_angle)
                P.append((r*math.cos(a[-1]), r*math.sin(a[-1])))
        return P, R, a

    def __call__(self, all_ranges):
        points, ranges, angles = self._cartesian(all_ranges)

        res = {'points':points}
        res['lines'] = self.line_extraction(points, ranges)
        res['circles'] = self.circle_extraction(points, *res['lines'])
        # print(len(points))
        return res
    
    def circle_extraction(self, points, Fl, Idl, BPl):
        Fc = []
        Idc = []
        i = 0
        while i < len(Fl):
            flag, F = self._seed_circle_detection(*Idl[i], Fl[i], points)
            # print(i, Idl[i], flag)
            if flag:
                Fc.append(F)
                Idc.append(Idl[i])

                Fl.pop(i)
                Idl.pop(i)
                BPl.pop(i)
                continue
            i += 1
        # print('l: ', len(Fl), Idl)
        # print('c: ', len(Fc), Idc)
        Fc, Idc = self._circle_overlap_processing(Fc, Idc, points)
        # print('c: ', len(Fc), Idc)
        return  Fc, Idc

    def _circle_overlap_processing(self, Fc, Idc, points):
        i = 0
        while i < len(Fc)-1:
            mi, ni = Idc[i]
            j = i + 1
            while j < len(Fc):
                mj, nj = Idc[j]
                si = circle_similarity(Fc[i], Fc[j])
                # print(f'{i}:{Idc[i]}, {j}:{Idc[j]}, {1-si}')
                if 1 - si < self.si_max_c:
                    mi = min(mi, mj)
                    ni = max(ni, nj)
                    F,_ = circle_fit(points[mi:ni])
                    
                    Fc[i] = F
                    Idc[i] = (mi, ni)
                    # print(f'up: {i}:{Idc[i]}')
                    Idc.pop(j)
                    Fc.pop(j)
                    continue
                j += 1
            # print(i, len(Fc), Idc, '\n')
            i += 1

        return Fc, Idc

    def _seed_circle_detection(self, i, j, Fl, points):
        flag = True
        Fc, _ = circle_fit(points[i:j])
        # print((i,j), Fc[2])
        flag = Fc[-1] < self.r_max
        if flag:
            for k in range(i, j):
                pk = points[k]
                d1 = line_prep_dist(Fl, pk)
                d2 = circle_prep_dist(Fc, pk)
                if d2 > self.eps_c or d2 > d1:
                    flag = False
                    break
        # print((i,j), k, d1, d2)
        if not flag:
            Fc = None
        return flag, Fc
        

    def line_extraction(self, points, ranges):
        Fl = []
        Idl = []
        i = 0
        while i < len(ranges) - max(self.p_min, self.s_num):
            flag, j, Fij, s_cache = self._seed_segment_detection(i, points, ranges)
            # print(f'{(i,j)}', end=' ')
            if flag:
                m, n, Fmn = self._segment_region_growing(i, j, Fij, s_cache, points, ranges)
                Ll = line_para_dist(Fmn, points[m], points[n-1])
                Pl = n - m
                # print(f'{(m,n)} {Ll:.4f} {Pl}')
                if Ll >= self.l_min and Pl >= self.p_min:
                    Idl.append((m,n))
                    Fl.append(Fmn)
                i = n
            else:
                i += 1
            # print()
            # Fs.append(F)
        # print(len(Fs), Ids)
        Fl, Idl = self._segment_overlap_processing(Fl, Idl, points)
        # print(len(Fl), Idl)
        BPl = self._segment_breakpoints(Fl, Idl, points)
        return Fl, Idl, BPl
    
    def _segment_breakpoints(self, Fl, Ids, points):
        BPl = []
        for (a, b, c), (m, n) in zip(Fl, Ids):
            bps = []
            for k in (m, n-1):
                x, y = points[k]
                xp = x - a*(a*x + b*y + c)
                yp = y - b*(a*x + b*y + c)
                bps.append((xp, yp))
            BPl.append(bps)
        return BPl
        
    def _segment_overlap_processing(self, Fl, Idl, points):
        # sort by start index
        Idl, Fl = zip(*sorted(zip(Idl, Fl))) # n0<=n1<=n2<=n3...
        Idl, Fl = list(Idl), list(Fl)
        # print(len(Fs), Ids)

        # merge collinear segments
        i = 0
        while i < len(Fl)-1:
            mi, ni = Idl[i]
            j = i + 1
            while j < len(Fl):
                mj, nj = Idl[j]
                # print(f'{i}:{Ids[i]}, {j}:{Ids[j]}, {mj, ni}')
                if ni > mj:

                    si = line_similarity(Fl[i], Fl[j])
                    # print(f'{1-si:.4f}', end=' ')
                    if 1 - si < self.si_max_l:
                        mi = min(mi, mj)
                        ni = max(ni, nj)
                        F,_ = line_fit(points[mi:ni])
                        
                        Fl[i] = F
                        Idl[i] = (mi, ni)
                        # print(f'up: {i}:{Ids[i]}')
                        Idl.pop(j)
                        Fl.pop(j)
                        continue
                    # print(i,j, len(Fs), Ids, '\n')
                j += 1
            i += 1
        # print(len(Fs), Ids)
        # return Fs, Ids
        # for i in range(len(Fs)-1):

        # split non-collinear segments
        i = 0
        while i < len(Fl)-1:
            j = i + 1
            mi, ni = Idl[i]
            mj, nj = Idl[j]
            # print(f'{i}: {Ids[i]}, {Ids[j]}')
            if ni > nj:
                Idl.pop(j)
                Fl.pop(j)
                continue
            if ni > mj:
                # print(f'k:{(mj, ni)}', end=': ')
                for k in range(mj, ni):
                    pk = points[k]
                    dik = line_prep_dist(Fl[i], pk)
                    djk = line_prep_dist(Fl[j], pk)
                    # print(f'{k}:({dik:.4},{djk:.4}) {dik > djk}', end=", ")
                    if dik >= djk:
                        break

                if (k - mi) < self.p_min:
                    # print(f'\ndrp i: {i} {Ids[i]}')
                    Idl.pop(i)
                    Fl.pop(i)
                    continue

                elif (nj - k) < self.p_min:
                    # print(f'\ndrp j: {j} {Ids[j]}')
                    Idl.pop(j)
                    Fl.pop(j)
                    continue

                else:
                    ni = mj = k
                    Idl[i] = (mi, ni)
                    Fl[i],_ = line_fit(points[mi:ni])
                    Idl[j] = (mj, nj)
                    Fl[j],_ = line_fit(points[mj:nj])

                # print(f'up: {Ids[i]}, {Ids[j]}\n{len(Fs)}, {Ids},\n')
            i += 1
        return Fl, Idl

    def _segment_region_growing(self, i, j, F, s_cache, points, ranges):      
        # print(f'{(i,j)}')  
        # forward
        # print('f', end=':')
        Np = len(points)
        while j < Np:
            pf = points[j]
            d1 = line_prep_dist(F, pf)

            D = self.dk * ranges[j-1] * self.delta_angle + self.eps_l
            d2 = math.hypot(points[j-1][0] - pf[0], points[j-1][1] - pf[1])
            # print(f"{j}: {d1:.4f} {d2:.4f}", end=', ')
            if d1 < self.eps_l and d2 < D:
                F, s_cache = line_fit(pf, *s_cache)
            else:
                break
            j += 1

        # backward
        # print('\nb', end=':')
        while i > 0:
            pb = points[i-1]
            d1 = line_prep_dist(F, pb)

            D = self.dk * ranges[i] * self.delta_angle + self.eps_l
            d2 = math.hypot(points[i][0] - pb[0], points[i][1] - pb[1])
            # print(f"{i-1}: {d1:.4f} {d2:.4f}", end=', ')
            if d1 < self.eps_l and d2 < D:
                F, s_cache = line_fit(pb, *s_cache)
            else:
                break
            i -= 1
        # print(f'\n{(i,j)}\n')
        return i, j, F

    def _seed_segment_detection(self, i, points, ranges):
        # print(len(points))
        Np = len(points)
        j = min(i + self.s_num, Np)
        # print((i,j), end=' ')
        for k in range(i, j):
            D = self.dk * ranges[k] * self.delta_angle + self.eps_l
            d1 = math.hypot(points[k][0] - points[k+1][0], points[k][1] - points[k+1][1])
            if d1 > D:
                # print(f'{k}, {d1:.4f}, {D:.4f}', end=' ')
                break
        j = k
        # print((i,j))
        flag = True
        if j - i >= self.s_min:
            F, s_cache = line_fit(points[i:j])
            for k in range(i, j):
                pk = points[k]
                d2 = line_prep_dist(F, pk)
                if d2 > self.eps_l:
                    flag = False
                    break
        else:
            F, s_cache = None, None
            flag = False
        # print(f'\n{(i,j)}, {flag}')
        return flag, j, F, s_cache



def line_fit(P, n=0, sx=0, sy=0, sxx=0, syy=0, sxy=0):
    if isinstance(P[0], (list,tuple)):
        n += len(P)
        sx += sum([p[0] for p in P])
        sy += sum([p[1] for p in P])
        sxx += sum([p[0]**2 for p in P])
        syy += sum([p[1]**2 for p in P])
        sxy += sum([p[0]*p[1] for p in P])
    else:
        p = P
        n += 1
        sx += p[0]
        sy += p[1]
        sxx += p[0]**2
        syy += p[1]**2
        sxy += p[0]*p[1]

    dxx = n*sxx - sx**2 
    dyy = n*syy - sy**2
    dxy = n*sxy - sx*sy
    theta = 0.5 * math.atan2(2*dxy, dxx - dyy)
    a = -math.sin(theta)
    b = math.cos(theta)
    c = - (a*sx + b*sy)/n
    
    return (a, b, c), (n, sx, sy, sxx, syy, sxy)

def line_intersect(F, F0):
    a, b, c = F
    a0, b0 = math.sin(F0), -math.cos(F0)
    d = (a0*b - a*b0)
    x = b0*c/d
    y = -a0*c/d
    return (x, y)

def line_prep_dist(F, p):
    a, b, c = F
    x, y = p
    return abs(a*x + b*y + c)

def line_para_dist(F, p_ref, p):
    a, b, c = F
    sth, cth = -a, b

    dx = p[0] - p_ref[0]
    dy = p[1] - p_ref[1]
    k = cth*dx + sth*dy
    return abs(k)

def line_similarity(F1, F2):
    a1, b1, c1 = F1
    a2, b2, c2 = F2
    si = abs(a1*a2 + b1*b2)
    return si


def circle_fit(P, n=0, sx=0, sy=0, sxx=0, syy=0, sxy=0, sxz=0, syz=0):
    if isinstance(P[0], (list,tuple)):
        n += len(P)
        sx += sum([p[0] for p in P])
        sy += sum([p[1] for p in P])
        sxx += sum([p[0]**2 for p in P])
        syy += sum([p[1]**2 for p in P])
        sxy += sum([p[0]*p[1] for p in P])
        sxz += sum([p[0]*(p[0]**2 + p[1]**2) for p in P])
        syz += sum([p[1]*(p[0]**2 + p[1]**2) for p in P])
    else:
        p = P
        n += 1
        sx += p[0]
        sy += p[1]
        sxx += p[0]**2
        syy += p[1]**2
        sxy += p[0]*p[1]
        sxz += p[0]*(p[0]**2 + p[1]**2)
        syz += p[1]*(p[0]**2 + p[1]**2)

    dxx = n*sxx - sx**2 
    dyy = n*syy - sy**2
    dxy = n*sxy - sx*sy
    dxz = n*sxz - sx*(sxx + syy)
    dyz = n*syz - sy*(sxx + syy)
    
    d = 2*(dxx*dyy - dxy**2)
    if abs(d) < 1e-6: # straight line
        return (0, 0, float('inf')), (n, sx, sy, sxx, syy, sxy, sxz, syz)
    a = (dxz*dyy - dyz*dxy) / d
    b = (dyz*dxx - dxz*dxy) / d
    c = (2*a*sx + 2*b*sy - sxx - syy)/n
    r = math.sqrt(a**2 + b**2 - c)
    
    return (a, b, r), (n, sx, sy, sxx, syy, sxy, sxz, syz)

def circle_prep_dist(F, p):
    a, b, r = F
    x, y = p
    return abs(math.sqrt((x - a)**2 + (y - b)**2) - r)

def circle_similarity(F1, F2):
    a1, b1, r1 = F1
    a2, b2, r2 = F2
    d = math.hypot(a1 - a2, b1 - b2)
    if d >= r1 + r2:
        return 0.0
    if d <= abs(r1 - r2):
        # Ai = math.pi * min(r1, r2)**2
        # A = math.pi * max(r1, r2)**2
        return 1.0 #Ai / A
    
    d1 = 0.5*(d**2 - r2**2 + r1**2)/d
    d2 = 0.5*(d**2 - r1**2 + r2**2)/d
    dr1 = max(-1.0, min(1.0, d1 / r1))
    dr2 = max(-1.0, min(1.0, d2 / r2))

    Ai = r1**2*math.acos(dr1) - d1*math.sqrt(max(0, r1**2-d1**2)) \
        + r2**2*math.acos(dr2) - d2*math.sqrt(max(0, r2**2-d2**2))
    A = math.pi * (r1**2 + r2**2) - Ai
    return Ai/A