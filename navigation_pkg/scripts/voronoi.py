import math
import numpy as np
import scipy.optimize as opt
from scipy.special import erfinv
from scipy.stats.distributions import chi2
from scipy.linalg import sqrtm
from scipy.spatial import ConvexHull, Delaunay
import cdd


class Cell:
    def __init__(self, pos=np.zeros(2), safety_radius=0):
        self.pos = pos
        self.safety_radius = safety_radius
        self.poly = None

class Obstacle:
    def __init__(self, points, lines, circles):
        self.points = np.array(points)

        Fl, Idl, BPl = lines
        self.segments = np.array(BPl)
        
        Fc, Idc = circles
        self.circles = np.array(Fc)
        
        self.N = len(self.segments) + len(self.circles)
        self.indices = [-4-i-1 for i in range(self.N)]


class Voronoi:
    def __init__(self, pos, safety_radius, xlim=[-5,5], ylim=[-5,5]):
        self.xlim, self.ylim = xlim, ylim
        
        self.cell = Cell(pos, safety_radius)
        self.obstacles = None
    
    def update_obstacles(self, obstacles):
        self.obstacles = Obstacle(**obstacles)
    
    def __call__(self):
        Ab, bb, idx = self.boundary()
        Al, bl = point_segment_svc(self.cell.pos, self.obstacles.segments)
        Ac, bc = point_circle_svc(self.cell.pos, self.obstacles.circles)
        
        A = np.concat([Ab, Al, Ac])
        b = np.concat([bb, bl+1, bc+1])
        idx.extend(self.obstacles.indices)
        idx = np.array(idx)

        buffer_radius = self.cell.safety_radius * np.linalg.norm(A, axis=-1)
        # buffer_collision = np.sqrt(2*aik @ self.cell[i].pos.cov[i] @ aik) * self.collision_buffer_const
        b = b - buffer_radius

        poly = Polytope(A=A, b=b)
        if poly.vertices is None:
            A, b, idx = self.if_empty(self.cell.pos)
            poly = Polytope(A, b)

        poly._neighbors = idx[poly.neighbors]
        # poly._faces = dict(zip(poly._neighbors, poly.faces.values()))
        poly._neighbors_to_ids =  dict(zip(poly._neighbors, poly._neighbors_to_ids.values()))
        self.cell.poly = poly
    
    def boundary(self): #, safety_radius, cov):
        A_bound = np.array([[-1,0],[1,0], [0,-1],[0,1]], dtype=np.float64)
        b_bound = np.array([-self.xlim[0], self.xlim[1], 
                            -self.ylim[0], self.ylim[1]], dtype=np.float64)
        index = [-1, -2, -3, -4]
        return A_bound, b_bound, index
    
    def if_empty(self, position):
        eps = 1e-5
        A_bound = np.array([[-1,0],[1,0], [0,-1],[0,1]])
        b_bound = np.array([-(position[0]-eps), (position[0]+eps), 
                            -(position[1]-eps), (position[1]+eps)])
        index = np.random.randint(low=-1e5, high=-1e4, size=4)
        return A_bound, b_bound, index


class Polytope:
    def __init__(self, A=None, b=None, points=None):
        if A is not None and b is not None:
            self.A, self.b = A, b
            self._extreme()
        elif points is not None:
            self._qhull(points)
        else:
            raise ValueError('invalid polytope')
        
        sort_idx = sort_2d_polygon_ccw(np.mean(self.vertices, 0), self.vertices)
        self.vertices = self.vertices[sort_idx]

        self._faces = None
        self._neighbors = None
        self._volume = None
    
    @property
    def faces(self):
        if self._faces is None:
           self._faces, self._neighbors = self.compute_faces()
        return self._faces
    
    @property
    def neighbors(self):
        if self._neighbors is None:
            self._faces, self._neighbors = self.compute_faces()
        return self._neighbors
    
    def compute_faces(self):
        # print(f"A={self.A.tolist()},\nb={self.b.tolist()},\n v={self.vertices.shape}")
        # print(np.abs(self.A @ self.vertices.T - self.b[:,None]))
        N, F = np.where(np.abs(self.A @ self.vertices.T - self.b[:,None]) < 1e-6)
        # N, F = np.where(np.abs(self._ineq[:,0,None] + self._ineq[:,1:] @ self.vertices.T) < 1e-6)
        self._faces, self._neighbors_to_ids = [], {}
        for n, f in zip(N, F):
            if n not in self._neighbors_to_ids:
                self._faces.append([f])
                self._neighbors_to_ids[n] = len(self._faces) - 1
            else:
                self._faces[self._neighbors_to_ids[n]].append(f)
        self._neighbors = list(self._neighbors_to_ids.keys())

        return self._faces, self._neighbors
    
    def face(self, neighbor, Hrep=False):
        if neighbor not in self.neighbors:
            return None
        if Hrep:
            return self.A[self._neighbors_to_ids[neighbor]], self.b[self._neighbors_to_ids[neighbor]]
        else:
            return self.vertices[self.faces[self._neighbors_to_ids[neighbor]]]
    
    @property
    def volume(self):
        if self._volume is None:
            self._volume = get_polygon_centroid_area(self.vertices, is_sorted=True, only_area=True)
        return self._volume

    def _extreme(self):
        # poly = pc.Polytope(self.A, self.b)
        # self.vertices = pc.extreme(poly)
        ineq = np.hstack([self.b[:,None], -self.A])/ np.linalg.norm(self.A, axis=-1, keepdims=True)
        mat = cdd.matrix_from_array(ineq, rep_type=cdd.RepType.INEQUALITY)
        cdd.matrix_canonicalize(mat)
        poly = cdd.polyhedron_from_matrix(mat)
        ext = cdd.copy_output(poly)
        self.vertices = np.array(ext.array)[:,1:] if ext.array else None
    
    def _qhull(self, points):
        # poly = pc.qhull(points)
        # self.A, self.b = poly.A, poly.b
        # self.vertices =poly.vertices
        self.vertices = points[ConvexHull(points).vertices]
        scale = 1/np.abs(self.vertices).max(0)
        mat = cdd.matrix_from_array(
            np.pad(self.vertices*scale, ((0,0),(1,0)), constant_values=1), rep_type=cdd.RepType.GENERATOR
        )
        poly = cdd.polyhedron_from_matrix(mat)
        ext = cdd.copy_output(poly)
        ineq = np.array(ext.array)
        self.A, self.b = -ineq[:,1:]*scale, ineq[:,0]
        # print((self.A/self.b[:,None]))
        
def sort_2d_polygon_ccw(interior_pt, vertices_2d):
    '''sort vertices of polygon in counter-clockwise order'''
    theta = np.arctan2(vertices_2d[:,1] - interior_pt[1], 
                       vertices_2d[:,0] - interior_pt[0])
    return np.argsort(theta)
        
def nearest_point_on_polytope(ref_point, polytope, point_in_polytope):
    func = lambda x: np.linalg.norm(x - ref_point)
    constr = opt.LinearConstraint(A=polytope.A.squeeze(), ub=polytope.b.squeeze())
    res = opt.minimize(fun=func, x0=point_in_polytope, constraints=constr)
    return res.x

def point_on_polytope_given_direction(ref_point, direction, polytope):
    direction = direction/np.linalg.norm(direction)
    func = lambda t: -t
    constr = opt.LinearConstraint(A=(polytope.A.squeeze() @ direction).reshape(-1,1), 
                                  ub=polytope.b.squeeze() - polytope.A.squeeze() @ ref_point)
    res = opt.minimize(fun=func, x0=0, constraints=constr)
    return ref_point + direction*res.x[0]
        
def compute_centroid(geom, vertices, faces=None, is_sorted=False):
    if geom == "line":
        C = (vertices[0] + vertices[1])*0.5
    elif geom == "plane":
        C,_ = get_polygon_centroid_area(vertices, is_sorted)
    else:
        raise ValueError("Invalid geom")
    return C

def get_polygon_centroid_area(vertices, is_sorted=False, only_area=False):
    '''https://paulbourke.net/geometry/polygonmesh/centroid.pdf'''
    n_vertices = len(vertices)
    if n_vertices < 3:
        raise ValueError("There must be at least 3 vertices.")

    if not is_sorted:
        sort_idx = sort_2d_polygon_ccw(interior_pt=np.mean(vertices,0), vertices_2d=vertices)
        vertices = vertices[sort_idx]
    X, Y = vertices.T
    
    indices = np.arange(n_vertices)
    indices_offset = indices - 1
    
    p = (X[indices_offset]*Y[indices] - X[indices]*Y[indices_offset])
    area = 0.5 * p.sum()
    if only_area:
        return area
    centroid = 1/(6*area) * np.array([
        ((X[indices_offset] + X[indices]) * p).sum(),
        ((Y[indices_offset] + Y[indices]) * p).sum()
    ])
    return centroid, area

def point_segment_svc(p, segment):
    if len(segment) == 0:
        return np.zeros((0,2)), np.zeros((0,))
    s0, s1 = segment[:,0], segment[:,1]
    s = s1 - s0
    ds2 = np.vecdot(s, s)
    t = np.vecdot(p - s0, s) / ds2
    t = np.clip(t, 0, 1)
    q = s0 + t[:,None] * s
    
    qp = q - p
    dqp = np.vecdot(qp, qp)
    A = 2 * qp / dqp[:,None]
    b = np.vecdot(q + p, qp)/dqp
    return A, b

def point_circle_svc(p, circle):
    if len(circle) == 0:
        return np.zeros((0,2)), np.zeros((0,))
    c, r = circle[:,:2], circle[:,2]
    s = c - p
    ds = np.linalg.norm(s, axis=-1)
    t = (ds - r)
    q = p +  (t/ds)[:,None] * s

    qp = q - p
    dqp = np.sum(qp**2, axis=-1)
    A = 2 * qp / dqp[:,None]
    b = np.vecdot(q + p, qp)/dqp
    return A, b