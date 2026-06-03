import math
import numpy as np
import scipy.optimize as sopt
import scipy.sparse as sparse
import osqp
import casadi as ca
from acados_template import AcadosOcp, AcadosOcpSolver, AcadosOcpOptions, AcadosModel


class CoridorMPC:
    def __init__(self, N=30, max_faces=12, max_poly=30, max_v=1.5, max_omega=1.5):
        self.N = N
        self.max_faces = max_faces
        self.max_poly = max_poly
        
        self.v_min, self.v_max = -max_v, max_v
        self.w_min, self.w_max = -max_omega, max_omega
        self.dt_min, self.dt_max = 1e-3, 2.0

        self._build_solver()
        self._prev_sol = None
        self._Ab_corr = np.empty((self.N, 2*self.max_faces, 3))
        self._X_init = np.empty((self.N+1, 4))
        self._U_init = np.empty((self.N, 3))
        self._params = np.empty((self.N+1, 2*self.max_faces*3 + 2))
    
    def _export_model(self):
        model = AcadosModel()
        model.name = 'corridor_mpc'
        
        x = ca.SX.sym('x')
        y = ca.SX.sym('y')
        theta = ca.SX.sym('theta')
        dt = ca.SX.sym('dt') 
        states = ca.vertcat(x, y, theta, dt)
        
        ds = ca.SX.sym('ds')
        dtheta = ca.SX.sym('dtheta')
        tau = ca.SX.sym('tau')
        controls = ca.vertcat(ds, dtheta, tau)

        x_dot = ca.SX.sym('x_dot')
        y_dot = ca.SX.sym('y_dot')
        theta_dot = ca.SX.sym('theta_dot')
        dt_dot = ca.SX.sym('dt_dot')
        states_dot = ca.vertcat(x_dot, y_dot, theta_dot, dt_dot)

        f_expl = ca.vertcat(
            ds * ca.cos(theta), 
            ds * ca.sin(theta), 
            dtheta,
            tau,
        )
        f_impl = states_dot - f_expl
        
        Ab = ca.SX.sym('Ab', 2*self.max_faces, 3)
        A, b = Ab[:,:2], Ab[:,2]
        target = ca.SX.sym('target', 2)
        sym_p = ca.vertcat(ca.vec(Ab), target)
        
        h_expr1 = A @ states[:2] - b
        h_expr2 = ds - self.v_max * dt, -ds + self.v_min * dt
        h_expr3 = dtheta - self.w_max * dt, -dtheta + self.w_min * dt
        h_expr = ca.vertcat(h_expr1, *h_expr2, *h_expr3)
        
        model.f_impl_expr = f_impl
        model.f_expl_expr = f_expl
        model.x = states
        model.xdot = states_dot
        model.u = controls

        model.p = sym_p
        model.con_h_expr_0 = h_expr
        model.con_h_expr = h_expr

        cost_T = dt**2
        cost_U = 1e-3*(ds**2 + dtheta**2 + tau**2)
        # cost_C = 0.5 * ca.sum1(ca.exp(5.0 * h_expr1))
        
        model.cost_expr_ext_cost = cost_T + cost_U #+ cost_C
        model.cost_expr_ext_cost_e = 100.0 * ((x - target[0])**2 + (y - target[1])**2)
        return model

    def _solver_options(self, Tf, N, M):
        solver_options = AcadosOcpOptions()
        
        solver_options.qp_solver = 'PARTIAL_CONDENSING_HPIPM'
        solver_options.qp_solver_cond_N = N
        solver_options.hessian_approx = 'EXACT'
        solver_options.integrator_type = 'ERK'
        solver_options.sim_method_num_steps = M

        solver_options.nlp_solver_type = 'SQP'
        solver_options.globalization = 'FUNNEL_L1PEN_LINESEARCH' #'FIXED_STEP'
        solver_options.nlp_solver_max_iter = 10
        solver_options.with_adaptive_levenberg_marquardt = True
        solver_options.regularize_method = 'PROJECT'
        
        # set prediction horizon
        solver_options.tf = Tf
        solver_options.print_level = 0
        

        return solver_options
    
    def _build_solver(self):
        ocp = AcadosOcp()
        ocp.model = self._export_model()
        
        ocp.cost.cost_type = 'EXTERNAL'
        ocp.cost.cost_type_e = 'EXTERNAL'

        # Initial conditions
        ocp.constraints.lbx_0 = np.zeros(3)
        ocp.constraints.ubx_0 = np.zeros(3)
        ocp.constraints.idxbx_0 = np.array([0,1,2])

        ocp.constraints.lbx = np.array([self.dt_min])
        ocp.constraints.ubx = np.array([self.dt_max])
        ocp.constraints.idxbx = np.array([3])
        
        # constraints on control
        ocp.constraints.lbu = np.array([-0.2])
        ocp.constraints.ubu = np.array([0.2])
        ocp.constraints.idxbu = np.array([2])

        # boundary constraints
        nh = 2*self.max_faces
        ocp.constraints.lh = -1e3 * np.ones(nh+4)
        ocp.constraints.uh = np.zeros(nh+4)
        ocp.constraints.lh_0 = -1e3 * np.ones(nh+4)
        ocp.constraints.uh_0 = np.zeros(nh+4)

        penalty = 1e5
        ocp.constraints.idxsh = np.arange(nh)
        ocp.constraints.idxsh_0 = np.arange(nh)
        ocp.cost.Zl = penalty * np.ones(nh)
        ocp.cost.Zu = penalty * np.ones(nh)
        ocp.cost.zl = penalty * np.ones(nh)
        ocp.cost.zu = penalty * np.ones(nh)

        # parameters
        ocp.parameter_values = np.zeros(nh*3+2)
        
        # settings
        Tf = float(self.N)
        ocp.solver_options = self._solver_options(Tf=Tf, N=self.N, M=2)
        ocp.solver_options.N_horizon = self.N

        self.solver = AcadosOcpSolver(ocp, json_file='safecorridor_acados.json')


    def _set_safecorr(self, safecorr, ref_path, X0):
        X0 = np.asarray(X0)
        self.solver.set(0, 'lbx', X0)
        self.solver.set(0, 'ubx', X0)

        N = self.N-1 if self.max_poly > self.N else self.max_poly
        safecorr = safecorr[:N]
        ref_path = np.array(ref_path[:N+1])

        A = np.stack([poly.A for poly in safecorr])
        b = np.stack([poly.b for poly in safecorr])
        Ab = np.concatenate([A, b[...,None]], axis=-1)
        num_poly, num_faces, _ = A.shape
        assert self.max_faces >= num_faces 
        
        num_path = len(ref_path)
        base_count = (self.N-2) // (num_path-2)
        extras = (self.N-2) % (num_path-2) 
        
        self._Ab_corr[...,:2] = 0 
        self._Ab_corr[...,2] = 1

        self._Ab_corr[0][-num_faces:,:] = Ab[0]
        self._Ab_corr[-1][-num_faces:,:] = Ab[-1]

        pos_r = np.empty((self.N+1, 2))
        pos_r[0] = ref_path[None,0]
        for i in range(1, num_path-1):
            k = max(i+2 - (num_path - extras), 0)
            j = 1 + (i-1)*base_count
            j1 = j + k - (1 if k>0 else 0)
            j2 = j + base_count + k
            # print(i, k, (j1,j2))

            self._Ab_corr[j1:j2, -num_faces:,:] = Ab[i]
            self._Ab_corr[j1,:num_faces,:] = Ab[i-1]

            if j2 - j1 == 1:
                p = ref_path[None,i]
            else:
                t = np.linspace(0, 1, j2-j1, endpoint=False)
                p = ref_path[i] + t[:,None]*(ref_path[i+1] - ref_path[i])
            pos_r[j1:j2] = p
        pos_r[-2:] = ref_path[None,-1]

        p = 0.8
        dpos_r = np.diff(pos_r, axis=0)   
        dt = np.hypot(dpos_r[:,0], dpos_r[:,1])/(p*self.v_max)
        dt = np.maximum(dt, 0.001)
        theta_r = np.arctan2(dpos_r[:,1], dpos_r[:,0])
        dtheta = (np.diff(theta_r) + np.pi) % (2*np.pi) - np.pi
        omega = np.clip(dtheta / dt[:-1], self.w_min, self.w_max)
        
        self._X_init[:] = 0
        self._X_init[:,:2] = pos_r
        self._X_init[:-1,2] = theta_r
        self._X_init[0,:3] = X0
        self._X_init[:-1,3] = dt

        self._U_init[:] = 0
        self._U_init[:,0] = p*self.v_max * dt #* np.cos(theta_r)
        self._U_init[:-1,1] = omega*dt[:-1]

        Xf = np.asarray(ref_path[-1])
        self._params[:-1,:-2] = self._Ab_corr.reshape(self.N,-1, order='F')
        self._params[:,-2:] = Xf
        # print(self._params.shape, self._Ab_corr.shape, Xf.shape)
        
        self._X_init[-1] = self._X_init[-2]
        self._params[-1] = self._params[-2]
        # print(self._params)
        
        self.solver.set_flat('x', self._X_init.flatten())
        self.solver.set_flat('u', self._U_init.flatten())
        self.solver.set_flat('p', self._params.flatten())

    def solve(self, X0, ref_path, safecorr):
        self._set_safecorr(safecorr, ref_path, X0)

        status = self.solver.solve()
        if status not in [0, 2]:
            return None
            
        X_sol = np.array([self.solver.get(i, "x") for i in range(self.N + 1)])
        U_sol = np.array([self.solver.get(i, "u") for i in range(self.N)])
        
        dt_sol = X_sol[:, 3] 
        opt_U = U_sol[:, :2] / dt_sol[:-1,None]
        pred_traj = X_sol[:,:3]
        
        # self._prev_sol = (X_sol, U_sol)
        
        return opt_U, pred_traj, dt_sol
    

class ReactiveFeedback:
    def __init__(self, control_gain=2.5, ds=0.1, gamma=2.0, v_max=1.5, w_max=1.5):
        self.control_gain = control_gain
        self.ds = ds
        self.gamma = gamma

        self.v_min, self.v_max = -v_max, v_max
        self.w_min, self.w_max = -w_max, w_max

    def __call__(self, pose, goal, polytope):
        pose = np.asarray(pose)
        goal = np.asarray(goal)
        c, s = math.cos(pose[2]), math.sin(pose[2])
        
        g_dir = goal - pose[:2]
        h_dir = np.array([c, s])

        g = nearest_point_on_polytope(goal, polytope, pose[:2])
        gw = point_on_polytope_given_direction(pose[:2], g_dir, polytope)
        gv = point_on_polytope_given_direction(pose[:2], h_dir, polytope)

        hp_dir = np.array([-s, c])
        q = (g + gw)*0.5 - pose[:2]
        turn_angle = math.atan2((hp_dir @ q), (h_dir @ q))
        velocity = self.control_gain * h_dir @ (gv - pose[:2])
        omega = self.control_gain * turn_angle
        
        velocity = max(min(velocity, 1.0), -1.0)
        omega = max(min(omega, 1.5), -1.5)
        return velocity, omega

def nearest_point_on_polytope(ref_point, polytope, point_in_polytope):
    A, b = polytope.A, polytope.b
    func = lambda x: np.linalg.norm(x - ref_point)
    constr = sopt.LinearConstraint(A=A, ub=b)
    res = sopt.minimize(fun=func, x0=point_in_polytope, constraints=constr)
    return res.x

def point_on_polytope_given_direction(ref_point, direction, polytope):
    A, b = polytope.A, polytope.b
    direction = direction/np.linalg.norm(direction)
    func = lambda t: -t
    constr = sopt.LinearConstraint(
        A=(A @ direction).reshape(-1,1), ub=(b - A @ ref_point)
    )
    res = sopt.minimize(fun=func, x0=0, constraints=constr)
    return ref_point + direction*res.x[0]