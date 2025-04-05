import numpy as np

class VehicleController:

    def __init__(self):
        self.params = {
            'm': 1000, 'Iz': 3200, 'Caf': 20000, 'Car': 35000, 
            'lf': 2, 'lr': 3, 'dt': 0.02, 'hpp': 20, 'u': 20, 
            'channel_width': 10, 'trajectory': 2, 'r': 4, 'f': 0.01, 
            'Tmax': 10, 'PID_FLAG': 0, 'response' : 2,
            'Kp_yaw': 6, 'Kd_yaw': 3.5, 'Ki_yaw': 4, 
            'Kp_Y': 6, 'Kd_Y': 3.5, 'Ki_Y': 4,
            'Q': np.eye(2), 'S': np.eye(2), 'R': np.eye(1)
        }
    def generate_path(self, t):
        u = self.params['u']
        trajectory = self.params['trajectory']
        
        x = np.linspace(0, u * t[-1], len(t))
        
        def straight_line(x, t):
            return -10 * np.ones_like(t)
        
        def S_curve(x, t):
            return 10 * np.tanh(t - t[-1] / 2)
        
        def curvilinear_trajectory(x, t):
            r = self.params['r']
            f = self.params['f']
            ch = self.params['channel_width']
            return 3 * r * np.sin(np.pi * f * x) * np.cos(0.02 * x + ch / 10)
        
        trajectory_funcs = {
            1: straight_line,
            2: S_curve,
            3: curvilinear_trajectory,
        }
        
        if trajectory not in trajectory_funcs:
            raise ValueError("Trajectory must be 1, 2, or 3.")
        
        y = trajectory_funcs[trajectory](x, t)
        
        dx = np.diff(x)
        dy = np.diff(y)
        psi = np.arctan2(dy, dx)
        psi = np.insert(psi, 0, psi[0])  
        
        psi_unwrapped = np.unwrap(psi)
        
        return psi_unwrapped, x, y

    def discretize_state_space(self):
        c = self.params
        A = np.array([
            [-(2 * c['Caf'] + 2 * c['Car']) / (c['m'] * c['u']), 0, (-2 * c['Caf'] * c['lf'] + 2 * c['Car'] * c['lr']) / (c['m'] * c['u']), 0],
            [0, 0, 1, 0],
            [(-2 * c['lf'] * c['Caf'] + 2 * c['lr'] * c['Car']) / (c['Iz'] * c['u']), 0, (-2 * c['lf']**2 * c['Caf'] - 2 * c['lr']**2 * c['Car']) / (c['Iz'] * c['u']), 0],
            [1, c['u'], 0, 0]
        ])
        B = np.array([[2 * c['Caf'] / c['m']], [0], [2 * c['lf'] * c['Caf'] / c['Iz']], [0]])
        C = np.array([[0, 1, 0, 0], [0, 0, 0, 1]])
        
        Ad, Bd = np.eye(A.shape[0]) + c['dt'] * A, c['dt'] * B
        return Ad, Bd, C


    def construct_mpc_matrices(self, Ad, Bd, Cd, hpp):
        n, m = Bd.shape  
        p, _ = Cd.shape

        A_aug = np.block([[Ad, Bd], [np.zeros((m, n)), np.eye(m)]])
        B_aug = np.vstack((Bd, np.eye(m)))
        C_aug = np.hstack((Cd, np.zeros((p, m))))
        
        Q, S, R = self.params['Q'], self.params['S'], self.params['R']

        CQC = C_aug.T @ Q @ C_aug
        CSC = C_aug.T @ S @ C_aug
        QC = Q @ C_aug
        SC = S @ C_aug

        n_aug = A_aug.shape[0]
        Q_block = np.zeros((hpp * n_aug, hpp * n_aug))
        T_block = np.zeros((hpp * QC.shape[0], hpp * QC.shape[1]))
        R_block = np.kron(np.eye(hpp), R)  
        Cv = np.zeros((hpp * B_aug.shape[0], hpp * B_aug.shape[1]))
        Av = np.zeros((hpp * n_aug, n_aug))

        for i in range(hpp):
            idx1, idx2 = i * n_aug, (i + 1) * n_aug
            Av[idx1:idx2, :] = np.linalg.matrix_power(A_aug, i + 1)
            
            if i == hpp - 1:
                Q_block[idx1:idx2, idx1:idx2] = CSC
                T_block[i * SC.shape[0]:(i + 1) * SC.shape[0], i * SC.shape[1]:(i + 1) * SC.shape[1]] = SC
            else:
                Q_block[idx1:idx2, idx1:idx2] = CQC
                T_block[i * QC.shape[0]:(i + 1) * QC.shape[0], i * QC.shape[1]:(i + 1) * QC.shape[1]] = QC

            for j in range(i + 1):
                Cv[idx1:idx2, j * B_aug.shape[1]:(j + 1) * B_aug.shape[1]] = np.linalg.matrix_power(A_aug, (i - j)) @ B_aug

        H = Cv.T @ Q_block @ Cv + R_block
        F = np.vstack([Av.T @ Q_block @ Cv, -T_block @ Cv])

        return H, F, Cv, Av
    
    def next_state_prediction(self, states, U1):
        c, dt = self.params, self.params['dt']
        
        def dynamics(state, U):
            y_dot, psi, psi_dot, Y = state
            y_dot_dot = (-(2 * c['Caf'] + 2 * c['Car']) / (c['m'] * c['u']) * y_dot +
                        (-c['u'] - (2 * c['Caf'] * c['lf'] - 2 * c['Car'] * c['lr']) / (c['m'] * c['u'])) * psi_dot +
                        2 * c['Caf'] / c['m'] * U)
            psi_dot_dot = (-(2 * c['lf'] * c['Caf'] - 2 * c['lr'] * c['Car']) / (c['Iz'] * c['u']) * y_dot -
                        (2 * c['lf']**2 * c['Caf'] + 2 * c['lr']**2 * c['Car']) / (c['Iz'] * c['u']) * psi_dot +
                        2 * c['lf'] * c['Caf'] / (c['Iz'] * c['u']) * U)
            Y_dot = np.sin(psi) * c['u'] + np.cos(psi) * y_dot
            return np.array([y_dot_dot, psi_dot, psi_dot_dot, Y_dot])
        
        k1 = dynamics(states, U1)
        k2 = dynamics(states + dt/2 * k1, U1)
        k3 = dynamics(states + dt/2 * k2, U1)
        k4 = dynamics(states + dt * k3, U1)
        
        new_state = states + (dt/6) * (k1 + 2*k2 + 2*k3 + k4)
        return new_state

