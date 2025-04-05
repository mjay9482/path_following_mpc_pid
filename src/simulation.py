import numpy as np
from Vehicle import VehicleController  

class Simulation:
    def __init__(self, vc: VehicleController):
        self.vc = vc
        self.params = vc.params
        
        self.dt = self.params['dt']
        self.response = self.params['response']
        self.hpp = self.params['hpp']
        self.Tmax = self.params['Tmax']
        
        self.t = np.arange(0, self.Tmax + self.dt, self.dt)
        self.psid, self.Xd, self.Yd = vc.generate_path(self.t)
        self.sim_length = len(self.t)
        
        self.signal_d = np.zeros(len(self.Xd) * self.response)
        k = 0
        for i in range(0, len(self.signal_d), self.response):
            self.signal_d[i] = self.psid[k]
            self.signal_d[i + 1] = self.Yd[k]
            k += 1
        
        self.y_dot, self.psi, self.psi_dot = 0.0, 0.0, 0.0
        self.Y = self.Yd[0] + 10.0  
        self.states = np.array([self.y_dot, self.psi, self.psi_dot, self.Y])
        
        self.total_states = np.zeros((self.sim_length, len(self.states)))
        self.total_states[0] = self.states.copy()
        self.psi_opt_total = np.zeros((self.sim_length, self.hpp))
        self.Y_opt_total = np.zeros((self.sim_length, self.hpp))
        
        self.U_i = 0.0
        self.U_all = np.zeros(self.sim_length)
        self.U_all[0] = self.U_i
 
        self.C_psi_opt = np.zeros((self.hpp, (len(self.states) + 1) * self.hpp))
        for i in range(1, self.hpp + 1):
            self.C_psi_opt[i - 1][i + 4 * (i - 1)] = 1

        self.C_Y_opt = np.zeros((self.hpp, (len(self.states) + 1) * self.hpp))
        for i in range(3, self.hpp + 3):
            self.C_Y_opt[i - 3][i + 4 * (i - 3)] = 1
        
        self.Ad, self.Bd, self.Cd = vc.discretize_state_space()
        self.H, self.F, self.Cv, self.Av = vc.construct_mpc_matrices(self.Ad, self.Bd, self.Cd, self.hpp)
        
        self.PID_FLAG = self.params['PID_FLAG']
        self.old_states = None
        self.e_int_pid_yaw = 0.0
        self.e_int_pid_Y = 0.0

    def run_simulation(self):
        k = 0
        for i in range(0, self.sim_length - 1):
            x_aug_t = np.transpose([np.concatenate((self.states, [self.U_i]), axis=0)])
            k += self.response
            
            curr_hpp = self.hpp
            if k + self.response * curr_hpp <= len(self.signal_d):
                r = self.signal_d[k:k + self.response * curr_hpp]
            else:
                r = self.signal_d[k:]
                curr_hpp = len(r) // self.response

            if curr_hpp < self.params['hpp']:
                self.H, self.F, self.Cv, self.Av = self.vc.construct_mpc_matrices(
                    self.Ad, self.Bd, self.Cd, curr_hpp
                )
            
            ft = np.hstack((x_aug_t.flatten(), r)) @ self.F  
            du = -np.linalg.inv(self.H) @ ft.reshape(-1, 1)
            x_aug_opt = self.Cv @ du + self.Av @ x_aug_t
            
            psi_opt = (self.C_psi_opt[:curr_hpp, : (len(self.states) + 1) * curr_hpp] @ x_aug_opt).flatten()
            Y_opt = (self.C_Y_opt[:curr_hpp, : (len(self.states) + 1) * curr_hpp] @ x_aug_opt).flatten()
            self.psi_opt_total[i + 1, :curr_hpp] = psi_opt
            self.Y_opt_total[i + 1, :curr_hpp] = Y_opt
            
            self.U_i += du[0, 0]

            if self.PID_FLAG == 1:
                if i == 0:
                    self.e_int_pid_yaw, self.e_int_pid_Y = 0.0, 0.0
                else:
                    e_pid_yaw_prev = self.psid[i - 1] - self.old_states[1]
                    e_pid_yaw_curr = self.psid[i] - self.states[1]
                    e_dot_pid_yaw = (e_pid_yaw_curr - e_pid_yaw_prev) / self.dt
                    self.e_int_pid_yaw += (e_pid_yaw_prev + e_pid_yaw_curr) * self.dt / 2
                    U_i_yaw = (self.params['Kp_yaw'] * e_pid_yaw_curr +
                               self.params['Kd_yaw'] * e_dot_pid_yaw +
                               self.params['Ki_yaw'] * self.e_int_pid_yaw)
                    
                    e_pid_Y_prev = self.Yd[i - 1] - self.old_states[3]
                    e_pid_Y_curr = self.Yd[i] - self.states[3]
                    e_dot_pid_Y = (e_pid_Y_curr - e_pid_Y_prev) / self.dt
                    self.e_int_pid_Y += (e_pid_Y_prev + e_pid_Y_curr) * self.dt / 2
                    U_i_Y = (self.params['Kp_Y'] * e_pid_Y_curr +
                             self.params['Kd_Y'] * e_dot_pid_Y +
                             self.params['Ki_Y'] * self.e_int_pid_Y)
                    
                    self.U_i = U_i_yaw + U_i_Y

                self.old_states = self.states.copy()

            self.U_i = np.clip(self.U_i, -np.pi/6, np.pi/6)
            self.U_all[i + 1] = self.U_i

            self.states = self.vc.next_state_prediction(self.states, self.U_i)
            self.total_states[i + 1][:len(self.states)] = self.states

        return {
            'total_states': self.total_states,
            'psi_opt_total': self.psi_opt_total,
            'Y_opt_total': self.Y_opt_total,
            'U_all': self.U_all,
            'Xd': self.Xd,
            'Yd': self.Yd,
            't': self.t
        }
