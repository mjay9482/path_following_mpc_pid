import numpy as np
import matplotlib.pyplot as plt
import matplotlib.animation as animation

class Animation:
    def __init__(self, sim_data, params):
        self.total_states = sim_data['total_states']
        self.psi_opt_total = sim_data['psi_opt_total']
        self.Y_opt_total = sim_data['Y_opt_total']
        self.U_all = sim_data['U_all']
        self.Xd = sim_data['Xd']
        self.Yd = sim_data['Yd']
        self.t = sim_data['t']
        self.params = params
        
        self.f_rate = int(params['Tmax'] / params['dt'])
        self.lf = params['lf']
        self.lr = params['lr']
        self.PID_FLAG = params['PID_FLAG']
        
        self.fig_x = 16
        self.fig_y = 9
        self.fig, self.ax0 = plt.subplots(figsize=(self.fig_x, self.fig_y), dpi=120, facecolor='black')
        self.setup_plot()

    def setup_plot(self):
        self.ax0.set_facecolor('black')
        self.ax0.plot(self.Xd, self.Yd, 'c', linewidth=1, label='Desired path')
        
        f_rate = self.f_rate
        ch_width = self.params['channel_width']
        self.ax0.plot([self.Xd[0], self.Xd[f_rate]], 
                      [ch_width / 2, ch_width / 2],
                      color='grey', linewidth=1)
        self.ax0.plot([self.Xd[0], self.Xd[f_rate]], 
                      [-ch_width / 2, -ch_width / 2],
                      color='grey', linewidth=1)
        self.ax0.plot([self.Xd[0], self.Xd[f_rate]], 
                      [ch_width / 2 + ch_width, ch_width / 2 + ch_width],
                      color='grey', linewidth=1)
        self.ax0.plot([self.Xd[0], self.Xd[f_rate]], 
                      [-ch_width / 2 - ch_width, -ch_width / 2 - ch_width],
                      color='grey', linewidth=1)
        self.ax0.plot([self.Xd[0], self.Xd[f_rate]], 
                      [ch_width / 2 + 2 * ch_width, ch_width / 2 + 2 * ch_width],
                      color='grey', linewidth=3)
        self.ax0.plot([self.Xd[0], self.Xd[f_rate]], 
                      [-ch_width / 2 - 2 * ch_width, -ch_width / 2 - 2 * ch_width],
                      color='grey', linewidth=3)
        
        self.vehicle_line, = self.ax0.plot([], [], 'w', linewidth=3, label='Vehicle')
        self.vehicle_pred_line, = self.ax0.plot([], [], '-y', linewidth=1, label='Predicted Path')
        self.vehicle_travel_line, = self.ax0.plot([], [], '-r', linewidth=1, label='Traveled Path')
        
        self.ax0.set_xlim(self.Xd[0], self.Xd[f_rate])
        self.ax0.set_ylim(-self.Xd[f_rate] / (2 * (self.fig_x / self.fig_y)), 
                          self.Xd[f_rate] / (2 * (self.fig_x / self.fig_y)))
        self.ax0.set_xlabel('Longitudinal movement (X)', fontsize=15, color='white')
        self.ax0.set_ylabel('Lateral movement (Y)', fontsize=15, color='white')
        self.ax0.tick_params(axis='x', colors='white')
        self.ax0.tick_params(axis='y', colors='white')
        self.ax0.legend()

    def update_plot(self, num):
        hpp = self.params['hpp']
        if num + hpp > len(self.t):
            hpp = len(self.t) - num
        
        self.vehicle_line.set_data(
            [self.Xd[num] - self.lr * np.cos(self.total_states[num, 1]),
             self.Xd[num] + self.lf * np.cos(self.total_states[num, 1])],
            [self.total_states[num, 3] - self.lr * np.sin(self.total_states[num, 1]),
             self.total_states[num, 3] + self.lf * np.sin(self.total_states[num, 1])]
        )
        self.vehicle_travel_line.set_data(self.Xd[:num], self.total_states[:num, 3]) 
        if self.PID_FLAG != 1 and num != 0:
            self.vehicle_pred_line.set_data(self.Xd[num:num + hpp], self.Y_opt_total[num, :hpp])
        
        return (self.vehicle_line, self.vehicle_pred_line, self.vehicle_travel_line)

    def animate(self):
        ani = animation.FuncAnimation(
            self.fig, 
            self.update_plot,
            frames=self.f_rate, 
            interval=20, 
            repeat=True, 
            blit=True
        )
        plt.show()
