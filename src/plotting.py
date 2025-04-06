import numpy as np
import matplotlib.pyplot as plt
import matplotlib.animation as animation

class PerformanceMetrics:
    def __init__(self, sim_data, params):
        self.total_states = sim_data['total_states']
        self.U_all = sim_data['U_all']
        self.Xd = sim_data['Xd']
        self.Yd = sim_data['Yd']
        self.t = sim_data['t']
        self.params = params
        
    def calculate_cross_track_error(self):
        """Calculate the cross track error (lateral deviation from desired path)"""
        # Extract the actual Y position from states
        Y_actual = self.total_states[:, 3]
        
        # Calculate the error at each point
        cte = np.abs(Y_actual - self.Yd)
        
        # Calculate mean, max, and RMS cross track error
        mean_cte = np.mean(cte)
        max_cte = np.max(cte)
        rms_cte = np.sqrt(np.mean(cte**2))
        
        return {
            'mean_cte': mean_cte,
            'max_cte': max_cte,
            'rms_cte': rms_cte,
            'cte': cte
        }
    
    def calculate_controller_effort(self):
        """Calculate the controller effort (energy in control inputs)"""
        # Calculate the RMS of control inputs
        rms_control = np.sqrt(np.mean(self.U_all**2))
        
        # Calculate the total control energy
        control_energy = np.sum(self.U_all**2) * self.params['dt']
        
        # Calculate the maximum control input
        max_control = np.max(np.abs(self.U_all))
        
        return {
            'rms_control': rms_control,
            'control_energy': control_energy,
            'max_control': max_control
        }
    
    def calculate_settling_time(self, error_threshold=0.1, window_size=10):
        """
        Calculate the settling time - time to reach and stay within error_threshold
        window_size: number of consecutive points that must be within threshold
        """
        # Calculate the error
        Y_actual = self.total_states[:, 3]
        error = np.abs(Y_actual - self.Yd)
        
        # Find points within threshold
        within_threshold = error < error_threshold
        
        # Find the first point where we stay within threshold for window_size consecutive points
        for i in range(len(within_threshold) - window_size + 1):
            if np.all(within_threshold[i:i+window_size]):
                settling_time = self.t[i]
                settling_index = i
                break
        else:
            # If we never settle, return the simulation time
            settling_time = self.t[-1]
            settling_index = len(self.t) - 1
        
        return {
            'settling_time': settling_time,
            'settling_index': settling_index,
            'error_threshold': error_threshold
        }
    
    def calculate_all_metrics(self):
        """Calculate all performance metrics"""
        cte_metrics = self.calculate_cross_track_error()
        control_metrics = self.calculate_controller_effort()
        settling_metrics = self.calculate_settling_time()
        
        return {
            'cross_track_error': cte_metrics,
            'controller_effort': control_metrics,
            'settling_time': settling_metrics
        }
    
    def plot_metrics(self):
        """Plot the performance metrics"""
        cte_metrics = self.calculate_cross_track_error()
        control_metrics = self.calculate_controller_effort()
        settling_metrics = self.calculate_settling_time()
        
        # Create a figure with subplots
        fig, axs = plt.subplots(3, 1, figsize=(12, 15), facecolor='black')
        
        # Plot cross track error
        axs[0].set_facecolor('black')
        axs[0].plot(self.t, cte_metrics['cte'], 'r', linewidth=2)
        axs[0].axhline(y=cte_metrics['mean_cte'], color='w', linestyle='--', label=f'Mean: {cte_metrics["mean_cte"]:.3f}')
        axs[0].set_title('Cross Track Error', color='white', fontsize=15)
        axs[0].set_xlabel('Time (s)', color='white')
        axs[0].set_ylabel('Error (m)', color='white')
        axs[0].tick_params(axis='x', colors='white')
        axs[0].tick_params(axis='y', colors='white')
        axs[0].grid(True, linestyle='--', alpha=0.7)
        axs[0].legend()
        
        # Plot control input
        axs[1].set_facecolor('black')
        axs[1].plot(self.t, self.U_all, 'g', linewidth=2)
        axs[1].axhline(y=control_metrics['rms_control'], color='w', linestyle='--', 
                      label=f'RMS: {control_metrics["rms_control"]:.3f}')
        axs[1].set_title('Control Input', color='white', fontsize=15)
        axs[1].set_xlabel('Time (s)', color='white')
        axs[1].set_ylabel('Steering Angle (rad)', color='white')
        axs[1].tick_params(axis='x', colors='white')
        axs[1].tick_params(axis='y', colors='white')
        axs[1].grid(True, linestyle='--', alpha=0.7)
        axs[1].legend()
        
        # Plot error with settling time indicator
        axs[2].set_facecolor('black')
        axs[2].plot(self.t, cte_metrics['cte'], 'b', linewidth=2)
        axs[2].axvline(x=settling_metrics['settling_time'], color='y', linestyle='--', 
                      label=f'Settling Time: {settling_metrics["settling_time"]:.2f}s')
        axs[2].axhline(y=settling_metrics['error_threshold'], color='r', linestyle=':', 
                      label=f'Threshold: {settling_metrics["error_threshold"]}')
        axs[2].set_title('Error with Settling Time', color='white', fontsize=15)
        axs[2].set_xlabel('Time (s)', color='white')
        axs[2].set_ylabel('Error (m)', color='white')
        axs[2].tick_params(axis='x', colors='white')
        axs[2].tick_params(axis='y', colors='white')
        axs[2].grid(True, linestyle='--', alpha=0.7)
        axs[2].legend()
        
        plt.tight_layout()
        plt.show()
        
        # Print summary metrics
        print("\n=== Performance Metrics Summary ===")
        print(f"Mean Cross Track Error: {cte_metrics['mean_cte']:.3f} m")
        print(f"Max Cross Track Error: {cte_metrics['max_cte']:.3f} m")
        print(f"RMS Cross Track Error: {cte_metrics['rms_cte']:.3f} m")
        print(f"RMS Control Input: {control_metrics['rms_control']:.3f} rad")
        print(f"Control Energy: {control_metrics['control_energy']:.3f} rad²·s")
        print(f"Settling Time: {settling_metrics['settling_time']:.2f} s")

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
