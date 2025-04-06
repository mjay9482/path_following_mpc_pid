import numpy as np
from Vehicle import VehicleController
from simulation import Simulation
from plotting import Animation, PerformanceMetrics

def run_simulation_and_analyze():
    # Initialize the vehicle controller
    vc = VehicleController()
    
    # Run the simulation
    sim = Simulation(vc)
    sim_data = sim.run_simulation()
    
    # Calculate performance metrics
    metrics = PerformanceMetrics(sim_data, vc.params)
    all_metrics = metrics.calculate_all_metrics()
    
    # Plot the metrics
    metrics.plot_metrics()
    
    # Animate the simulation
    anim = Animation(sim_data, vc.params)
    anim.animate()
    
    return all_metrics

def compare_controllers():
    """Compare MPC vs PID controller performance"""
    # Initialize the vehicle controller
    vc = VehicleController()
    
    # Run simulation with MPC (default)
    vc.params['PID_FLAG'] = 0
    sim_mpc = Simulation(vc)
    sim_data_mpc = sim_mpc.run_simulation()
    metrics_mpc = PerformanceMetrics(sim_data_mpc, vc.params)
    mpc_metrics = metrics_mpc.calculate_all_metrics()
    
    # Run simulation with PID
    vc.params['PID_FLAG'] = 1
    sim_pid = Simulation(vc)
    sim_data_pid = sim_pid.run_simulation()
    metrics_pid = PerformanceMetrics(sim_data_pid, vc.params)
    pid_metrics = metrics_pid.calculate_all_metrics()
    
    # Print comparison
    print("\n=== Controller Performance Comparison ===")
    print("Metric                  | MPC         | PID         | Improvement")
    print("-" * 65)
    
    mpc_cte = mpc_metrics['cross_track_error']['mean_cte']
    pid_cte = pid_metrics['cross_track_error']['mean_cte']
    cte_improvement = ((pid_cte - mpc_cte) / pid_cte) * 100
    
    mpc_energy = mpc_metrics['controller_effort']['control_energy']
    pid_energy = pid_metrics['controller_effort']['control_energy']
    energy_improvement = ((pid_energy - mpc_energy) / pid_energy) * 100
    
    mpc_settling = mpc_metrics['settling_time']['settling_time']
    pid_settling = pid_metrics['settling_time']['settling_time']
    settling_improvement = ((pid_settling - mpc_settling) / pid_settling) * 100
    
    print(f"Mean Cross Track Error | {mpc_cte:.3f}m     | {pid_cte:.3f}m     | {cte_improvement:.1f}%")
    print(f"Control Energy         | {mpc_energy:.3f}   | {pid_energy:.3f}   | {energy_improvement:.1f}%")
    print(f"Settling Time          | {mpc_settling:.2f}s    | {pid_settling:.2f}s    | {settling_improvement:.1f}%")
    
    return {
        'mpc': mpc_metrics,
        'pid': pid_metrics
    }

if __name__ == "__main__":
    # Run a single simulation and analyze
    metrics = run_simulation_and_analyze()
    
    # Uncomment to compare MPC vs PID
    comparison = compare_controllers() 