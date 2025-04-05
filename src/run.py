from Vehicle import VehicleController 
from simulation import Simulation 
from plotting import Animation

if __name__ == '__main__':
    vc = VehicleController()
    sim = Simulation(vc)
    sim_data = sim.run_simulation()
    animation = Animation(sim_data, vc.params)
    animation.animate()
