from NeuronNetworkSimulation import NeuronNetworkSimulation
import argparse

def main(input_net1=9, input_net2=12, cop_val=0.1, simulation_time=10000, dt=1):
    simulation = NeuronNetworkSimulation(input_net1=input_net1, input_net2=input_net2, cop_val=cop_val, 
                                         simulation_time=simulation_time, dt=dt)
    simulation.run_simulation()
    signal1, signal2, px = simulation.analyze_results()
    simulation.plot_results(signal1, signal2, px)

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Run Neuron Network Simulation")
    
    parser.add_argument("--input_net1", type=float, default=9, help="Excitatory input to network 1")
    parser.add_argument("--input_net2", type=float, default=12, help="Excitatory input to network 2")
    parser.add_argument("--cop_val", type=float, default=0.1, help="Cross-network connection strength")
    parser.add_argument("--simulation_time", type=int, default=10000, help="Total simulation time")
    parser.add_argument("--dt", type=int, default=1, help="Time step")
    
    args = parser.parse_args()
    
    main(input_net1=args.input_net1, input_net2=args.input_net2, cop_val=args.cop_val, 
         simulation_time=args.simulation_time, dt=args.dt)

