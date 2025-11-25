import argparse
from pathlib import Path

from controller import load_network
from racetrack import RaceTrack
from simulator import Simulator, plt

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Run racing simulator with visualization")
    parser.add_argument("track", type=str, help="Path to track CSV file")
    parser.add_argument("raceline", type=str, help="Path to raceline CSV file")
    parser.add_argument(
        "--network", "-n", type=str, default="neat_winner.pkl", help="Path to trained NEAT network pickle file"
    )

    args = parser.parse_args()

    # Load trained network
    network_path = Path(args.network)
    if network_path.exists():
        print(f"Loading network from {network_path}")
        load_network(network_path)
    else:
        print(f"Warning: Network file not found: {network_path}")
        print("Running with fallback controller (go straight slowly)")

    racetrack = RaceTrack(args.track, args.raceline)

    simulator = Simulator(racetrack)
    simulator.start()
    plt.show()
