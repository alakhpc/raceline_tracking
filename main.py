import argparse
import os
import pickle

import matplotlib.pyplot as plt
import neat

from racetrack import RaceTrack
from simulator import Simulator

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Run racing simulator with visualization")
    parser.add_argument("track", type=str, help="Path to track CSV file")
    parser.add_argument("raceline", type=str, help="Path to raceline CSV file")
    parser.add_argument(
        "--config",
        type=str,
        required=True,
        help="Path to saved NEAT genome pickle file (e.g., neat_winner.pkl)",
    )

    args = parser.parse_args()

    # Load NEAT config
    local_dir = os.path.dirname(__file__)
    config_path = os.path.join(local_dir, "neat_config.txt")

    neat_config = neat.Config(
        neat.DefaultGenome,
        neat.DefaultReproduction,
        neat.DefaultSpeciesSet,
        neat.DefaultStagnation,
        config_path,
    )

    # Load genome from pickle file
    print(f"Loading NEAT genome from {args.config}...")
    with open(args.config, "rb") as f:
        genome = pickle.load(f)

    # Create network from genome
    neat_net = neat.nn.RecurrentNetwork.create(genome, neat_config)
    print(
        f"Loaded genome with {len(genome.nodes)} nodes and {len([c for c in genome.connections.values() if c.enabled])} connections"
    )

    racetrack = RaceTrack(args.track, args.raceline)

    simulator = Simulator(racetrack, neat_net=neat_net)
    simulator.start()
    plt.show()
