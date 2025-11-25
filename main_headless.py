import argparse
import os
import pickle

import neat

from racetrack import RaceTrack
from simulator_headless import HeadlessSimulator

TRACKS = [
    ("IMS", "racetracks/IMS.csv", "racetracks/IMS_raceline.csv"),
    ("Montreal", "racetracks/Montreal.csv", "racetracks/Montreal_raceline.csv"),
    ("Monza", "racetracks/Monza.csv", "racetracks/Monza_raceline.csv"),
]

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Run headless racing simulation with NEAT controller")
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

    results_table = []

    for track_name, track_path, raceline_path in TRACKS:
        print(f"Running simulation on {track_name}...")
        racetrack = RaceTrack(track_path, raceline_path)
        simulator = HeadlessSimulator(racetrack, neat_net=neat_net)
        results = simulator.run()
        results_table.append(
            {
                "track": track_name,
                "lap_finished": results["lap_finished"],
                "sim_time": results["sim_time_elapsed"],
                "violations": results["track_limit_violations"],
            }
        )

    # Print results table
    print("\n" + "=" * 55)
    print("SIMULATION RESULTS")
    print("=" * 55)
    print(f"{'Track':<12} {'Status':<12} {'Sim Time (s)':<15} {'Violations':<10}")
    print("-" * 55)
    for row in results_table:
        status = "✓ Finished" if row["lap_finished"] else "✗ DNF"
        print(f"{row['track']:<12} {status:<12} {row['sim_time']:<15.2f} {row['violations']:<10}")
    print("=" * 55)
