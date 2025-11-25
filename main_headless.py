import argparse
from pathlib import Path

from controller import load_network
from racetrack import RaceTrack
from simulator_headless import HeadlessSimulator

TRACKS = [
    ("IMS", "racetracks/IMS.csv", "racetracks/IMS_raceline.csv"),
    ("Montreal", "racetracks/Montreal.csv", "racetracks/Montreal_raceline.csv"),
    ("Monza", "racetracks/Monza.csv", "racetracks/Monza_raceline.csv"),
]

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Run headless racing simulation")
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

    results_table = []

    for track_name, track_path, raceline_path in TRACKS:
        print(f"Running simulation on {track_name}...")
        racetrack = RaceTrack(track_path, raceline_path)
        simulator = HeadlessSimulator(racetrack)
        results = simulator.run()
        results_table.append(
            {
                "track": track_name,
                "sim_time": results["sim_time_elapsed"],
                "violations": results["track_limit_violations"],
                "lap_finished": results["lap_finished"],
            }
        )

    # Print results table
    print("\n" + "=" * 55)
    print("SIMULATION RESULTS")
    print("=" * 55)
    print(f"{'Track':<12} {'Sim Time (s)':<15} {'Violations':<12} {'Finished':<10}")
    print("-" * 55)
    for row in results_table:
        finished = "Yes" if row["lap_finished"] else "No"
        print(f"{row['track']:<12} {row['sim_time']:<15.2f} {row['violations']:<12} {finished:<10}")
    print("=" * 55)
