import argparse

from controller import ControllerParams
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
        "--config",
        type=str,
        default=None,
        help="Path to controller config JSON file",
    )

    args = parser.parse_args()

    ctrl_params = None
    if args.config:
        ctrl_params = ControllerParams.from_file(args.config)
        print(f"Using controller parameters from {args.config}:\n{ctrl_params}")

    results_table = []

    for track_name, track_path, raceline_path in TRACKS:
        print(f"Running simulation on {track_name}...")
        racetrack = RaceTrack(track_path, raceline_path)
        simulator = HeadlessSimulator(racetrack, ctrl_params=ctrl_params)
        results = simulator.run()
        results_table.append(
            {
                "track": track_name,
                "sim_time": results["sim_time_elapsed"],
                "violations": results["track_limit_violations"],
            }
        )

    # Print results table
    print("\n" + "=" * 40)
    print("SIMULATION RESULTS")
    print("=" * 40)
    print(f"{'Track':<12} {'Sim Time (s)':<15} {'Violations':<10}")
    print("-" * 40)
    for row in results_table:
        print(f"{row['track']:<12} {row['sim_time']:<15.2f} {row['violations']:<10}")
    print("=" * 40)
