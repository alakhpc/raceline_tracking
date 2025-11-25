import argparse

from controller import ControllerParams
from racetrack import RaceTrack
from simulator_headless import HeadlessSimulator

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Run headless racing simulation")
    parser.add_argument("track", type=str, help="Path to track CSV file")
    parser.add_argument(
        "--config",
        type=str,
        default=None,
        help="Path to controller config JSON file",
    )

    args = parser.parse_args()

    racetrack = RaceTrack(args.track)

    ctrl_params = None
    if args.config:
        ctrl_params = ControllerParams.from_file(args.config)
        print(f"Using controller parameters from {args.config}:\n{ctrl_params}")

    simulator = HeadlessSimulator(racetrack, ctrl_params=ctrl_params)

    print("Running headless simulation...")
    results = simulator.run()

    print("\nSimulation Results:")
    print(f"  Lap completed: {results['lap_finished']}")
    print(f"  Wall time: {results['lap_time_elapsed']:.2f} seconds")
    print(f"  Sim time: {results['sim_time_elapsed']:.2f} seconds")
    print(f"  Track violations: {results['track_limit_violations']}")
    print(f"  Iterations: {results['iterations']}")
