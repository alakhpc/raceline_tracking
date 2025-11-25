import argparse

from controller import ControllerParams
from racetrack import RaceTrack
from simulator import Simulator, plt

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Run racing simulator with visualization")
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

    simulator = Simulator(racetrack, ctrl_params=ctrl_params)
    simulator.start()
    plt.show()
