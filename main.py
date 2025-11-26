import argparse
from pathlib import Path

from controller import ControllerConfig, get_track_fingerprint
from racetrack import RaceTrack
from simulator import Simulator, plt

DEFAULT_CONFIG = "controller_config.json"

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Run racing simulator with visualization")
    parser.add_argument("track", type=str, help="Path to track CSV file")
    parser.add_argument("raceline", type=str, help="Path to raceline CSV file")
    parser.add_argument(
        "--config",
        type=str,
        default=DEFAULT_CONFIG,
        help=f"Path to controller config JSON file (default: {DEFAULT_CONFIG})",
    )

    args = parser.parse_args()

    racetrack = RaceTrack(args.track, args.raceline)
    fingerprint = get_track_fingerprint(racetrack)

    ctrl_params = None
    if Path(args.config).exists():
        config = ControllerConfig.from_file(args.config)
        ctrl_params = config.get_params(track=racetrack)
        has_override = fingerprint in config.overrides
        override_info = f" (track-specific: {fingerprint})" if has_override else " (base)"
        print(f"Using config from {args.config}{override_info}:\n{ctrl_params}")
    else:
        print(f"Config file not found: {args.config}, using defaults")

    simulator = Simulator(racetrack, ctrl_params=ctrl_params)
    simulator.start()
    plt.show()
