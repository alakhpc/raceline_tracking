#!/usr/bin/env python3
import argparse
import sys

import matplotlib.pyplot as plt

from simulator import HeadlessSimulator, RaceTrack, Simulator


def main():
    parser = argparse.ArgumentParser(description="Raceline tracking simulator")
    parser.add_argument("track", help="Path to track CSV file")
    parser.add_argument("raceline", help="Path to raceline CSV file")
    parser.add_argument(
        "--headless",
        action="store_true",
        help="Run in headless mode (no visualization, fast simulation)",
    )
    parser.add_argument(
        "--max-time",
        type=float,
        default=300.0,
        help="Maximum simulation time in seconds (default: 300)",
    )

    args = parser.parse_args()

    racetrack = RaceTrack(args.track)

    if args.headless:
        simulator = HeadlessSimulator(racetrack, max_time=args.max_time)
        report = simulator.run()
        report.print()
        sys.exit(0 if report.completed else 1)
    else:
        simulator = Simulator(racetrack)
        simulator.start()
        plt.show()


if __name__ == "__main__":
    main()
