import argparse

from controller import ControllerParams
from racetrack import RaceTrack
from simulator_headless import HeadlessSimulator

# Original tracks
TRACKS = [
    ("IMS", "racetracks/IMS.csv"),
    ("Montreal", "racetracks/Montreal.csv"),
    ("Monza", "racetracks/Monza.csv"),
]

# TUM tracks (additional circuits)
TUM_TRACKS = [
    ("Austin", "tum_tracks/tracks/Austin.csv"),
    ("BrandsHatch", "tum_tracks/tracks/BrandsHatch.csv"),
    ("Budapest", "tum_tracks/tracks/Budapest.csv"),
    ("Catalunya", "tum_tracks/tracks/Catalunya.csv"),
    ("Hockenheim", "tum_tracks/tracks/Hockenheim.csv"),
    ("Melbourne", "tum_tracks/tracks/Melbourne.csv"),
    ("MexicoCity", "tum_tracks/tracks/MexicoCity.csv"),
    ("MoscowRaceway", "tum_tracks/tracks/MoscowRaceway.csv"),
    ("Norisring", "tum_tracks/tracks/Norisring.csv"),
    ("Nuerburgring", "tum_tracks/tracks/Nuerburgring.csv"),
    ("Oschersleben", "tum_tracks/tracks/Oschersleben.csv"),
    ("Sakhir", "tum_tracks/tracks/Sakhir.csv"),
    ("SaoPaulo", "tum_tracks/tracks/SaoPaulo.csv"),
    ("Sepang", "tum_tracks/tracks/Sepang.csv"),
    ("Shanghai", "tum_tracks/tracks/Shanghai.csv"),
    ("Silverstone", "tum_tracks/tracks/Silverstone.csv"),
    ("Sochi", "tum_tracks/tracks/Sochi.csv"),
    ("Spa", "tum_tracks/tracks/Spa.csv"),
    ("Spielberg", "tum_tracks/tracks/Spielberg.csv"),
    ("Suzuka", "tum_tracks/tracks/Suzuka.csv"),
    ("YasMarina", "tum_tracks/tracks/YasMarina.csv"),
    ("Zandvoort", "tum_tracks/tracks/Zandvoort.csv"),
]

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Run headless racing simulation")
    parser.add_argument(
        "--config",
        type=str,
        default=None,
        help="Path to controller config JSON file",
    )
    parser.add_argument(
        "--tum",
        action="store_true",
        help="Include TUM tracks (22 additional circuits)",
    )
    parser.add_argument(
        "--all",
        action="store_true",
        help="Run all available tracks",
    )

    args = parser.parse_args()

    ctrl_params = None
    if args.config:
        ctrl_params = ControllerParams.from_file(args.config)
        print(f"Using controller parameters from {args.config}:\n{ctrl_params}")

    # Select which tracks to run
    if args.all:
        tracks_to_run = TRACKS + TUM_TRACKS
    elif args.tum:
        tracks_to_run = TUM_TRACKS
    else:
        tracks_to_run = TRACKS

    results_table = []

    for track_name, track_path in tracks_to_run:
        print(f"Running simulation on {track_name}...")
        racetrack = RaceTrack(track_path)
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
