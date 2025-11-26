import argparse

from controller import ControllerParams
from racetrack import RaceTrack
from simulator_headless import HeadlessSimulator

# Original tracks: (name, track_path, raceline_path)
TRACKS = [
    ("IMS", "racetracks/IMS.csv", "racetracks/IMS_raceline.csv"),
    ("Montreal", "racetracks/Montreal.csv", "racetracks/Montreal_raceline.csv"),
    ("Monza", "racetracks/Monza.csv", "racetracks/Monza_raceline.csv"),
]

# TUM tracks (additional circuits): (name, track_path, raceline_path)
TUM_TRACKS = [
    ("Austin", "tum_tracks/tracks/Austin.csv", "tum_tracks/racelines/Austin.csv"),
    ("BrandsHatch", "tum_tracks/tracks/BrandsHatch.csv", "tum_tracks/racelines/BrandsHatch.csv"),
    ("Budapest", "tum_tracks/tracks/Budapest.csv", "tum_tracks/racelines/Budapest.csv"),
    ("Catalunya", "tum_tracks/tracks/Catalunya.csv", "tum_tracks/racelines/Catalunya.csv"),
    ("Hockenheim", "tum_tracks/tracks/Hockenheim.csv", "tum_tracks/racelines/Hockenheim.csv"),
    ("Melbourne", "tum_tracks/tracks/Melbourne.csv", "tum_tracks/racelines/Melbourne.csv"),
    ("MexicoCity", "tum_tracks/tracks/MexicoCity.csv", "tum_tracks/racelines/MexicoCity.csv"),
    ("MoscowRaceway", "tum_tracks/tracks/MoscowRaceway.csv", "tum_tracks/racelines/MoscowRaceway.csv"),
    ("Norisring", "tum_tracks/tracks/Norisring.csv", "tum_tracks/racelines/Norisring.csv"),
    ("Nuerburgring", "tum_tracks/tracks/Nuerburgring.csv", "tum_tracks/racelines/Nuerburgring.csv"),
    ("Oschersleben", "tum_tracks/tracks/Oschersleben.csv", "tum_tracks/racelines/Oschersleben.csv"),
    ("Sakhir", "tum_tracks/tracks/Sakhir.csv", "tum_tracks/racelines/Sakhir.csv"),
    ("SaoPaulo", "tum_tracks/tracks/SaoPaulo.csv", "tum_tracks/racelines/SaoPaulo.csv"),
    ("Sepang", "tum_tracks/tracks/Sepang.csv", "tum_tracks/racelines/Sepang.csv"),
    ("Shanghai", "tum_tracks/tracks/Shanghai.csv", "tum_tracks/racelines/Shanghai.csv"),
    ("Silverstone", "tum_tracks/tracks/Silverstone.csv", "tum_tracks/racelines/Silverstone.csv"),
    ("Sochi", "tum_tracks/tracks/Sochi.csv", "tum_tracks/racelines/Sochi.csv"),
    ("Spa", "tum_tracks/tracks/Spa.csv", "tum_tracks/racelines/Spa.csv"),
    ("Spielberg", "tum_tracks/tracks/Spielberg.csv", "tum_tracks/racelines/Spielberg.csv"),
    ("Suzuka", "tum_tracks/tracks/Suzuka.csv", "tum_tracks/racelines/Suzuka.csv"),
    ("YasMarina", "tum_tracks/tracks/YasMarina.csv", "tum_tracks/racelines/YasMarina.csv"),
    ("Zandvoort", "tum_tracks/tracks/Zandvoort.csv", "tum_tracks/racelines/Zandvoort.csv"),
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
        help="Include TUM tracks (runs all 25 tracks: 3 original + 22 TUM)",
    )

    args = parser.parse_args()

    ctrl_params = None
    if args.config:
        ctrl_params = ControllerParams.from_file(args.config)
        print(f"Using controller parameters from {args.config}:\n{ctrl_params}")

    # Select which tracks to run
    if args.tum:
        tracks_to_run = TRACKS + TUM_TRACKS
    else:
        tracks_to_run = TRACKS

    results_table = []

    for track_name, track_path, raceline_path in tracks_to_run:
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
