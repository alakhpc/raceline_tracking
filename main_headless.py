import argparse
from pathlib import Path

from controller import ControllerConfig, get_track_fingerprint
from racetrack import RaceTrack
from simulator_headless import HeadlessSimulator

DEFAULT_CONFIG = "controller_config.json"

# All tracks: (name, track_path, raceline_path)
ALL_TRACKS = [
    # Core tracks
    ("IMS", "racetracks/IMS.csv", "racetracks/IMS_raceline.csv"),
    ("Montreal", "racetracks/Montreal.csv", "racetracks/Montreal_raceline.csv"),
    ("Monza", "racetracks/Monza.csv", "racetracks/Monza_raceline.csv"),
    # TUM tracks
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


def run_simulation(racetrack: RaceTrack, ctrl_params) -> dict:
    """Run simulation and return results."""
    simulator = HeadlessSimulator(racetrack, ctrl_params=ctrl_params)
    return simulator.run()


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Run headless racing simulation on all tracks")
    parser.add_argument(
        "--config",
        type=str,
        default=DEFAULT_CONFIG,
        help=f"Path to controller config JSON file (default: {DEFAULT_CONFIG})",
    )

    args = parser.parse_args()

    # Load config (or use defaults)
    config = None
    if Path(args.config).exists():
        config = ControllerConfig.from_file(args.config)
        print(f"Loaded config from {args.config}")
        print(f"  Base params + {len(config.overrides)} track-specific overrides\n")
    else:
        print(f"Config file not found: {args.config}, using defaults\n")
        config = ControllerConfig()  # Default config

    results_table = []

    for track_name, track_path, raceline_path in ALL_TRACKS:
        print(f"Running simulation on {track_name}...", end=" ", flush=True)
        racetrack = RaceTrack(track_path, raceline_path)
        fingerprint = get_track_fingerprint(racetrack)

        # Always run with base params
        base_params = config.base
        base_result = run_simulation(racetrack, base_params)

        # Check if track has specific override
        has_override = fingerprint in config.overrides

        if has_override:
            # Run with track-specific params
            specific_params = config.get_params(track=racetrack)
            specific_result = run_simulation(racetrack, specific_params)
            print("(base + specific)")
        else:
            specific_result = None
            print("(base only)")

        results_table.append(
            {
                "track": track_name,
                "base_time": base_result["sim_time_elapsed"],
                "base_violations": base_result["track_limit_violations"],
                "base_finished": base_result["lap_finished"],
                "specific_time": specific_result["sim_time_elapsed"] if specific_result else None,
                "specific_violations": specific_result["track_limit_violations"] if specific_result else None,
                "specific_finished": specific_result["lap_finished"] if specific_result else None,
                "has_override": has_override,
            }
        )

    # Print results table
    print("\n" + "=" * 100)
    print("SIMULATION RESULTS")
    print("=" * 100)
    header = f"{'Track':<14} │ {'Base Time':>10} {'Viol':>5} │ {'Spec Time':>10} {'Viol':>5} │ {'Δ Time':>10}"
    print(header)
    print("-" * 100)

    total_base_time = 0
    total_base_violations = 0
    total_specific_time = 0
    total_specific_violations = 0
    n_with_specific = 0
    n_improved = 0
    n_regressed = 0

    for row in results_table:
        base_time_str = f"{row['base_time']:.2f}s"
        base_viol_str = str(row["base_violations"])

        if row["has_override"] and row["specific_time"] is not None:
            specific_time_str = f"{row['specific_time']:.2f}s"
            specific_viol_str = str(row["specific_violations"])

            # Calculate improvement (negative = faster = better)
            time_diff = row["specific_time"] - row["base_time"]
            pct_change = (time_diff / row["base_time"]) * 100 if row["base_time"] > 0 else 0

            if pct_change < -0.1:  # Improved by more than 0.1%
                delta_str = f"{pct_change:+.1f}% ✓"
                n_improved += 1
            elif pct_change > 0.1:  # Regressed by more than 0.1%
                delta_str = f"{pct_change:+.1f}% ✗"
                n_regressed += 1
            else:
                delta_str = f"{pct_change:+.1f}%"

            n_with_specific += 1
            total_specific_time += row["specific_time"]
            total_specific_violations += row["specific_violations"]
        else:
            specific_time_str = "-"
            specific_viol_str = "-"
            delta_str = "-"

        total_base_time += row["base_time"]
        total_base_violations += row["base_violations"]

        print(
            f"{row['track']:<14} │ {base_time_str:>10} {base_viol_str:>5} │ "
            f"{specific_time_str:>10} {specific_viol_str:>5} │ {delta_str:>10}"
        )

    print("=" * 100)

    # Summary
    print(f"\nBase totals: {total_base_time:.2f}s | {total_base_violations} violations")

    if n_with_specific > 0:
        print(
            f"Specific totals ({n_with_specific} tracks): {total_specific_time:.2f}s | {total_specific_violations} violations"
        )
        print(
            f"  Improved: {n_improved} | Regressed: {n_regressed} | Unchanged: {n_with_specific - n_improved - n_regressed}"
        )
