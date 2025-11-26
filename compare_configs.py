"""
Compare all track-specific configurations against the base configuration.

Tests whether any single track-specific override outperforms the base when
applied universally across ALL tracks with 0 violations.

For each config (base + each track-specific override):
- Run all tracks
- Record time and violations
- Check if it beats base on every track with 0 violations
"""

import argparse
from dataclasses import dataclass
from pathlib import Path

from controller import ControllerConfig, ControllerParams, get_track_fingerprint
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


@dataclass
class TrackResult:
    track_name: str
    time: float
    violations: int
    finished: bool


@dataclass
class ConfigResults:
    config_name: str
    fingerprint: str
    results: list[TrackResult]
    total_time: float
    total_violations: int
    all_finished: bool

    @property
    def zero_violations(self) -> bool:
        return self.total_violations == 0


def run_simulation(racetrack: RaceTrack, params: ControllerParams) -> dict:
    """Run simulation and return results."""
    simulator = HeadlessSimulator(racetrack, ctrl_params=params)
    return simulator.run()


def extract_params_from_override(override: dict) -> ControllerParams:
    """Convert an override dict (with metadata) to ControllerParams."""
    # Filter out metadata keys (underscore-prefixed)
    params_only = {k: v for k, v in override.items() if not k.startswith("_")}
    return ControllerParams(**params_only)


def run_config_on_all_tracks(
    config_name: str,
    fingerprint: str,
    params: ControllerParams,
    tracks: list[tuple[str, str, str]],
    verbose: bool = True,
) -> ConfigResults:
    """Run a configuration on all tracks and return results."""
    results = []
    total_time = 0.0
    total_violations = 0
    all_finished = True

    if verbose:
        print(f"\n{'=' * 60}")
        print(f"Testing: {config_name}")
        print(f"{'=' * 60}")

    for track_name, track_path, raceline_path in tracks:
        if verbose:
            print(f"  {track_name}...", end=" ", flush=True)

        racetrack = RaceTrack(track_path, raceline_path)
        result = run_simulation(racetrack, params)

        track_result = TrackResult(
            track_name=track_name,
            time=result["sim_time_elapsed"],
            violations=result["track_limit_violations"],
            finished=result["lap_finished"],
        )
        results.append(track_result)

        total_time += track_result.time
        total_violations += track_result.violations
        if not track_result.finished:
            all_finished = False

        if verbose:
            status = "✓" if track_result.violations == 0 else f"✗ ({track_result.violations} viol)"
            print(f"{track_result.time:.2f}s {status}")

    if verbose:
        print(f"  Total: {total_time:.2f}s | {total_violations} violations")

    return ConfigResults(
        config_name=config_name,
        fingerprint=fingerprint,
        results=results,
        total_time=total_time,
        total_violations=total_violations,
        all_finished=all_finished,
    )


def compare_to_base(config_results: ConfigResults, base_results: ConfigResults) -> dict:
    """Compare a config's results to the base."""
    n_tracks = len(config_results.results)
    beats_base_count = 0
    ties_base_count = 0
    worse_than_base_count = 0

    track_comparisons = []

    for config_track, base_track in zip(config_results.results, base_results.results):
        time_diff = config_track.time - base_track.time
        viol_diff = config_track.violations - base_track.violations

        # Determine if this config beats base on this track
        # Beats = faster time OR same time with fewer violations
        if config_track.violations < base_track.violations:
            beats_base = True
        elif config_track.violations > base_track.violations:
            beats_base = False
        else:  # Same violations
            if time_diff < -0.1:  # More than 0.1s faster
                beats_base = True
            elif time_diff > 0.1:  # More than 0.1s slower
                beats_base = False
            else:
                beats_base = None  # Tie

        if beats_base is True:
            beats_base_count += 1
        elif beats_base is False:
            worse_than_base_count += 1
        else:
            ties_base_count += 1

        track_comparisons.append(
            {
                "track": config_track.track_name,
                "time_diff": time_diff,
                "viol_diff": viol_diff,
                "beats_base": beats_base,
            }
        )

    # A config "beats base universally" if it beats or ties on EVERY track
    # AND has zero violations everywhere
    beats_universally = worse_than_base_count == 0 and config_results.zero_violations and config_results.all_finished

    return {
        "beats_base_count": beats_base_count,
        "ties_base_count": ties_base_count,
        "worse_than_base_count": worse_than_base_count,
        "beats_universally": beats_universally,
        "track_comparisons": track_comparisons,
    }


def main():
    parser = argparse.ArgumentParser(description="Compare all track-specific configs against base")
    parser.add_argument(
        "--config",
        type=str,
        default=DEFAULT_CONFIG,
        help=f"Path to controller config JSON file (default: {DEFAULT_CONFIG})",
    )
    parser.add_argument(
        "--verbose",
        "-v",
        action="store_true",
        help="Show detailed per-track results",
    )
    args = parser.parse_args()

    # Load config
    if not Path(args.config).exists():
        print(f"Error: Config file not found: {args.config}")
        return

    config = ControllerConfig.from_file(args.config)
    print(f"Loaded config from {args.config}")
    print(f"  Base params + {len(config.overrides)} track-specific overrides\n")

    # First: Run base config on all tracks
    print("=" * 70)
    print("PHASE 1: Running BASE configuration on all tracks")
    print("=" * 70)

    base_results = run_config_on_all_tracks(
        config_name="BASE",
        fingerprint="base",
        params=config.base,
        tracks=ALL_TRACKS,
        verbose=args.verbose,
    )

    print(f"\nBase totals: {base_results.total_time:.2f}s | {base_results.total_violations} violations")

    # Second: Run each track-specific override on ALL tracks
    print("\n" + "=" * 70)
    print("PHASE 2: Testing each track-specific config on ALL tracks")
    print("=" * 70)

    all_config_results: list[tuple[ConfigResults, dict]] = []

    for fingerprint, override in config.overrides.items():
        # Get friendly name if available
        friendly_name = override.get("_friendly_name", fingerprint[:12])
        config_name = f"{friendly_name} ({fingerprint})"

        # Extract params (full override, not merged with base)
        params = extract_params_from_override(override)

        results = run_config_on_all_tracks(
            config_name=config_name,
            fingerprint=fingerprint,
            params=params,
            tracks=ALL_TRACKS,
            verbose=args.verbose,
        )

        comparison = compare_to_base(results, base_results)
        all_config_results.append((results, comparison))

    # Print summary table
    print("\n" + "=" * 100)
    print("COMPARISON SUMMARY")
    print("=" * 100)
    print(f"{'Config':<20} │ {'Total Time':>12} │ {'Violations':>10} │ {'vs Base':>20} │ {'Universal?':>10}")
    print("-" * 100)

    # Print base first
    print(
        f"{'BASE':<20} │ {base_results.total_time:>10.2f}s │ {base_results.total_violations:>10} │ {'-':>20} │ {'-':>10}"
    )
    print("-" * 100)

    universal_winners = []

    for results, comparison in all_config_results:
        # Shorten name if needed
        name = results.config_name
        if len(name) > 18:
            name = name[:15] + "..."

        time_diff = results.total_time - base_results.total_time
        time_diff_str = f"{time_diff:+.2f}s"

        vs_base = (
            f"↑{comparison['beats_base_count']} ↓{comparison['worse_than_base_count']} ={comparison['ties_base_count']}"
        )

        if comparison["beats_universally"]:
            universal = "✓ YES"
            universal_winners.append((results, comparison))
        else:
            universal = "✗ No"

        print(
            f"{name:<20} │ {results.total_time:>10.2f}s │ {results.total_violations:>10} │ {vs_base:>20} │ {universal:>10}"
        )

    print("=" * 100)

    # Print legend
    print("\nLegend: ↑ = beats base | ↓ = worse than base | = = ties base")
    print("Universal winner = beats/ties base on EVERY track with 0 violations\n")

    # Highlight universal winners
    if universal_winners:
        print("=" * 70)
        print("🏆 UNIVERSAL WINNERS (beat/tie base everywhere, 0 violations)")
        print("=" * 70)
        for results, comparison in universal_winners:
            print(f"\n  {results.config_name}")
            print(f"    Total time: {results.total_time:.2f}s (base: {base_results.total_time:.2f}s)")
            print(f"    Time diff: {results.total_time - base_results.total_time:+.2f}s")
            print(f"    Tracks beaten: {comparison['beats_base_count']}/{len(ALL_TRACKS)}")
    else:
        print("=" * 70)
        print("No universal winners found.")
        print("=" * 70)

        # Find the closest candidates
        print("\nClosest candidates (fewest tracks worse than base):")
        sorted_results = sorted(
            all_config_results, key=lambda x: (x[1]["worse_than_base_count"], x[0].total_violations)
        )
        for results, comparison in sorted_results[:5]:
            print(f"  {results.config_name}")
            print(f"    Worse on {comparison['worse_than_base_count']} tracks | {results.total_violations} violations")

    # Detailed track-by-track for best configs
    if args.verbose and all_config_results:
        print("\n" + "=" * 100)
        print("DETAILED TRACK-BY-TRACK COMPARISON (Top 3 configs)")
        print("=" * 100)

        # Sort by (violations, worse_count, total_time)
        sorted_results = sorted(
            all_config_results, key=lambda x: (x[0].total_violations, x[1]["worse_than_base_count"], x[0].total_time)
        )

        for results, comparison in sorted_results[:3]:
            print(f"\n{results.config_name}")
            print("-" * 80)
            print(f"{'Track':<14} │ {'Base':>10} │ {'Config':>10} │ {'Δ Time':>10} │ {'Δ Viol':>8} │ Status")
            print("-" * 80)

            for i, track_comp in enumerate(comparison["track_comparisons"]):
                base_time = base_results.results[i].time
                config_time = results.results[i].time
                base_viol = base_results.results[i].violations
                config_viol = results.results[i].violations

                time_diff = track_comp["time_diff"]
                viol_diff = track_comp["viol_diff"]

                if track_comp["beats_base"] is True:
                    status = "✓ Better"
                elif track_comp["beats_base"] is False:
                    status = "✗ Worse"
                else:
                    status = "= Tie"

                print(
                    f"{track_comp['track']:<14} │ "
                    f"{base_time:>8.2f}s │ "
                    f"{config_time:>8.2f}s │ "
                    f"{time_diff:>+8.2f}s │ "
                    f"{viol_diff:>+8} │ "
                    f"{status}"
                )


if __name__ == "__main__":
    main()
