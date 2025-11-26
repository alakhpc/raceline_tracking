"""
CMA-ES training script for controller parameter optimization.

Training modes:
- base: Train on core tracks (IMS, Montreal, Monza) + Shanghai for generalization
- <track_name>: Train on a specific single track (e.g., Austin, Spa)

Configuration is saved to controller_config.json in the format:
{
    "base": { ...base parameters... },
    "<fingerprint>": { ...track-specific overrides... }
}
"""

import argparse
import os
from dataclasses import asdict
from pathlib import Path

import cma
import numpy as np

from controller import ControllerConfig, ControllerParams, get_fingerprint_from_path
from racetrack import RaceTrack
from simulator_headless import HeadlessSimulator

# =============================================================================
# CONFIGURATION
# =============================================================================

PARAM_BOUNDS = {
    "lookahead_base": (5.0, 30.0),
    "lookahead_gain": (0.1, 1.0),
    "v_max": (90.0, 100.0),
    "k_curvature": (40.0, 200.0),
    "brake_lookahead": (150.0, 280.0),
    "v_min": (10.0, 20.0),
    "kp_steer": (3.0, 6.0),
    "kp_vel": (1.5, 8.0),
    "decel_factor": (0.5, 0.95),
    "steer_anticipation": (0.8, 3.0),
    "raceline_blend": (0.2, 0.8),
    "straight_lookahead_mult": (1.2, 3.0),
    "corner_exit_boost": (1.2, 1.8),
}

PARAM_NAMES = list(PARAM_BOUNDS.keys())

# All available tracks: name -> (track_path, raceline_path)
ALL_TRACKS = {
    # Core tracks
    "IMS": ("racetracks/IMS.csv", "racetracks/IMS_raceline.csv"),
    "Montreal": ("racetracks/Montreal.csv", "racetracks/Montreal_raceline.csv"),
    "Monza": ("racetracks/Monza.csv", "racetracks/Monza_raceline.csv"),
    # TUM tracks
    "Austin": ("tum_tracks/tracks/Austin.csv", "tum_tracks/racelines/Austin.csv"),
    "BrandsHatch": ("tum_tracks/tracks/BrandsHatch.csv", "tum_tracks/racelines/BrandsHatch.csv"),
    "Budapest": ("tum_tracks/tracks/Budapest.csv", "tum_tracks/racelines/Budapest.csv"),
    "Catalunya": ("tum_tracks/tracks/Catalunya.csv", "tum_tracks/racelines/Catalunya.csv"),
    "Hockenheim": ("tum_tracks/tracks/Hockenheim.csv", "tum_tracks/racelines/Hockenheim.csv"),
    "Melbourne": ("tum_tracks/tracks/Melbourne.csv", "tum_tracks/racelines/Melbourne.csv"),
    "MexicoCity": ("tum_tracks/tracks/MexicoCity.csv", "tum_tracks/racelines/MexicoCity.csv"),
    "MoscowRaceway": ("tum_tracks/tracks/MoscowRaceway.csv", "tum_tracks/racelines/MoscowRaceway.csv"),
    "Norisring": ("tum_tracks/tracks/Norisring.csv", "tum_tracks/racelines/Norisring.csv"),
    "Nuerburgring": ("tum_tracks/tracks/Nuerburgring.csv", "tum_tracks/racelines/Nuerburgring.csv"),
    "Oschersleben": ("tum_tracks/tracks/Oschersleben.csv", "tum_tracks/racelines/Oschersleben.csv"),
    "Sakhir": ("tum_tracks/tracks/Sakhir.csv", "tum_tracks/racelines/Sakhir.csv"),
    "SaoPaulo": ("tum_tracks/tracks/SaoPaulo.csv", "tum_tracks/racelines/SaoPaulo.csv"),
    "Sepang": ("tum_tracks/tracks/Sepang.csv", "tum_tracks/racelines/Sepang.csv"),
    "Shanghai": ("tum_tracks/tracks/Shanghai.csv", "tum_tracks/racelines/Shanghai.csv"),
    "Silverstone": ("tum_tracks/tracks/Silverstone.csv", "tum_tracks/racelines/Silverstone.csv"),
    "Sochi": ("tum_tracks/tracks/Sochi.csv", "tum_tracks/racelines/Sochi.csv"),
    "Spa": ("tum_tracks/tracks/Spa.csv", "tum_tracks/racelines/Spa.csv"),
    "Spielberg": ("tum_tracks/tracks/Spielberg.csv", "tum_tracks/racelines/Spielberg.csv"),
    "Suzuka": ("tum_tracks/tracks/Suzuka.csv", "tum_tracks/racelines/Suzuka.csv"),
    "YasMarina": ("tum_tracks/tracks/YasMarina.csv", "tum_tracks/racelines/YasMarina.csv"),
    "Zandvoort": ("tum_tracks/tracks/Zandvoort.csv", "tum_tracks/racelines/Zandvoort.csv"),
}

BASE_TRAINING_TRACKS = ["IMS", "Montreal", "Monza", "Shanghai"]
CONFIG_FILE = "controller_config.json"
VIOLATION_PENALTY = 20.0

# =============================================================================
# GENOME CONVERSION
# =============================================================================


def genome_to_params(genome: np.ndarray) -> ControllerParams:
    """Convert a normalized genome [0, 1]^N to ControllerParams."""
    params = {
        name: PARAM_BOUNDS[name][0] + np.clip(genome[i], 0.0, 1.0) * (PARAM_BOUNDS[name][1] - PARAM_BOUNDS[name][0])
        for i, name in enumerate(PARAM_NAMES)
    }
    return ControllerParams(**params)


def params_to_genome(params: ControllerParams) -> np.ndarray:
    """Convert ControllerParams to a normalized genome [0, 1]^N."""
    params_dict = asdict(params)
    return np.array(
        [
            (params_dict[name] - PARAM_BOUNDS[name][0]) / (PARAM_BOUNDS[name][1] - PARAM_BOUNDS[name][0])
            for name in PARAM_NAMES
        ]
    )


# =============================================================================
# FITNESS EVALUATION
# =============================================================================


def evaluate_single_track(ctrl_params: ControllerParams, track_path: str, raceline_path: str) -> float:
    """Run simulation on a single track and return fitness score."""
    track = RaceTrack(track_path, raceline_path)
    sim = HeadlessSimulator(track, ctrl_params=ctrl_params)
    results = sim.run(stop_on_violation=False)

    if results["lap_finished"]:
        return results["sim_time_elapsed"] + results["track_limit_violations"] * VIOLATION_PENALTY
    return 500.0 + results["track_limit_violations"] * 10.0


def evaluate_fitness_base(genome: np.ndarray) -> float:
    """Evaluate fitness for base training mode (core tracks + Shanghai)."""
    ctrl_params = genome_to_params(genome)
    total = sum(evaluate_single_track(ctrl_params, *ALL_TRACKS[name]) for name in BASE_TRAINING_TRACKS)
    return total / len(BASE_TRAINING_TRACKS)


class SingleTrackEvaluator:
    """Picklable evaluator for a single track (needed for multiprocessing)."""

    def __init__(self, track_name: str):
        self.track_path, self.raceline_path = ALL_TRACKS[track_name]

    def __call__(self, genome: np.ndarray) -> float:
        return evaluate_single_track(genome_to_params(genome), self.track_path, self.raceline_path)


def evaluate_detailed(params: ControllerParams, track_names: list[str]) -> dict:
    """Evaluate on tracks and return detailed per-track results."""
    results = {}
    for name in track_names:
        track_path, raceline_path = ALL_TRACKS[name]
        track = RaceTrack(track_path, raceline_path)
        sim = HeadlessSimulator(track, ctrl_params=params)
        result = sim.run(stop_on_violation=False)
        results[name] = {
            "time": result["sim_time_elapsed"],
            "violations": result["track_limit_violations"],
            "finished": result["lap_finished"],
        }
    return results


# =============================================================================
# MAIN TRAINING
# =============================================================================


def save_config(config: ControllerConfig, params: ControllerParams, fingerprint: str | None):
    """Save params to config - base if fingerprint is None, otherwise as override."""
    if fingerprint is None:
        config.base = params
    else:
        config.set_full_override(fingerprint, params)
    config.to_file(CONFIG_FILE)


def main():
    parser = argparse.ArgumentParser(
        description="Train controller using CMA-ES optimization.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=f"""
Training modes:
  base          Train on core tracks ({", ".join(BASE_TRAINING_TRACKS)})
  <track_name>  Train on a specific single track

Available tracks: {", ".join(ALL_TRACKS.keys())}

Examples:
  %(prog)s --generations 100 --mode base
  %(prog)s --generations 50 --mode Austin --resume
        """,
    )
    parser.add_argument("--generations", type=int, required=True)
    parser.add_argument("--workers", type=int, default=0, help="0 = auto-detect CPU count")
    parser.add_argument("--popsize", type=int, default=15)
    parser.add_argument(
        "--mode",
        choices=["base"] + list(ALL_TRACKS.keys()),
        default="base",
        help="Training mode: 'base' for core tracks, or a specific track name",
    )
    parser.add_argument(
        "--resume",
        action="store_true",
        help="Resume from existing config instead of defaults",
    )
    args = parser.parse_args()

    is_base = args.mode == "base"
    track_name = None if is_base else args.mode
    fingerprint = None if is_base else get_fingerprint_from_path(ALL_TRACKS[track_name][0])
    n_workers = args.workers or os.cpu_count() or 1

    # Header
    print("=" * 70)
    print("CMA-ES Controller Optimization")
    print("=" * 70)
    if is_base:
        print(f"Mode: BASE ({len(BASE_TRAINING_TRACKS)} tracks: {', '.join(BASE_TRAINING_TRACKS)})")
    else:
        print(f"Mode: SINGLE TRACK ({track_name}) | Fingerprint: {fingerprint}")
    print(f"Generations: {args.generations} | Workers: {n_workers} | Violation penalty: {VIOLATION_PENALTY}s\n")

    # Load or create config
    config = ControllerConfig.from_file(CONFIG_FILE) if Path(CONFIG_FILE).exists() else ControllerConfig()
    print(f"{'Loaded' if Path(CONFIG_FILE).exists() else 'Created'} config: {CONFIG_FILE}")

    # Initial parameters
    if args.resume:
        initial_params = config.base if is_base else config.get_params(fingerprint=fingerprint)
        sigma0 = 0.15
        print(f"Resuming from {'base' if is_base else fingerprint}...")
    else:
        initial_params = ControllerParams()
        sigma0 = 0.3

    print(f"Initial params:\n{initial_params}\n")

    # CMA-ES setup
    opts = {"maxiter": args.generations, "bounds": [0.0, 1.0], "verbose": -1, "verb_disp": 0}
    if args.popsize:
        opts["popsize"] = args.popsize

    es = cma.CMAEvolutionStrategy(params_to_genome(initial_params), sigma0, opts)
    evaluate_fn = evaluate_fitness_base if is_base else SingleTrackEvaluator(track_name)

    print(f"Population size: {es.popsize}")
    print("Starting optimization...\n")

    best_fitness, best_genome = float("inf"), None

    with cma.fitness_transformations.EvalParallel2(evaluate_fn, n_workers) as eval_parallel:
        for generation in range(1, args.generations + 1):
            solutions = es.ask()
            fitnesses = eval_parallel(solutions)

            for sol, fit in zip(solutions, fitnesses):
                if fit < best_fitness:
                    best_fitness, best_genome = fit, sol.copy()

            es.tell(solutions, fitnesses)

            improved = "★" if best_fitness == min(fitnesses) else " "
            print(
                f"Gen {generation:4d} {improved} | Best: {best_fitness:7.2f} | "
                f"Mean: {np.mean(fitnesses):7.2f} | Min: {np.min(fitnesses):7.2f}"
            )

            if best_genome is not None:
                save_config(config, genome_to_params(best_genome), fingerprint)

            if es.stop():
                break

    # Final results
    print("\n" + "=" * 70)
    print("Optimization Complete!")
    print("=" * 70)

    best_params = genome_to_params(best_genome if best_genome is not None else es.result.xbest)
    print(f"\nBest fitness: {best_fitness:.2f}\n\nBest parameters:\n{best_params}")

    save_config(config, best_params, fingerprint)
    print(f"\nConfig saved to {CONFIG_FILE}")

    # Detailed evaluation
    print("\n" + "-" * 70)
    print("Per-Track Results:")
    print("-" * 70)

    eval_tracks = BASE_TRAINING_TRACKS if is_base else [track_name]
    results = evaluate_detailed(best_params, eval_tracks)

    total_time, total_violations = 0, 0
    for name, r in results.items():
        status = "✓" if r["finished"] and r["violations"] == 0 else "⚠" if r["finished"] else "✗"
        print(f"  {status} {name:20s} | Time: {r['time']:6.2f}s | Violations: {r['violations']}")
        total_time += r["time"]
        total_violations += r["violations"]

    print("-" * 70)
    print(f"  Total: {total_time:.2f}s | {total_violations} violations")

    if fingerprint:
        print(f"\nTrack fingerprint saved: {fingerprint}")


if __name__ == "__main__":
    main()
