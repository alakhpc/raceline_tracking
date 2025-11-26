"""
CMA-ES training script for controller parameter optimization.

Optimized for evaluation on: IMS, Montreal, Monza + 2 random tracks.

Strategy:
- Core tracks (IMS, Montreal, Monza) weighted 3x since they're guaranteed
- Diverse selection of other tracks for generalization to unknowns
- Tracks categorized by characteristics for balanced coverage
"""

import argparse
import os
from dataclasses import asdict
from pathlib import Path

import cma
import numpy as np

from controller import ControllerParams
from racetrack import RaceTrack
from simulator_headless import HeadlessSimulator

# =============================================================================
# PARAMETER BOUNDS
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

# =============================================================================
# TRACK DEFINITIONS - Categorized for optimal training
# =============================================================================

# Format: (name, track_path, raceline_path, weight)
# Weight determines importance in fitness calculation

# CORE TRACKS - These are guaranteed in evaluation, weight heavily
CORE_TRACKS = [
    ("IMS", "racetracks/IMS.csv", "racetracks/IMS_raceline.csv", 3.0),
    # ("Montreal", "racetracks/Montreal.csv", "racetracks/Montreal_raceline.csv", 3.0),
    # ("Monza", "racetracks/Monza.csv", "racetracks/Monza_raceline.csv", 3.0),
]

# HIGH-SPEED TRACKS - Long straights, fast corners (helps generalize Monza-like tracks)
HIGH_SPEED_TRACKS = [
    ("Spa", "tum_tracks/tracks/Spa.csv", "tum_tracks/racelines/Spa.csv", 1.0),
    ("Silverstone", "tum_tracks/tracks/Silverstone.csv", "tum_tracks/racelines/Silverstone.csv", 1.0),
]

# TECHNICAL TRACKS - Tight corners, chicanes (helps generalize Montreal-like tracks)
TECHNICAL_TRACKS = [
    ("Budapest", "tum_tracks/tracks/Budapest.csv", "tum_tracks/racelines/Budapest.csv", 1.0),
    ("Zandvoort", "tum_tracks/tracks/Zandvoort.csv", "tum_tracks/racelines/Zandvoort.csv", 1.0),
    ("Norisring", "tum_tracks/tracks/Norisring.csv", "tum_tracks/racelines/Norisring.csv", 1.0),
]

# MIXED/COMPLEX TRACKS - Variety of corner types (helps generalize IMS-like tracks)
MIXED_TRACKS = [
    ("Shanghai", "tum_tracks/tracks/Shanghai.csv", "tum_tracks/racelines/Shanghai.csv", 1.5),
    ("Suzuka", "tum_tracks/tracks/Suzuka.csv", "tum_tracks/racelines/Suzuka.csv", 1.0),
    ("Austin", "tum_tracks/tracks/Austin.csv", "tum_tracks/racelines/Austin.csv", 1.0),
]

# STREET-LIKE TRACKS - Narrow, unforgiving (edge cases)
STREET_TRACKS = [
    ("Melbourne", "tum_tracks/tracks/Melbourne.csv", "tum_tracks/racelines/Melbourne.csv", 1.0),
    ("Sochi", "tum_tracks/tracks/Sochi.csv", "tum_tracks/racelines/Sochi.csv", 1.0),
]

# =============================================================================
# TRAINING PRESETS
# =============================================================================

PRESETS = {
    # Minimal: Just core tracks + 1 from each category (6 tracks)
    # Good for quick iteration
    "minimal": CORE_TRACKS
    + [
        HIGH_SPEED_TRACKS[0],  # Spa
        TECHNICAL_TRACKS[0],  # Budapest
        MIXED_TRACKS[0],  # Suzuka
    ],
    # Balanced: Core + good coverage (9 tracks) - RECOMMENDED
    # Best trade-off between speed and generalization
    "balanced": CORE_TRACKS
    + [
        HIGH_SPEED_TRACKS[0],  # Spa
        HIGH_SPEED_TRACKS[1],  # Silverstone
        TECHNICAL_TRACKS[0],  # Budapest
        TECHNICAL_TRACKS[1],  # Zandvoort
        MIXED_TRACKS[0],  # Suzuka
        MIXED_TRACKS[1],  # Austin
    ],
    # Comprehensive: Core + broad coverage (12 tracks)
    # For final training when you want maximum generalization
    "comprehensive": CORE_TRACKS
    + [
        HIGH_SPEED_TRACKS[0],
        HIGH_SPEED_TRACKS[1],
        TECHNICAL_TRACKS[0],
        TECHNICAL_TRACKS[1],
        TECHNICAL_TRACKS[2],
        MIXED_TRACKS[0],
        MIXED_TRACKS[1],
        STREET_TRACKS[0],
        STREET_TRACKS[1],
    ],
    # Core only: Just the 3 guaranteed tracks
    # Fast but may not generalize to random tracks
    "core": CORE_TRACKS,
}

# Violation penalty (seconds added per violation)
VIOLATION_PENALTY = 20.0  # High penalty to prioritize clean laps

# =============================================================================
# GENOME CONVERSION
# =============================================================================


def genome_to_params(genome: np.ndarray) -> ControllerParams:
    """Convert a normalized genome [0, 1]^N to ControllerParams."""
    params = {}
    for i, name in enumerate(PARAM_NAMES):
        low, high = PARAM_BOUNDS[name]
        val = np.clip(genome[i], 0.0, 1.0)
        params[name] = low + val * (high - low)
    return ControllerParams(**params)


def params_to_genome(params: ControllerParams) -> np.ndarray:
    """Convert ControllerParams to a normalized genome [0, 1]^N."""
    genome = []
    params_dict = asdict(params)
    for name in PARAM_NAMES:
        low, high = PARAM_BOUNDS[name]
        val = (params_dict[name] - low) / (high - low)
        genome.append(val)
    return np.array(genome)


# =============================================================================
# FITNESS EVALUATION
# =============================================================================


def make_evaluate_fitness(track_list: list, violation_penalty: float):
    """
    Create a fitness evaluation function with tracks baked in.
    This is needed for multiprocessing - the tracks must be defined
    in the function itself, not passed via globals.
    """
    # Capture track_list in closure - but this won't work with multiprocessing
    # So we'll just use the balanced preset directly in evaluate_fitness
    pass


def evaluate_fitness(genome: np.ndarray) -> float:
    """
    Evaluate fitness using weighted track scores.

    Uses the balanced preset by default for multiprocessing compatibility.
    """
    ctrl_params = genome_to_params(genome)

    # Hardcode balanced preset for multiprocessing compatibility
    track_list = [
        ("IMS", "racetracks/IMS.csv", "racetracks/IMS_raceline.csv", 3.0),
        ("Montreal", "racetracks/Montreal.csv", "racetracks/Montreal_raceline.csv", 3.0),
        ("Monza", "racetracks/Monza.csv", "racetracks/Monza_raceline.csv", 3.0),
        ("Spa", "tum_tracks/tracks/Spa.csv", "tum_tracks/racelines/Spa.csv", 1.0),
        ("Silverstone", "tum_tracks/tracks/Silverstone.csv", "tum_tracks/racelines/Silverstone.csv", 1.0),
        ("Budapest", "tum_tracks/tracks/Budapest.csv", "tum_tracks/racelines/Budapest.csv", 1.0),
        ("Zandvoort", "tum_tracks/tracks/Zandvoort.csv", "tum_tracks/racelines/Zandvoort.csv", 1.0),
        ("Suzuka", "tum_tracks/tracks/Suzuka.csv", "tum_tracks/racelines/Suzuka.csv", 1.5),
        ("Austin", "tum_tracks/tracks/Austin.csv", "tum_tracks/racelines/Austin.csv", 1.0),
    ]

    total_weighted_fitness = 0.0
    total_weight = 0.0

    for name, track_path, raceline_path, weight in track_list:
        track = RaceTrack(track_path, raceline_path)
        sim = HeadlessSimulator(track, ctrl_params=ctrl_params)
        results = sim.run(stop_on_violation=False)

        if results["lap_finished"]:
            fitness = results["sim_time_elapsed"]
            fitness += results["track_limit_violations"] * VIOLATION_PENALTY
        else:
            # DNF: heavy penalty
            fitness = 500.0 + results["track_limit_violations"] * 10.0

        total_weighted_fitness += fitness * weight
        total_weight += weight

    return total_weighted_fitness / total_weight


def evaluate_detailed(params: ControllerParams, track_list: list) -> dict:
    """Evaluate on tracks and return detailed per-track results."""
    results = {}
    for name, track_path, raceline_path, weight in track_list:
        track = RaceTrack(track_path, raceline_path)
        sim = HeadlessSimulator(track, ctrl_params=params)
        result = sim.run(stop_on_violation=False)
        results[name] = {
            "time": result["sim_time_elapsed"],
            "violations": result["track_limit_violations"],
            "finished": result["lap_finished"],
            "weight": weight,
        }
    return results


# =============================================================================
# MAIN TRAINING
# =============================================================================


def main():
    parser = argparse.ArgumentParser(
        description="Train controller using CMA-ES optimization.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Presets (optimized for evaluation on IMS, Montreal, Monza + 2 random):
  core          3 tracks  - Just guaranteed tracks (fast, may not generalize)
  minimal       6 tracks  - Core + 1 from each category (good for iteration)
  balanced      9 tracks  - Core + good coverage (RECOMMENDED)
  comprehensive 12 tracks - Maximum generalization (slow but robust)

Examples:
  %(prog)s --generations 100 --preset balanced
  %(prog)s --generations 50 --preset minimal --resume
        """,
    )
    parser.add_argument("--generations", type=int, required=True)
    parser.add_argument("--workers", type=int, default=0, help="0 = auto-detect CPU count")
    parser.add_argument("--popsize", type=int, default=None)
    parser.add_argument(
        "--preset",
        choices=list(PRESETS.keys()),
        default="balanced",
        help="Track selection preset (default: balanced)",
    )
    parser.add_argument(
        "--resume",
        action="store_true",
        help="Resume from cmaes_winner.json instead of defaults",
    )
    args = parser.parse_args()

    # Select tracks based on preset (for display only - evaluate_fitness uses balanced)
    track_list = PRESETS[args.preset]

    if args.preset != "balanced":
        print(f"Note: --preset {args.preset} is for display only.")
        print("      Fitness evaluation always uses 'balanced' preset for multiprocessing.")
        print()

    # Determine workers
    n_workers = os.cpu_count() or 1 if args.workers == 0 else args.workers

    # Calculate total weight for display
    total_weight = sum(t[3] for t in track_list)
    core_weight = sum(t[3] for t in track_list if t[3] >= 3.0)

    print("=" * 70)
    print("CMA-ES Controller Optimization")
    print("=" * 70)
    print(f"Preset: {args.preset} ({len(track_list)} tracks)")
    print(f"Generations: {args.generations}")
    print(f"Workers: {n_workers}")
    print(f"Violation penalty: {VIOLATION_PENALTY}s")
    print(f"Core track weight: {core_weight:.0f}/{total_weight:.1f} ({100 * core_weight / total_weight:.0f}%)")
    print("\nTrack selection:")
    for name, _, _, weight in track_list:
        weight_str = f" [weight: {weight}x]" if weight != 1.0 else ""
        core_str = " ★ CORE" if weight >= 3.0 else ""
        print(f"  • {name}{weight_str}{core_str}")
    print()

    # Initial genome - resume or start fresh
    if args.resume and Path("cmaes_winner.json").exists():
        print("Resuming from cmaes_winner.json...")
        initial_params = ControllerParams.from_file("cmaes_winner.json")
        sigma0 = 0.15  # Smaller step size when resuming
    else:
        initial_params = ControllerParams()
        sigma0 = 0.3

    initial_genome = params_to_genome(initial_params)
    print(f"Initial params:\n{initial_params}\n")

    # CMA-ES setup
    opts = {
        "maxiter": args.generations,
        "bounds": [0.0, 1.0],
        "verbose": -1,
        "verb_disp": 0,
    }
    if args.popsize:
        opts["popsize"] = args.popsize

    es = cma.CMAEvolutionStrategy(initial_genome, sigma0, opts)
    print(f"Population size: {es.popsize}")
    print("Starting optimization...\n")

    generation = 0
    best_fitness = float("inf")
    best_genome = None

    with cma.fitness_transformations.EvalParallel2(evaluate_fitness, n_workers) as eval_parallel:
        while not es.stop():
            solutions = es.ask()
            fitnesses = eval_parallel(solutions)

            # Track best
            for sol, fit in zip(solutions, fitnesses):
                if fit < best_fitness:
                    best_fitness = fit
                    best_genome = sol.copy()

            es.tell(solutions, fitnesses)
            generation += 1

            # Progress output
            improved = "★" if best_fitness == min(fitnesses) else " "
            print(
                f"Gen {generation:4d} {improved} | Best: {best_fitness:7.2f} | "
                f"Mean: {np.mean(fitnesses):7.2f} | Min: {np.min(fitnesses):7.2f}"
            )

            # Save best after each generation
            if best_genome is not None:
                genome_to_params(best_genome).to_file("cmaes_winner.json")

    print("\n" + "=" * 70)
    print("Optimization Complete!")
    print("=" * 70)

    # Get best parameters
    best_params = genome_to_params(best_genome if best_genome is not None else es.result.xbest)
    print(f"\nBest weighted fitness: {best_fitness:.2f}")
    print(f"\nBest parameters:\n{best_params}")
    best_params.to_file("cmaes_winner.json")

    # Detailed per-track evaluation
    print("\n" + "-" * 70)
    print("Per-Track Results:")
    print("-" * 70)
    results = evaluate_detailed(best_params, track_list)

    total_time = 0
    total_violations = 0
    core_time = 0
    core_violations = 0

    for name, r in results.items():
        status = "✓" if r["finished"] and r["violations"] == 0 else "⚠" if r["finished"] else "✗"
        weight_str = f" [{r['weight']}x]" if r["weight"] != 1.0 else ""
        print(f"  {status} {name:20s} | Time: {r['time']:6.2f}s | Violations: {r['violations']}{weight_str}")
        total_time += r["time"]
        total_violations += r["violations"]
        if r["weight"] >= 3.0:
            core_time += r["time"]
            core_violations += r["violations"]

    print("-" * 70)
    print(f"  Core tracks (IMS/Montreal/Monza): {core_time:.2f}s | {core_violations} violations")
    print(f"  All tracks total: {total_time:.2f}s | {total_violations} violations")


if __name__ == "__main__":
    main()
