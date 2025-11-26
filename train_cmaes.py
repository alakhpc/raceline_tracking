"""
CMA-ES training script for controller parameter optimization.

Uses Covariance Matrix Adaptation Evolution Strategy to tune the 10 controller
parameters for optimal lap time with minimal track violations.

Supports parallel evaluation with --workers flag.
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

# Parameter ranges from ControllerParams docstrings
PARAM_BOUNDS = {
    "lookahead_base": (5.0, 40.0),
    "lookahead_gain": (0.2, 1.5),
    "v_max": (70.0, 100.0),
    "k_curvature": (80.0, 400.0),
    "brake_lookahead": (80.0, 250.0),
    "v_min": (10.0, 25.0),
    "kp_steer": (1.0, 6.0),
    "kp_vel": (2.0, 12.0),
    "decel_factor": (0.4, 0.9),
    "steer_anticipation": (1.0, 5.0),
    "raceline_blend": (0.0, 1.0),  # 0 = centerline, 1 = raceline
    "straight_lookahead_mult": (1.5, 4.0),  # Lookahead multiplier on straights
    "corner_exit_boost": (1.0, 1.5),  # Velocity boost when exiting corners
}

PARAM_NAMES = list(PARAM_BOUNDS.keys())

# Track files to evaluate on (use all available tracks for robust optimization)
TRACK_FILES = [
    "racetracks/IMS.csv",
    "racetracks/Montreal.csv",
    "racetracks/Monza.csv",
]


# =============================================================================
# GENOME <-> PARAMS CONVERSION
# =============================================================================


def genome_to_params(genome: np.ndarray) -> ControllerParams:
    """
    Convert a normalized genome [0, 1]^10 to ControllerParams.
    """
    params = {}
    for i, name in enumerate(PARAM_NAMES):
        low, high = PARAM_BOUNDS[name]
        # Clamp to [0, 1] then scale to actual range
        val = np.clip(genome[i], 0.0, 1.0)
        params[name] = low + val * (high - low)
    return ControllerParams(**params)


def params_to_genome(params: ControllerParams) -> np.ndarray:
    """
    Convert ControllerParams to a normalized genome [0, 1]^10.
    """
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


def evaluate_fitness(genome: np.ndarray) -> float:
    """
    Evaluate fitness of a genome by running simulation on all tracks.

    Lower fitness is better (CMA-ES minimizes).

    Fitness components:
    - Simulation time (primary objective - minimize lap time)
    - Track violations penalty
    - DNF penalty (if lap not finished)

    Note: Tracks are loaded inside each call to support multiprocessing.
    """
    ctrl_params = genome_to_params(genome)

    # Load tracks in each worker process
    tracks = [RaceTrack(path) for path in TRACK_FILES]

    total_fitness = 0.0

    for track in tracks:
        sim = HeadlessSimulator(track, ctrl_params=ctrl_params)
        results = sim.run(stop_on_violation=False)

        if results["lap_finished"]:
            # Completed lap: fitness = sim time + violation penalty
            fitness = results["sim_time_elapsed"]
            fitness += results["track_limit_violations"] * 5.0  # 5 second penalty per violation
        else:
            # DNF: heavy penalty based on how much time elapsed
            # Cap at 500 to avoid extreme values
            fitness = 500.0 + results["track_limit_violations"] * 10.0

        total_fitness += fitness

    # Average across tracks
    return total_fitness / len(tracks)


# =============================================================================
# MAIN TRAINING LOOP
# =============================================================================


def main():
    parser = argparse.ArgumentParser(description="Tune controller parameters using CMA-ES optimization.")
    parser.add_argument(
        "--generations",
        type=int,
        required=True,
        help="Number of generations to run CMA-ES",
    )
    parser.add_argument(
        "--workers",
        type=int,
        default=0,
        help="Number of parallel workers (0 = auto-detect CPU count, 1 = sequential)",
    )
    parser.add_argument(
        "--popsize",
        type=int,
        default=None,
        help="Population size (default: auto based on dimension)",
    )
    args = parser.parse_args()

    # Determine number of workers
    if args.workers == 0:
        n_workers = os.cpu_count() or 1
    else:
        n_workers = args.workers

    print("=" * 60)
    print("CMA-ES Controller Parameter Optimization")
    print("=" * 60)
    print(f"Generations: {args.generations}")
    print(f"Workers: {n_workers}")
    print(f"Parameters: {PARAM_NAMES}")
    print(f"Tracks: {TRACK_FILES}")
    print()

    # Start from default params (normalized to [0, 1])
    default_params = ControllerParams()
    initial_genome = params_to_genome(default_params)

    print(f"Initial params:\n{default_params}\n")

    # CMA-ES setup
    # sigma0: initial step size (0.3 works well for [0,1] normalized space)
    sigma0 = 0.3

    # Bounds for CMA-ES (all params normalized to [0, 1])
    opts = {
        "maxiter": args.generations,
        "bounds": [0.0, 1.0],  # All dimensions bounded to [0, 1]
        "verbose": -1,  # Suppress CMA-ES internal output
        "verb_disp": 0,
    }

    if args.popsize is not None:
        opts["popsize"] = args.popsize

    es = cma.CMAEvolutionStrategy(initial_genome, sigma0, opts)

    print(f"Population size: {es.popsize}")
    print("Starting CMA-ES optimization...\n")

    generation = 0
    best_fitness = float("inf")
    best_genome = None

    # Use EvalParallel2 for parallel fitness evaluation
    with cma.fitness_transformations.EvalParallel2(evaluate_fitness, n_workers) as eval_parallel:
        while not es.stop():
            # Get candidate solutions
            solutions = es.ask()

            # Evaluate fitness in parallel
            fitnesses = eval_parallel(solutions)

            # Track best
            for sol, fit in zip(solutions, fitnesses):
                if fit < best_fitness:
                    best_fitness = fit
                    best_genome = sol.copy()

            # Update CMA-ES
            es.tell(solutions, fitnesses)

            generation += 1
            print(
                f"Gen {generation:4d} | Best fitness: {best_fitness:8.2f} | "
                f"Pop mean: {np.mean(fitnesses):8.2f} | Pop min: {np.min(fitnesses):8.2f}"
            )

            # Save best params after each generation
            if best_genome is not None:
                best_params_so_far = genome_to_params(best_genome)
                output_path = Path("cmaes_winner.json")
                best_params_so_far.to_file(output_path)

    print("\n" + "=" * 60)
    print("Optimization Complete!")
    print("=" * 60)

    # Get final best solution
    final_best = es.result.xbest
    final_fitness = es.result.fbest

    # Use the tracked best if it's better (can happen due to elitism differences)
    if best_fitness < final_fitness:
        final_best = best_genome
        final_fitness = best_fitness

    best_params = genome_to_params(final_best)

    print(f"\nBest fitness: {final_fitness:.2f}")
    print(f"\nBest parameters:\n{best_params}")

    # Save to JSON
    output_path = Path("cmaes_winner.json")
    best_params.to_file(output_path)
    print(f"\nSaved best parameters to {output_path}")

    # Run final evaluation to show results
    print("\n" + "-" * 60)
    print("Final Evaluation Results:")
    print("-" * 60)
    tracks = [RaceTrack(path) for path in TRACK_FILES]
    for track_path, track in zip(TRACK_FILES, tracks):
        sim = HeadlessSimulator(track, ctrl_params=best_params)
        results = sim.run()
        status = "✓ Finished" if results["lap_finished"] else "✗ DNF"
        print(
            f"  {track_path:30s} | {status} | "
            f"Time: {results['sim_time_elapsed']:6.2f}s | "
            f"Violations: {results['track_limit_violations']}"
        )


if __name__ == "__main__":
    main()
