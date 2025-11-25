"""
CMA-ES training script for controller parameter optimization.

Uses Covariance Matrix Adaptation Evolution Strategy to tune the 8 controller
parameters for optimal lap time with minimal track violations.
"""

import argparse
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
    "lookahead_base": (10.0, 50.0),
    "lookahead_gain": (0.5, 2.0),
    "v_max": (50.0, 100.0),
    "k_curvature": (50.0, 500.0),
    "brake_lookahead": (50.0, 200.0),
    "v_min": (10.0, 30.0),
    "kp_steer": (1.0, 5.0),
    "kp_vel": (2.0, 10.0),
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
    Convert a normalized genome [0, 1]^8 to ControllerParams.
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
    Convert ControllerParams to a normalized genome [0, 1]^8.
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


def evaluate_fitness(genome: np.ndarray, tracks: list[RaceTrack]) -> float:
    """
    Evaluate fitness of a genome by running simulation on all tracks.

    Lower fitness is better (CMA-ES minimizes).

    Fitness components:
    - Simulation time (primary objective - minimize lap time)
    - Track violations penalty
    - DNF penalty (if lap not finished)
    """
    ctrl_params = genome_to_params(genome)

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
    args = parser.parse_args()

    print("=" * 60)
    print("CMA-ES Controller Parameter Optimization")
    print("=" * 60)
    print(f"Generations: {args.generations}")
    print(f"Parameters: {PARAM_NAMES}")
    print(f"Tracks: {TRACK_FILES}")
    print()

    # Load tracks once
    print("Loading tracks...")
    tracks = [RaceTrack(path) for path in TRACK_FILES]
    print(f"Loaded {len(tracks)} tracks.\n")

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
        "verbose": 1,
        "verb_disp": 1,
    }

    es = cma.CMAEvolutionStrategy(initial_genome, sigma0, opts)

    print("Starting CMA-ES optimization...\n")

    generation = 0
    best_fitness = float("inf")
    best_genome = None

    while not es.stop():
        # Get candidate solutions
        solutions = es.ask()

        # Evaluate fitness for each solution
        fitnesses = []
        for sol in solutions:
            fitness = evaluate_fitness(sol, tracks)
            fitnesses.append(fitness)

            # Track best
            if fitness < best_fitness:
                best_fitness = fitness
                best_genome = sol.copy()

        # Update CMA-ES
        es.tell(solutions, fitnesses)

        generation += 1
        print(
            f"Gen {generation:4d} | Best fitness: {best_fitness:8.2f} | "
            f"Pop mean: {np.mean(fitnesses):8.2f} | Pop min: {np.min(fitnesses):8.2f}"
        )

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
