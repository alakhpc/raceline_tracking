"""
NEAT training script for direct racing controller.

Uses NEAT to evolve a recurrent neural network that directly outputs
steering rate and acceleration commands.

Based on best practices from:
https://neat-python.readthedocs.io/en/latest/cookbook.html

Usage:
    python train_neat.py --generations 100
"""

import argparse
import multiprocessing
import os
import pickle
from pathlib import Path

import neat

from racetrack import RaceTrack
from simulator_headless import HeadlessSimulator

# =============================================================================
# CONFIGURATION
# =============================================================================

# Track files to evaluate on
TRACK_FILES = [
    # ("racetracks/IMS.csv", "racetracks/IMS_raceline.csv"),
    ("racetracks/Montreal.csv", "racetracks/Montreal_raceline.csv"),
    # ("racetracks/Monza.csv", "racetracks/Monza_raceline.csv"),
]

# Fitness parameters
VIOLATION_PENALTY = 10.0  # Fitness penalty per track limit violation
DNF_FITNESS = 0.1  # Fitness for not finishing
COMPLEXITY_PENALTY = 0.01  # Penalty per connection/node to prevent bloat


# =============================================================================
# GLOBAL TRACK CACHE (for parallel evaluation)
# =============================================================================

# These will be initialized in main() and accessed by workers
_tracks = None


def init_worker(track_files):
    """Initialize worker process with loaded tracks."""
    global _tracks
    _tracks = [RaceTrack(track_path, raceline_path) for track_path, raceline_path in track_files]


# =============================================================================
# FITNESS EVALUATION
# =============================================================================


def eval_genome(genome, config):
    """
    Evaluate fitness of a single genome.

    IMPORTANT: Returns fitness value (doesn't set genome.fitness).
    This is required for ParallelEvaluator.

    Args:
        genome: NEAT genome to evaluate
        config: NEAT config

    Returns:
        Fitness value (float)
    """
    global _tracks

    # Create recurrent network from genome
    net = neat.nn.RecurrentNetwork.create(genome, config)

    total_fitness = 0.0

    for track in _tracks:
        # Use HeadlessSimulator for consistent simulation
        sim = HeadlessSimulator(track, neat_net=net)
        results = sim.run()

        if results["lap_finished"]:
            # Reward: inverse of lap time (faster = higher fitness)
            # Scale by 1000 to get reasonable fitness values
            fitness = 1000.0 / results["sim_time_elapsed"]
            # Penalty for track violations
            fitness -= results["track_limit_violations"] * VIOLATION_PENALTY
        else:
            # DNF gets minimal fitness
            fitness = DNF_FITNESS

        # Ensure positive fitness (required to avoid species extinction)
        total_fitness += max(0.001, fitness)

    # Average across tracks
    avg_fitness = total_fitness / len(_tracks)

    # Complexity penalty to prevent network bloat
    num_connections = len([c for c in genome.connections.values() if c.enabled])
    num_nodes = len(genome.nodes)
    complexity = num_connections + num_nodes
    avg_fitness -= COMPLEXITY_PENALTY * complexity

    # Ensure final fitness is positive
    return max(0.001, avg_fitness)


# =============================================================================
# MAIN TRAINING LOOP
# =============================================================================


def main():
    parser = argparse.ArgumentParser(description="Train NEAT controller for racing.")
    parser.add_argument(
        "--generations",
        type=int,
        required=True,
        help="Number of generations to run NEAT",
    )
    parser.add_argument(
        "--checkpoint",
        type=str,
        default=None,
        help="Path to checkpoint file to restore from",
    )
    parser.add_argument(
        "--workers",
        type=int,
        default=None,
        help="Number of worker processes (default: CPU count)",
    )
    args = parser.parse_args()

    print("=" * 60)
    print("NEAT Direct Controller Training")
    print("=" * 60)
    print(f"Generations: {args.generations}")
    print(f"Tracks: {[t[0] for t in TRACK_FILES]}")
    print(f"Workers: {args.workers or multiprocessing.cpu_count()}")
    print()

    # Determine config file path (same directory as this script)
    local_dir = os.path.dirname(__file__)
    config_path = os.path.join(local_dir, "neat_config.txt")

    # Load NEAT configuration
    config = neat.Config(
        neat.DefaultGenome,
        neat.DefaultReproduction,
        neat.DefaultSpeciesSet,
        neat.DefaultStagnation,
        config_path,
    )

    # Create or restore population
    if args.checkpoint:
        print(f"Restoring from checkpoint: {args.checkpoint}")
        p = neat.Checkpointer.restore_checkpoint(args.checkpoint)
    else:
        p = neat.Population(config)

    # Add reporters
    p.add_reporter(neat.StdOutReporter(True))
    stats = neat.StatisticsReporter()
    p.add_reporter(stats)

    # Add checkpointer (save every 10 generations)
    checkpointer = neat.Checkpointer(
        generation_interval=10,
        time_interval_seconds=None,
        filename_prefix="neat-checkpoint-",
    )
    p.add_reporter(checkpointer)

    # Run evolution with parallel evaluation
    num_workers = args.workers or multiprocessing.cpu_count()

    print(f"\nStarting evolution with {num_workers} workers...\n")

    # Use context manager for proper cleanup of worker pool
    with neat.ParallelEvaluator(
        num_workers,
        eval_genome,
        initializer=init_worker,
        initargs=(TRACK_FILES,),
    ) as evaluator:
        winner = p.run(evaluator.evaluate, args.generations)

    # Print winner info
    print("\n" + "=" * 60)
    print("Evolution Complete!")
    print("=" * 60)

    print(f"\nBest genome fitness: {winner.fitness:.4f}")
    print(f"Best genome nodes: {len(winner.nodes)}")
    print(f"Best genome connections: {len([c for c in winner.connections.values() if c.enabled])}")

    # Save winner genome
    winner_path = Path("neat_winner.pkl")
    with open(winner_path, "wb") as f:
        pickle.dump(winner, f)
    print(f"\nSaved winner genome to {winner_path}")

    # Run final evaluation to show results
    print("\n" + "-" * 60)
    print("Final Evaluation Results:")
    print("-" * 60)

    # Load tracks for final evaluation
    tracks = [RaceTrack(track_path, raceline_path) for track_path, raceline_path in TRACK_FILES]
    winner_net = neat.nn.RecurrentNetwork.create(winner, config)

    for (track_path, _), track in zip(TRACK_FILES, tracks):
        sim = HeadlessSimulator(track, neat_net=winner_net)
        results = sim.run()
        status = "✓ Finished" if results["lap_finished"] else "✗ DNF"
        print(
            f"  {track_path:30s} | {status} | "
            f"Time: {results['sim_time_elapsed']:6.2f}s | "
            f"Violations: {results['track_limit_violations']}"
        )


if __name__ == "__main__":
    main()
