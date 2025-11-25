"""
NEAT training script for the racing controller.
Evolves neural networks to drive the racecar around the track.
Optimized with parallel evaluation and early termination.
"""

import multiprocessing
import pickle
from pathlib import Path

import neat

from controller import set_network
from racetrack import RaceTrack
from simulator_headless import HeadlessSimulator

# Training track
TRACK_PATH = "racetracks/Montreal.csv"
RACELINE_PATH = "racetracks/Montreal_raceline.csv"

# Global racetrack (loaded once per process)
_racetrack = None


def get_racetrack():
    """Get or create the racetrack (cached per process)."""
    global _racetrack
    if _racetrack is None:
        _racetrack = RaceTrack(TRACK_PATH, RACELINE_PATH)
    return _racetrack


def evaluate_genome(genome_config_tuple) -> float:
    """
    Evaluate a single genome by running the simulation.
    Designed for parallel execution.

    Returns:
        Fitness score (higher is better).
    """
    genome, config = genome_config_tuple
    racetrack = get_racetrack()

    # Create neural network from genome
    network = neat.nn.FeedForwardNetwork.create(genome, config)
    set_network(network)

    # Run simulation with early termination
    simulator = HeadlessSimulator(racetrack)
    results = run_with_progress_check(simulator, max_iterations=3000)

    # Compute fitness
    sim_time = results["sim_time_elapsed"]
    violations = results["track_limit_violations"]
    lap_finished = results["lap_finished"]
    progress = results.get("progress", 0)
    avg_velocity = results.get("avg_velocity", 0)

    # Primary fitness: progress along track
    fitness = progress * 3.0

    # Strong velocity bonus - encourages going fast
    fitness += avg_velocity * 3.0

    # Big bonus for completing the lap
    if lap_finished:
        fitness += 5000.0
        # Massive bonus for fast lap times (lower sim_time = much better)
        # A lap in 60s gets +1400 bonus, 120s gets +800, 180s gets +200
        fitness += max(0, 200 - sim_time) * 10.0

    # Penalty for violations - but not as harsh now that it's learned
    if violations > 0:
        fitness = fitness * 0.5  # 50% reduction

    # Ensure fitness is non-negative
    return max(0.0, fitness)


def run_with_progress_check(simulator: HeadlessSimulator, max_iterations: int):
    """
    Run simulation with progress tracking and early termination if stuck.
    """
    from time import time

    simulator.lap_start_time = time()
    iterations = 0
    stuck_counter = 0
    max_progress_idx = 0
    total_velocity = 0.0

    racetrack = simulator.rt

    while True:
        if not simulator.run_step(stop_on_violation=True):
            break

        iterations += 1
        total_velocity += simulator.car.state[3]  # Add current velocity

        # Check progress every 100 iterations
        if iterations % 100 == 0:
            car_pos = simulator.car.state[0:2]
            _, closest_idx = racetrack.raceline_kdtree.query(car_pos)

            # Track maximum progress (handle lap wraparound)
            if closest_idx > max_progress_idx or (
                max_progress_idx > len(racetrack.raceline) * 0.9 and closest_idx < len(racetrack.raceline) * 0.1
            ):
                max_progress_idx = closest_idx
                stuck_counter = 0
            else:
                stuck_counter += 1

            # Early termination if stuck for too long (300 iterations = 30 seconds)
            if stuck_counter > 3:
                break

        if iterations >= max_iterations:
            break

    avg_velocity = total_velocity / max(iterations, 1)

    return {
        "lap_finished": simulator.lap_finished,
        "lap_time_elapsed": simulator.lap_time_elapsed,
        "sim_time_elapsed": simulator.sim_time_elapsed,
        "track_limit_violations": simulator.track_limit_violations,
        "iterations": iterations,
        "progress": max_progress_idx,
        "avg_velocity": avg_velocity,
    }


def evaluate_genomes_parallel(genomes, config):
    """
    Evaluate all genomes in parallel using multiprocessing.
    """
    # Prepare genome-config pairs for parallel evaluation
    genome_config_pairs = [(genome, config) for _, genome in genomes]

    # Use all available CPU cores
    num_workers = multiprocessing.cpu_count()

    with multiprocessing.Pool(num_workers) as pool:
        fitnesses = pool.map(evaluate_genome, genome_config_pairs)

    # Assign fitnesses back to genomes
    for (genome_id, genome), fitness in zip(genomes, fitnesses):
        genome.fitness = fitness


def evaluate_genomes_serial(genomes, config):
    """
    Evaluate all genomes serially (for debugging or single-core systems).
    """
    for genome_id, genome in genomes:
        genome.fitness = evaluate_genome((genome, config))


def run_neat(config_path: str, generations: int = 100, parallel: bool = True):
    """
    Run NEAT evolution.

    Args:
        config_path: Path to NEAT configuration file.
        generations: Number of generations to run.
        parallel: Whether to use parallel evaluation.
    """
    # Load configuration
    config = neat.Config(
        neat.DefaultGenome,
        neat.DefaultReproduction,
        neat.DefaultSpeciesSet,
        neat.DefaultStagnation,
        config_path,
    )

    # Create population
    population = neat.Population(config)

    # Add reporters for progress
    population.add_reporter(neat.StdOutReporter(True))
    stats = neat.StatisticsReporter()
    population.add_reporter(stats)

    # Choose evaluation function
    eval_func = evaluate_genomes_parallel if parallel else evaluate_genomes_serial

    # Run evolution
    winner = population.run(eval_func, generations)

    # Save winner
    print(f"\nBest genome fitness: {winner.fitness}")

    # Save the winner genome
    with open("neat_winner.pkl", "wb") as f:
        winner_network = neat.nn.FeedForwardNetwork.create(winner, config)
        pickle.dump(winner_network, f)

    print("Winner saved to neat_winner.pkl")

    return winner


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Train NEAT racing controller")
    parser.add_argument("--generations", "-g", type=int, default=100, help="Number of generations to run")
    parser.add_argument("--config", "-c", type=str, default="neat_config.txt", help="Path to NEAT config file")
    parser.add_argument("--serial", "-s", action="store_true", help="Use serial evaluation (no multiprocessing)")
    args = parser.parse_args()

    # Check if config exists
    config_path = Path(args.config)
    if not config_path.exists():
        print(f"Error: Config file not found: {config_path}")
        exit(1)

    print(f"Using {'serial' if args.serial else 'parallel'} evaluation")
    run_neat(str(config_path), args.generations, parallel=not args.serial)
