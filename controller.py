import pickle
from pathlib import Path

import numpy as np
from numpy.typing import ArrayLike

from racetrack import RaceTrack

# Global network instance (loaded from trained NEAT genome)
_network = None


def load_network(path: str | Path) -> None:
    """
    Load a trained NEAT network from a pickle file.

    Args:
        path: Path to the pickle file containing the network.
    """
    global _network
    with open(path, "rb") as f:
        _network = pickle.load(f)


def get_network():
    """Get the current loaded network (used by training script)."""
    return _network


def set_network(network) -> None:
    """Set the network directly (used by training script)."""
    global _network
    _network = network


def compute_inputs(state: ArrayLike, racetrack: RaceTrack) -> ArrayLike:
    """
    Compute the 7 neural network inputs from state and racetrack.
    Optimized version using KD-Trees and precomputed track data.

    Args:
        state: Array of shape (5,) representing the current vehicle state.
            [0] x position (m)
            [1] y position (m)
            [2] steering angle (rad)
            [3] velocity (m/s)
            [4] heading angle (rad)
        racetrack: RaceTrack object containing track geometry.

    Returns:
        Array of shape (7,) containing normalized inputs:
            [0] cross-track error (lateral distance from raceline, normalized)
            [1] heading error (angle from raceline direction, normalized to [-1, 1])
            [2] current velocity (normalized)
            [3] current steering angle (normalized)
            [4] upcoming curvature (normalized)
            [5] distance to left boundary (normalized)
            [6] distance to right boundary (normalized)
    """
    car_pos = state[0:2]
    car_heading = state[4]
    car_velocity = state[3]
    car_steering = state[2]

    # Find closest point on raceline using KD-Tree (O(log n) instead of O(n))
    _, closest_idx = racetrack.raceline_kdtree.query(car_pos)

    # Cross-track error using precomputed direction
    closest_point = racetrack.raceline[closest_idx]
    to_car = car_pos - closest_point
    raceline_dir = racetrack.raceline_directions[closest_idx]

    # Cross-track error (positive = right of raceline, negative = left)
    cross_track = np.cross(raceline_dir, to_car)
    cross_track_normalized = np.clip(cross_track / 10.0, -1.0, 1.0)

    # Heading error using precomputed heading
    heading_error = car_heading - racetrack.raceline_headings[closest_idx]
    heading_error = np.arctan2(np.sin(heading_error), np.cos(heading_error))
    heading_error_normalized = heading_error / np.pi

    # Velocity and steering (normalized)
    velocity_normalized = car_velocity / 100.0
    steering_normalized = car_steering / 0.9

    # Curvature from precomputed values
    curvature_normalized = racetrack.raceline_curvatures[closest_idx] / np.pi

    # Distance to boundaries using KD-Tree
    _, center_closest_idx = racetrack.centerline_kdtree.query(car_pos)

    left_boundary_point = racetrack.left_boundary[center_closest_idx]
    right_boundary_point = racetrack.right_boundary[center_closest_idx]

    dist_left = np.linalg.norm(car_pos - left_boundary_point)
    dist_right = np.linalg.norm(car_pos - right_boundary_point)

    dist_left_normalized = np.clip(dist_left / 15.0, 0.0, 1.0)
    dist_right_normalized = np.clip(dist_right / 15.0, 0.0, 1.0)

    return np.array(
        [
            cross_track_normalized,
            heading_error_normalized,
            velocity_normalized,
            steering_normalized,
            curvature_normalized,
            dist_left_normalized,
            dist_right_normalized,
        ]
    )


def lower_controller(state: ArrayLike, desired: ArrayLike, parameters: ArrayLike) -> ArrayLike:
    """
    Lower-level controller that converts desired commands to control inputs.
    Simply passes through the desired values as control commands.

    Args:
        state: Array of shape (5,) representing the current vehicle state.
            [0] x position (m)
            [1] y position (m)
            [2] steering angle (rad)
            [3] velocity (m/s)
            [4] heading angle (rad)
        desired: Array of shape (2,) representing desired commands.
            [0] desired steering angle (rad)
            [1] desired velocity (m/s)
        parameters: Array of shape (11,) containing vehicle parameters.

    Returns:
        Array of shape (2,) representing control inputs.
            [0] steering rate command (rad/s)
            [1] acceleration command (m/s²)
    """
    assert desired.shape == (2,)

    # Simple proportional control to reach desired values
    current_steering = state[2]
    current_velocity = state[3]

    # Steering rate to reach desired steering angle
    steering_error = desired[0] - current_steering
    steering_rate = np.clip(steering_error * 2.0, -0.4, 0.4)  # Max steering rate

    # Acceleration to reach desired velocity
    velocity_error = desired[1] - current_velocity
    acceleration = np.clip(velocity_error * 2.0, -20.0, 20.0)  # Max acceleration

    return np.array([steering_rate, acceleration])


def controller(state: ArrayLike, parameters: ArrayLike, racetrack: RaceTrack) -> ArrayLike:
    """
    High-level controller that returns desired commands using NEAT network.

    Args:
        state: Array of shape (5,) representing the current vehicle state.
            [0] x position (m)
            [1] y position (m)
            [2] steering angle (rad)
            [3] velocity (m/s)
            [4] heading angle (rad)
        parameters: Array of shape (11,) containing vehicle parameters.
            [0] wheelbase (m)
        racetrack: RaceTrack object containing track geometry.

    Returns:
        Array of shape (2,) representing desired commands.
            [0] desired steering angle (rad)
            [1] desired velocity (m/s)
    """
    if _network is None:
        # Fallback if no network loaded - go straight slowly
        return np.array([0.0, 10.0])

    # Compute inputs for neural network
    inputs = compute_inputs(state, racetrack)

    # Run neural network
    outputs = _network.activate(inputs)

    # Convert outputs to desired steering and velocity
    # Network outputs are in range [-1, 1] due to tanh activation
    desired_steering = outputs[0] * 0.6  # Moderate steering authority
    # Scale velocity to [20, 90] m/s
    desired_velocity = 20.0 + (outputs[1] + 1.0) * 35.0  # [20, 90] m/s

    return np.array([desired_steering, desired_velocity])
