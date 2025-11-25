import numpy as np
from numpy.typing import ArrayLike

from racetrack import RaceTrack


def lower_controller(state: ArrayLike, desired: ArrayLike, parameters: ArrayLike) -> ArrayLike:
    """
    Lower-level controller that converts desired steering angle and velocity
    into steering velocity and acceleration commands.

    state: [x, y, steering_angle, velocity, heading]
    desired: [desired_steering_angle, desired_velocity]
    returns: [steering_velocity, acceleration]
    """
    assert desired.shape == (2,)

    desired_steer = desired[0]
    desired_vel = desired[1]

    current_steer = state[2]
    current_vel = state[3]

    # Simple proportional control for steering velocity
    steer_error = desired_steer - current_steer
    steering_velocity = 2.0 * steer_error  # P-gain for steering

    # Simple proportional control for velocity
    vel_error = desired_vel - current_vel
    acceleration = 1.0 * vel_error  # P-gain for acceleration

    return np.array([steering_velocity, acceleration])


def controller(state: ArrayLike, parameters: ArrayLike, racetrack: RaceTrack) -> ArrayLike:
    """
    Simple pure pursuit controller for following the track centerline.

    state: [x, y, steering_angle, velocity, heading]
    returns: [desired_steering_angle, desired_velocity]
    """
    # Extract current position and heading
    pos = state[0:2]
    heading = state[4]
    velocity = state[3]

    # Find closest point on centerline
    distances = np.linalg.norm(racetrack.centerline - pos, axis=1)
    closest_idx = np.argmin(distances)

    # Look ahead - more lookahead at higher speeds
    lookahead_points = max(5, int(velocity / 5))
    target_idx = (closest_idx + lookahead_points) % len(racetrack.centerline)
    target_point = racetrack.centerline[target_idx]

    # Calculate desired heading to target
    dx = target_point[0] - pos[0]
    dy = target_point[1] - pos[1]
    desired_heading = np.arctan2(dy, dx)

    # Calculate heading error (normalized to [-pi, pi])
    heading_error = desired_heading - heading
    heading_error = np.arctan2(np.sin(heading_error), np.cos(heading_error))

    # Convert heading error to steering angle using simple proportional control
    # Wheelbase is parameters[0] = 3.6m
    wheelbase = parameters[0]
    lookahead_dist = np.linalg.norm(target_point - pos)

    # Pure pursuit steering angle formula (simplified)
    if lookahead_dist > 0.1:
        desired_steer = np.arctan2(2.0 * wheelbase * np.sin(heading_error), lookahead_dist)
    else:
        desired_steer = 0.0

    # Clamp steering to max steering angle
    max_steer = 0.9  # from racecar.py
    desired_steer = np.clip(desired_steer, -max_steer, max_steer)

    # Simple speed control - slow down in corners
    # Use a constant moderate speed for simplicity
    desired_velocity = 30.0  # m/s - a moderate speed for testing

    return np.array([desired_steer, desired_velocity])
