import json
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Optional

import numpy as np
from numpy.typing import ArrayLike

from racetrack import RaceTrack

# =============================================================================
# CONTROLLER PARAMETERS
# =============================================================================


@dataclass
class ControllerParams:
    """
    Controller parameters that can be tuned.
    Default values provide reasonable baseline performance.
    """

    lookahead_base: float = 20.0  # Base lookahead distance (m) - range: [10, 50]
    lookahead_gain: float = 0.8  # Velocity scaling for lookahead - range: [0.5, 2.0]
    v_max: float = 80.0  # Maximum target velocity (m/s) - range: [50, 100]
    k_curvature: float = 200.0  # Curvature slowdown factor - range: [50, 500]
    brake_lookahead: float = 100.0  # How far ahead to look for braking (m) - range: [50, 200]
    v_min: float = 15.0  # Minimum velocity to maintain (m/s) - range: [10, 30]
    kp_steer: float = 2.0  # Proportional gain for steering rate - range: [1, 5]
    kp_vel: float = 5.0  # Proportional gain for acceleration - range: [2, 10]

    def __repr__(self) -> str:
        return (
            f"ControllerParams(\n"
            f"  lookahead_base={self.lookahead_base:.2f},\n"
            f"  lookahead_gain={self.lookahead_gain:.2f},\n"
            f"  v_max={self.v_max:.2f},\n"
            f"  k_curvature={self.k_curvature:.2f},\n"
            f"  brake_lookahead={self.brake_lookahead:.2f},\n"
            f"  v_min={self.v_min:.2f},\n"
            f"  kp_steer={self.kp_steer:.2f},\n"
            f"  kp_vel={self.kp_vel:.2f}\n"
            f")"
        )

    def to_file(self, path: str | Path) -> None:
        """Save controller parameters to a JSON file."""
        with open(path, "w") as f:
            json.dump(asdict(self), f, indent=2)

    @classmethod
    def from_file(cls, path: str | Path) -> "ControllerParams":
        """Load controller parameters from a JSON file."""
        with open(path) as f:
            data = json.load(f)
        return cls(**data)


# Default parameters (used when no ControllerParams is provided)
DEFAULT_PARAMS = ControllerParams()


# =============================================================================
# HELPER FUNCTIONS
# =============================================================================


def find_closest_point(position: ArrayLike, raceline: ArrayLike, start_idx: int = None) -> int:
    """
    Find the index of the closest point on the raceline to the given position.

    Args:
        position: Array of shape (2,) representing [x, y] position
        raceline: Array of shape (N, 2) representing raceline points
        start_idx: Optional starting index hint for optimization (searches nearby points first)

    Returns:
        Index of the closest raceline point
    """
    # Use squared distances to avoid sqrt (faster)
    if start_idx is not None and 0 <= start_idx < len(raceline):
        # Search in a window around start_idx for better performance
        # Check nearby points first (car usually moves forward)
        n = len(raceline)
        window_size = min(50, n // 4)  # Check up to 50 points or 1/4 of track
        start = max(0, start_idx - window_size // 2)
        end = min(n, start_idx + window_size // 2)

        # Check window
        window_distances_sq = np.sum((raceline[start:end] - position) ** 2, axis=1)
        local_min_idx = np.argmin(window_distances_sq)
        local_min_idx += start

        # If local minimum is at window edge, check full array
        if (local_min_idx == start or local_min_idx == end - 1) and n > window_size:
            # Fall back to full search
            distances_sq = np.sum((raceline - position) ** 2, axis=1)
            return np.argmin(distances_sq)
        else:
            return local_min_idx
    else:
        # Full search using squared distances (faster than norm)
        distances_sq = np.sum((raceline - position) ** 2, axis=1)
        return np.argmin(distances_sq)


def compute_curvature(p1: ArrayLike, p2: ArrayLike, p3: ArrayLike) -> float:
    """
    Compute the curvature at point p2 using the Menger curvature formula.
    Curvature = 4 * triangle_area / (|p1-p2| * |p2-p3| * |p3-p1|)

    Args:
        p1, p2, p3: Arrays of shape (2,) representing consecutive points

    Returns:
        Curvature value (1/radius). Higher = tighter turn.
    """
    # Edge lengths
    a = np.linalg.norm(p2 - p1)
    b = np.linalg.norm(p3 - p2)
    c = np.linalg.norm(p3 - p1)

    # Avoid division by zero
    if a < 1e-6 or b < 1e-6 or c < 1e-6:
        return 0.0

    # Triangle area using cross product (2D)
    # Area = 0.5 * |cross product of vectors|
    v1 = p2 - p1
    v2 = p3 - p1
    area = 0.5 * abs(v1[0] * v2[1] - v1[1] * v2[0])

    # Menger curvature
    curvature = 4.0 * area / (a * b * c)
    return curvature


def find_lookahead_point(
    position: ArrayLike, raceline: ArrayLike, closest_idx: int, lookahead_dist: float
) -> ArrayLike:
    """
    Find a point on the raceline at approximately lookahead_dist ahead.

    Args:
        position: Current [x, y] position
        raceline: Array of shape (N, 2) of raceline points
        closest_idx: Index of closest point on raceline
        lookahead_dist: Target distance ahead (m)

    Returns:
        Array of shape (2,) representing the lookahead point [x, y]
    """
    n = len(raceline)
    cumulative_dist = 0.0
    idx = closest_idx

    # Walk along raceline accumulating distance
    while cumulative_dist < lookahead_dist:
        next_idx = (idx + 1) % n
        segment_dist = np.linalg.norm(raceline[next_idx] - raceline[idx])
        cumulative_dist += segment_dist
        idx = next_idx

        # Safety: don't loop forever
        if idx == closest_idx:
            break

    return raceline[idx]


def get_max_curvature_ahead(raceline: ArrayLike, closest_idx: int, lookahead_dist: float) -> float:
    """
    Find the maximum curvature in the upcoming segment of the raceline.

    Args:
        raceline: Array of shape (N, 2) of raceline points
        closest_idx: Current closest index on raceline
        lookahead_dist: How far ahead to look (m)

    Returns:
        Maximum curvature value in the lookahead segment
    """
    n = len(raceline)
    cumulative_dist = 0.0
    idx = closest_idx
    max_curvature = 0.0

    while cumulative_dist < lookahead_dist:
        # Get three consecutive points for curvature
        prev_idx = (idx - 1) % n
        next_idx = (idx + 1) % n

        curv = compute_curvature(raceline[prev_idx], raceline[idx], raceline[next_idx])
        max_curvature = max(max_curvature, curv)

        # Move to next point
        segment_dist = np.linalg.norm(raceline[next_idx] - raceline[idx])
        cumulative_dist += segment_dist
        idx = next_idx

        if idx == closest_idx:
            break

    return max_curvature


# =============================================================================
# CONTROLLERS
# =============================================================================


def lower_controller(
    state: ArrayLike,
    desired: ArrayLike,
    parameters: ArrayLike,
    ctrl_params: Optional[ControllerParams] = None,
) -> ArrayLike:
    """
    Lower-level controller that converts desired commands to control inputs.
    Uses proportional control for both steering rate and acceleration.

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
        ctrl_params: Optional ControllerParams. Uses defaults if None.

    Returns:
        Array of shape (2,) representing control inputs.
            [0] steering rate command (rad/s)
            [1] acceleration command (m/s²)
    """
    assert desired.shape == (2,)

    if ctrl_params is None:
        ctrl_params = DEFAULT_PARAMS

    current_steering = state[2]
    current_velocity = state[3]

    desired_steering = desired[0]
    desired_velocity = desired[1]

    # Proportional control for steering rate
    steering_error = desired_steering - current_steering
    steering_rate = ctrl_params.kp_steer * steering_error

    # Proportional control for acceleration
    velocity_error = desired_velocity - current_velocity
    acceleration = ctrl_params.kp_vel * velocity_error

    # Clamp to parameter limits
    min_steer_rate, min_accel = parameters[7], parameters[8]
    max_steer_rate, max_accel = parameters[9], parameters[10]

    steering_rate = np.clip(steering_rate, min_steer_rate, max_steer_rate)
    acceleration = np.clip(acceleration, min_accel, max_accel)

    return np.array([steering_rate, acceleration])


def controller(
    state: ArrayLike,
    parameters: ArrayLike,
    racetrack: RaceTrack,
    ctrl_params: Optional[ControllerParams] = None,
    closest_idx_hint: int = None,
    return_closest_idx: bool = False,
) -> ArrayLike | tuple[ArrayLike, int]:
    """
    High-level controller using Pure Pursuit for steering and curvature-based velocity.

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
        ctrl_params: Optional ControllerParams. Uses defaults if None.
        closest_idx_hint: Optional hint for closest point index (for optimization).
        return_closest_idx: If True, returns tuple (desired_commands, closest_idx).
                           If False (default), returns only desired_commands for backward compatibility.

    Returns:
        If return_closest_idx=False: Array of shape (2,) representing desired commands.
            [0] desired steering angle (rad)
            [1] desired velocity (m/s)
        If return_closest_idx=True: Tuple of (desired_commands, closest_idx)
    """
    if ctrl_params is None:
        ctrl_params = DEFAULT_PARAMS

    # Extract state
    x, y = state[0], state[1]
    current_velocity = state[3]
    heading = state[4]

    position = np.array([x, y])
    wheelbase = parameters[0]
    max_steering = parameters[4]

    # Use centerline as raceline
    raceline = racetrack.centerline

    # Find closest point on raceline (use hint for optimization)
    closest_idx = find_closest_point(position, raceline, start_idx=closest_idx_hint)

    # =========================================================================
    # PURE PURSUIT STEERING
    # =========================================================================

    # Compute velocity-dependent lookahead distance
    lookahead_dist = ctrl_params.lookahead_base + ctrl_params.lookahead_gain * abs(current_velocity)

    # Find lookahead point
    lookahead_point = find_lookahead_point(position, raceline, closest_idx, lookahead_dist)

    # Vector from car to lookahead point
    dx = lookahead_point[0] - x
    dy = lookahead_point[1] - y

    # Transform to vehicle frame
    # Rotation by -heading to get local coordinates
    local_x = dx * np.cos(-heading) - dy * np.sin(-heading)
    local_y = dx * np.sin(-heading) + dy * np.cos(-heading)

    # Distance to lookahead point
    ld = np.sqrt(local_x**2 + local_y**2)

    if ld < 1e-6:
        # Already at lookahead point, no steering needed
        desired_steering = 0.0
    else:
        # Pure pursuit formula: delta = atan(2 * L * sin(alpha) / ld)
        # where alpha is angle to lookahead point in vehicle frame
        # sin(alpha) = local_y / ld
        desired_steering = np.arctan2(2.0 * wheelbase * local_y, ld**2)

    # Clamp steering angle
    desired_steering = np.clip(desired_steering, -max_steering, max_steering)

    # =========================================================================
    # CURVATURE-BASED VELOCITY
    # =========================================================================

    # Find max curvature in upcoming segment
    max_curv = get_max_curvature_ahead(raceline, closest_idx, ctrl_params.brake_lookahead)

    # Target velocity decreases with curvature
    # v = V_MAX / (1 + K_CURVATURE * curvature)
    desired_velocity = ctrl_params.v_max / (1.0 + ctrl_params.k_curvature * max_curv)

    # Clamp velocity
    desired_velocity = np.clip(desired_velocity, ctrl_params.v_min, ctrl_params.v_max)

    desired_commands = np.array([desired_steering, desired_velocity])

    if return_closest_idx:
        return desired_commands, closest_idx
    else:
        return desired_commands
