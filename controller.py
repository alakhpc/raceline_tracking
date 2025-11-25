import numpy as np
from numpy.typing import ArrayLike

from racetrack import RaceTrack

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


def compute_boundary_distances(
    position: ArrayLike,
    closest_idx: int,
    racetrack: RaceTrack,
) -> tuple[float, float]:
    """
    Compute distances from car position to left and right track boundaries.

    Args:
        position: Array of shape (2,) representing [x, y] position
        closest_idx: Index of closest point on centerline
        racetrack: RaceTrack object containing track geometry

    Returns:
        Tuple of (distance_to_left, distance_to_right)
    """
    centerline_point = racetrack.centerline[closest_idx]
    left_boundary_point = racetrack.left_boundary[closest_idx]
    right_boundary_point = racetrack.right_boundary[closest_idx]

    # Vector from centerline to boundaries
    to_left = left_boundary_point - centerline_point
    to_right = right_boundary_point - centerline_point

    # Vector from centerline to car
    to_car = position - centerline_point

    # Project car position onto boundary direction to get signed distance
    left_dist = np.linalg.norm(to_left)
    right_dist = np.linalg.norm(to_right)

    if left_dist > 1e-6:
        # How far along the left direction is the car?
        proj_left = np.dot(to_car, to_left) / left_dist
        dist_to_left = left_dist - proj_left
    else:
        dist_to_left = 0.0

    if right_dist > 1e-6:
        proj_right = np.dot(to_car, to_right) / right_dist
        dist_to_right = right_dist - proj_right
    else:
        dist_to_right = 0.0

    return max(0.0, dist_to_left), max(0.0, dist_to_right)


def compute_heading_error(
    heading: float,
    closest_idx: int,
    raceline: ArrayLike,
) -> float:
    """
    Compute the heading error between car heading and raceline tangent.

    Args:
        heading: Current car heading (rad)
        closest_idx: Index of closest point on raceline
        raceline: Array of shape (N, 2) of raceline points

    Returns:
        Heading error in radians, normalized to [-pi, pi]
    """
    n = len(raceline)
    next_idx = (closest_idx + 1) % n

    # Raceline tangent direction
    tangent = raceline[next_idx] - raceline[closest_idx]
    raceline_heading = np.arctan2(tangent[1], tangent[0])

    # Heading error
    error = raceline_heading - heading

    # Normalize to [-pi, pi]
    error = np.arctan2(np.sin(error), np.cos(error))

    return error


def compute_lateral_error(
    position: ArrayLike,
    heading: float,
    closest_idx: int,
    raceline: ArrayLike,
) -> float:
    """
    Compute signed lateral error from raceline (positive = left of raceline).

    Args:
        position: Array of shape (2,) representing [x, y] position
        heading: Current car heading (rad)
        closest_idx: Index of closest point on raceline
        raceline: Array of shape (N, 2) of raceline points

    Returns:
        Signed lateral error in meters
    """
    raceline_point = raceline[closest_idx]
    to_car = position - raceline_point

    # Get raceline tangent for direction
    n = len(raceline)
    next_idx = (closest_idx + 1) % n
    tangent = raceline[next_idx] - raceline[closest_idx]
    tangent_norm = np.linalg.norm(tangent)

    if tangent_norm < 1e-6:
        return 0.0

    tangent = tangent / tangent_norm

    # Perpendicular (left-pointing) vector
    perp = np.array([-tangent[1], tangent[0]])

    # Signed distance (positive = left of raceline)
    lateral_error = np.dot(to_car, perp)

    return lateral_error


# =============================================================================
# NEAT CONTROLLER
# =============================================================================


def neat_controller(
    state: ArrayLike,
    parameters: ArrayLike,
    racetrack: RaceTrack,
    neat_net,
    closest_idx_hint: int = None,
    return_closest_idx: bool = False,
) -> ArrayLike | tuple[ArrayLike, int]:
    """
    NEAT-based controller - outputs steering_rate and acceleration directly.

    Args:
        state: Array of shape (5,) representing the current vehicle state.
            [0] x position (m)
            [1] y position (m)
            [2] steering angle (rad)
            [3] velocity (m/s)
            [4] heading angle (rad)
        parameters: Array of shape (11,) containing vehicle parameters.
        racetrack: RaceTrack object containing track geometry.
        neat_net: NEAT RecurrentNetwork instance.
        closest_idx_hint: Optional hint for closest point index (for optimization).
        return_closest_idx: If True, returns tuple (controls, closest_idx).

    Returns:
        If return_closest_idx=False: Array of shape (2,) representing control inputs.
            [0] steering rate (rad/s)
            [1] acceleration (m/s²)
        If return_closest_idx=True: Tuple of (controls, closest_idx)
    """
    # Extract state
    x, y = state[0], state[1]
    steering_angle = state[2]
    velocity = state[3]
    heading = state[4]

    position = np.array([x, y])

    # Extract vehicle parameters
    max_steering = parameters[4]
    max_velocity = parameters[5]
    min_steer_rate = parameters[7]
    min_accel = parameters[8]
    max_steer_rate = parameters[9]
    max_accel = parameters[10]

    # Use raceline for path following
    raceline = racetrack.raceline

    # Find closest point on raceline
    closest_idx = find_closest_point(position, raceline, start_idx=closest_idx_hint)

    # =========================================================================
    # COMPUTE 11 NORMALIZED INPUTS
    # =========================================================================

    # 1. Velocity normalized [0, 1]
    velocity_norm = np.clip(velocity / max_velocity, 0.0, 1.0)

    # 2. Steering angle normalized [-1, 1]
    steering_norm = np.clip(steering_angle / max_steering, -1.0, 1.0)

    # 3. Lateral error normalized [-1, 1] (assuming ~10m half-track-width)
    lateral_error = compute_lateral_error(position, heading, closest_idx, raceline)
    half_track_width = 10.0  # Approximate, could be computed from boundaries
    lateral_error_norm = np.clip(lateral_error / half_track_width, -1.0, 1.0)

    # 4. Heading error normalized [-1, 1] (divide by pi)
    heading_error = compute_heading_error(heading, closest_idx, raceline)
    heading_error_norm = heading_error / np.pi

    # 5-7. Curvature at current position and ahead
    n = len(raceline)
    prev_idx = (closest_idx - 1) % n
    next_idx = (closest_idx + 1) % n
    curv_current = compute_curvature(raceline[prev_idx], raceline[closest_idx], raceline[next_idx])
    curv_ahead_50m = get_max_curvature_ahead(raceline, closest_idx, 50.0)
    curv_ahead_100m = get_max_curvature_ahead(raceline, closest_idx, 100.0)

    # Scale curvature to reasonable range [0, 1] (curvature rarely exceeds 0.1)
    curv_scale = 10.0
    curv_current_norm = np.clip(curv_current * curv_scale, 0.0, 1.0)
    curv_ahead_50m_norm = np.clip(curv_ahead_50m * curv_scale, 0.0, 1.0)
    curv_ahead_100m_norm = np.clip(curv_ahead_100m * curv_scale, 0.0, 1.0)

    # 8-9. Boundary distances normalized [0, 1]
    dist_left, dist_right = compute_boundary_distances(position, closest_idx, racetrack)
    dist_left_norm = np.clip(dist_left / half_track_width, 0.0, 2.0) / 2.0
    dist_right_norm = np.clip(dist_right / half_track_width, 0.0, 2.0) / 2.0

    # 10-11. Lookahead point in vehicle frame
    lookahead_dist = 30.0  # Fixed lookahead for NEAT
    lookahead_point = find_lookahead_point(position, raceline, closest_idx, lookahead_dist)

    dx = lookahead_point[0] - x
    dy = lookahead_point[1] - y

    # Transform to vehicle frame
    local_x = dx * np.cos(-heading) - dy * np.sin(-heading)
    local_y = dx * np.sin(-heading) + dy * np.cos(-heading)

    # Normalize by lookahead distance
    lookahead_local_x_norm = np.clip(local_x / lookahead_dist, 0.0, 2.0) / 2.0
    lookahead_local_y_norm = np.clip(local_y / lookahead_dist, -1.0, 1.0)

    # =========================================================================
    # NEAT NETWORK ACTIVATION
    # =========================================================================

    inputs = [
        velocity_norm,
        steering_norm,
        lateral_error_norm,
        heading_error_norm,
        curv_current_norm,
        curv_ahead_50m_norm,
        curv_ahead_100m_norm,
        dist_left_norm,
        dist_right_norm,
        lookahead_local_x_norm,
        lookahead_local_y_norm,
    ]

    # Activate network (returns list of outputs in [-1, 1] due to tanh)
    outputs = neat_net.activate(inputs)

    # Scale outputs to actual control ranges
    # Output 0: steering rate [-max_steer_rate, max_steer_rate]
    # Output 1: acceleration [-max_accel, max_accel]
    steering_rate = outputs[0] * max_steer_rate
    acceleration = outputs[1] * max_accel

    # Clamp to limits
    steering_rate = np.clip(steering_rate, min_steer_rate, max_steer_rate)
    acceleration = np.clip(acceleration, min_accel, max_accel)

    controls = np.array([steering_rate, acceleration])

    if return_closest_idx:
        return controls, closest_idx
    else:
        return controls
