import numpy as np
from numpy.typing import ArrayLike

from config import get_config
from racetrack import RaceTrack

# =============================================================================
# CONTROLLERS
# =============================================================================


def lower_controller(state: ArrayLike, desired: ArrayLike, parameters: ArrayLike) -> ArrayLike:
    """
    Lower-level controller that converts desired commands to control inputs.
    Uses proportional control for both steering rate and acceleration.
    """
    assert desired.shape == (3,)
    ctrl_params = desired[2]

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


def controller(state: ArrayLike, parameters: ArrayLike, racetrack: RaceTrack) -> ArrayLike:
    """
    High-level controller using Pure Pursuit for steering and physics-based velocity planning.
    """
    # Get optimal parameters based on track heuristics
    ctrl_params = get_config().get_params(racetrack)

    # Extract state
    x, y = state[0], state[1]
    current_velocity = state[3]
    heading = state[4]

    position = np.array([x, y])
    wheelbase = parameters[0]
    max_steering = parameters[4]
    max_decel = abs(parameters[8])  # Maximum deceleration (positive value)

    # Blend centerline and raceline (0 = conservative centerline, 1 = aggressive raceline)
    blend = ctrl_params.raceline_blend
    n_centerline = len(racetrack.centerline)
    # Resample raceline to match centerline length
    if len(racetrack.raceline) != n_centerline:
        indices = np.linspace(0, len(racetrack.raceline) - 1, n_centerline)
        resampled = np.column_stack(
            (
                np.interp(
                    indices,
                    np.arange(len(racetrack.raceline)),
                    racetrack.raceline[:, 0],
                ),
                np.interp(
                    indices,
                    np.arange(len(racetrack.raceline)),
                    racetrack.raceline[:, 1],
                ),
            )
        )
    else:
        resampled = racetrack.raceline
    raceline = (1 - blend) * racetrack.centerline + blend * resampled
    n = len(raceline)

    # Find closest point on raceline
    closest_idx = np.argmin(np.sum((raceline - position) ** 2, axis=1))

    # Get local curvature to detect straights vs corners
    prev_idx = (closest_idx - 1) % n
    next_idx = (closest_idx + 1) % n
    local_curvature = compute_curvature(raceline[prev_idx], raceline[closest_idx], raceline[next_idx])

    # Velocity-dependent lookahead, longer on straights to reduce oscillation
    base_lookahead = ctrl_params.lookahead_base + ctrl_params.lookahead_gain * abs(current_velocity)
    straight_factor = 1.0 / (1.0 + 500.0 * local_curvature)
    lookahead_dist = base_lookahead * (1.0 + (ctrl_params.straight_lookahead_mult - 1.0) * straight_factor)

    # Find lookahead point by walking along raceline
    cumulative_dist, idx = 0.0, closest_idx
    while cumulative_dist < lookahead_dist:
        nxt = (idx + 1) % n
        cumulative_dist += np.linalg.norm(raceline[nxt] - raceline[idx])
        idx = nxt
        if idx == closest_idx:
            break
    lookahead_point = raceline[idx]

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
    # PHYSICS-BASED VELOCITY PLANNING
    # =========================================================================

    # Use the velocity profile that considers braking distances and steering limits
    desired_velocity = get_velocity_profile_ahead(
        raceline,
        closest_idx,
        ctrl_params.brake_lookahead,
        ctrl_params.v_max,
        ctrl_params.v_min,
        ctrl_params.k_curvature,
        max_decel,
        ctrl_params.decel_factor,
        ctrl_params.steer_anticipation,
    )

    # Corner exit boost: if curvature is decreasing, we're exiting a corner
    # Look at curvature a bit ahead to detect exit
    ahead_idx = (closest_idx + 5) % n
    ahead_prev = (ahead_idx - 1) % n
    ahead_next = (ahead_idx + 1) % n
    ahead_curvature = compute_curvature(raceline[ahead_prev], raceline[ahead_idx], raceline[ahead_next])

    # If ahead curvature is less than current (exiting corner), boost velocity
    if ahead_curvature < local_curvature * 0.8:  # Curvature decreasing by 20%+
        # Apply corner exit boost (accelerate earlier out of corners)
        exit_boost = 1.0 + (ctrl_params.corner_exit_boost - 1.0) * (1.0 - ahead_curvature / (local_curvature + 1e-6))
        exit_boost = min(exit_boost, ctrl_params.corner_exit_boost)  # Cap at max boost
        desired_velocity = min(desired_velocity * exit_boost, ctrl_params.v_max)

    desired_commands = np.array([desired_steering, desired_velocity, ctrl_params])

    return desired_commands


# =============================================================================
# HELPER FUNCTIONS
# =============================================================================


def compute_curvature(p1: ArrayLike, p2: ArrayLike, p3: ArrayLike) -> float:
    """
    Compute the curvature at point p2 using the Menger curvature formula.
    Curvature = 4 * triangle_area / (|p1-p2| * |p2-p3| * |p3-p1|)
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


def get_heading_change_ahead(raceline: ArrayLike, closest_idx: int, lookahead_dist: float) -> float:
    """
    Compute the total absolute heading change in the upcoming segment.
    This helps identify chicanes and S-curves that require steering agility.
    """
    n = len(raceline)
    cumulative_dist = 0.0
    idx = closest_idx
    total_heading_change = 0.0

    # Get initial heading
    next_idx = (idx + 1) % n
    prev_heading = np.arctan2(
        raceline[next_idx][1] - raceline[idx][1],
        raceline[next_idx][0] - raceline[idx][0],
    )

    while cumulative_dist < lookahead_dist:
        next_idx = (idx + 1) % n
        next_next_idx = (next_idx + 1) % n

        # Compute heading at next point
        curr_heading = np.arctan2(
            raceline[next_next_idx][1] - raceline[next_idx][1],
            raceline[next_next_idx][0] - raceline[next_idx][0],
        )

        # Compute heading change (handle wraparound)
        heading_diff = curr_heading - prev_heading
        heading_diff = np.arctan2(np.sin(heading_diff), np.cos(heading_diff))
        total_heading_change += abs(heading_diff)

        prev_heading = curr_heading

        # Move to next point
        segment_dist = np.linalg.norm(raceline[next_idx] - raceline[idx])
        cumulative_dist += segment_dist
        idx = next_idx

        if idx == closest_idx:
            break

    return total_heading_change


def get_velocity_profile_ahead(
    raceline: ArrayLike,
    closest_idx: int,
    lookahead_dist: float,
    v_max: float,
    v_min: float,
    k_curvature: float,
    max_decel: float,
    decel_factor: float,
    steer_anticipation: float,
) -> float:
    """
    Compute target velocity by looking at upcoming corners and braking requirements.
    """
    n = len(raceline)
    cumulative_dist = 0.0
    idx = closest_idx
    min_required_v = v_max
    usable_decel = max_decel * decel_factor

    # Check heading changes for chicanes (steering rate limits)
    heading_change = get_heading_change_ahead(raceline, closest_idx, lookahead_dist * 0.4)
    if heading_change > 0.3:  # More than ~17 degrees
        steering_penalty = 1.0 / (1.0 + steer_anticipation * heading_change)
        min_required_v = min(min_required_v, v_max * steering_penalty)

    while cumulative_dist < lookahead_dist:
        prev_idx = (idx - 1) % n
        next_idx = (idx + 1) % n

        curv = compute_curvature(raceline[prev_idx], raceline[idx], raceline[next_idx])

        # Target velocity for curvature: v = v_max / (1 + k * curvature)
        target_v = v_max if curv < 1e-6 else np.clip(v_max / (1.0 + k_curvature * curv), v_min, v_max)

        # Braking velocity: what speed now to reach target_v? (kinematic: v² = v0² + 2ad)
        if cumulative_dist > 0:
            v_sq = target_v**2 + 2.0 * usable_decel * cumulative_dist
            required_v = target_v if v_sq < 0 else np.sqrt(v_sq)
        else:
            required_v = target_v

        min_required_v = min(min_required_v, required_v)

        segment_dist = np.linalg.norm(raceline[next_idx] - raceline[idx])
        cumulative_dist += segment_dist
        idx = next_idx

        if idx == closest_idx:
            break

    return np.clip(min_required_v, v_min, v_max)
