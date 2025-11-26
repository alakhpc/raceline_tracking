import json
from dataclasses import asdict, dataclass, fields
from pathlib import Path
from typing import Optional

import numpy as np
from numpy.typing import ArrayLike

from racetrack import RaceTrack


def get_track_fingerprint(track: RaceTrack) -> str:
    """
    Generate a fingerprint for a track based on its first x,y coordinates.
    Rounded to 3 decimal places for consistent matching.
    """
    x, y = track.centerline[0, 0], track.centerline[0, 1]
    return f"{x:.3f}_{y:.3f}"


def get_fingerprint_from_path(track_path: str) -> str:
    """
    Generate a fingerprint from a track file path without loading the full track.
    """
    with open(track_path) as f:
        for line in f:
            line = line.strip()
            if line and not line.startswith("#"):
                parts = line.split(",")
                x, y = float(parts[0]), float(parts[1])
                return f"{x:.3f}_{y:.3f}"
    raise ValueError(f"No data found in {track_path}")


# =============================================================================
# CONTROLLER PARAMETERS
# =============================================================================


@dataclass
class ControllerParams:
    """
    Controller parameters that can be tuned.
    Defaults are centered in PARAM_BOUNDS for unbiased CMA-ES exploration.
    """

    # Lookahead and speed limits
    lookahead_base: float = 17.5  # Base lookahead distance (m) - range: [5, 30]
    lookahead_gain: float = 0.55  # Velocity scaling for lookahead - range: [0.1, 1.0]
    v_max: float = 95.0  # Maximum target velocity (m/s) - range: [90, 100]
    k_curvature: float = 120.0  # Curvature slowdown factor - range: [40, 200]
    brake_lookahead: float = 215.0  # How far ahead to look for braking (m) - range: [150, 280]
    v_min: float = 15.0  # Minimum velocity to maintain (m/s) - range: [10, 20]
    # Control gains
    kp_steer: float = 4.5  # Proportional gain for steering rate - range: [3, 6]
    kp_vel: float = 4.75  # Proportional gain for acceleration - range: [1.5, 8]
    # Velocity planning parameters
    decel_factor: float = 0.725  # Fraction of max decel to use for braking - range: [0.5, 0.95]
    steer_anticipation: float = 1.9  # How much to slow for steering changes - range: [0.8, 3.0]
    raceline_blend: float = 0.5  # Blend between centerline (0) and raceline (1) - range: [0.2, 0.8]
    straight_lookahead_mult: float = 2.1  # Lookahead multiplier on straights - range: [1.2, 3.0]
    corner_exit_boost: float = 1.5  # Velocity multiplier when exiting corners - range: [1.2, 1.8]

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
            f"  kp_vel={self.kp_vel:.2f},\n"
            f"  decel_factor={self.decel_factor:.2f},\n"
            f"  steer_anticipation={self.steer_anticipation:.2f},\n"
            f"  raceline_blend={self.raceline_blend:.2f},\n"
            f"  straight_lookahead_mult={self.straight_lookahead_mult:.2f},\n"
            f"  corner_exit_boost={self.corner_exit_boost:.2f}\n"
            f")"
        )

    def to_dict(self) -> dict:
        """Convert parameters to a dictionary."""
        return asdict(self)

    def merge_with(self, overrides: dict) -> "ControllerParams":
        """
        Create a new ControllerParams with overrides applied.
        Only overrides keys that exist in this dataclass.
        """
        base_dict = asdict(self)
        valid_keys = {f.name for f in fields(self)}
        for key, value in overrides.items():
            if key in valid_keys:
                base_dict[key] = value
        return ControllerParams(**base_dict)


# Default parameters (used when no ControllerParams is provided)
DEFAULT_PARAMS = ControllerParams()


class ControllerConfig:
    """
    Multi-track configuration with base parameters and per-track overrides.

    Config file format:
    {
        "base": { ...base params... },
        "fingerprint1": { ...override params... },
        "fingerprint2": { ...override params... }
    }

    Fingerprints are generated from the first x,y coordinates of the track.
    """

    def __init__(self, base: ControllerParams = None, overrides: dict[str, dict] = None):
        self.base = base or ControllerParams()
        self.overrides = overrides or {}

    def get_params(self, track: RaceTrack = None, fingerprint: str = None) -> ControllerParams:
        """
        Get parameters for a specific track.
        If fingerprint matches an override, merge those overrides with base.
        Otherwise, return base parameters.
        """
        if track is not None:
            fingerprint = get_track_fingerprint(track)

        if fingerprint and fingerprint in self.overrides:
            return self.base.merge_with(self.overrides[fingerprint])
        return self.base

    def set_override(self, fingerprint: str, params: ControllerParams) -> None:
        """Set override parameters for a specific track fingerprint."""
        # Store only the differences from base
        base_dict = self.base.to_dict()
        params_dict = params.to_dict()
        diff = {k: v for k, v in params_dict.items() if v != base_dict.get(k)}
        if diff:
            self.overrides[fingerprint] = diff
        elif fingerprint in self.overrides:
            del self.overrides[fingerprint]

    def set_full_override(self, fingerprint: str, params: ControllerParams) -> None:
        """Set full override parameters for a specific track fingerprint (all params, not just diff)."""
        self.overrides[fingerprint] = params.to_dict()

    def to_file(self, path: str | Path) -> None:
        """Save configuration to a JSON file."""
        data = {"base": self.base.to_dict()}
        data.update(self.overrides)
        with open(path, "w") as f:
            json.dump(data, f, indent=2)

    @classmethod
    def from_file(cls, path: str | Path) -> "ControllerConfig":
        """Load configuration from a JSON file."""
        with open(path) as f:
            data = json.load(f)

        # Handle legacy format (flat params without "base" key)
        if "base" not in data:
            # Legacy format: treat entire file as base params
            return cls(base=ControllerParams(**data), overrides={})

        base_data = data.pop("base")
        base = ControllerParams(**base_data)
        overrides = data  # Remaining keys are fingerprint overrides
        return cls(base=base, overrides=overrides)


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


def get_weighted_curvature_ahead(
    raceline: ArrayLike, closest_idx: int, lookahead_dist: float, decay: float = 0.7
) -> float:
    """
    Find a weighted curvature metric in the upcoming segment.
    Nearby curvatures are weighted more heavily using exponential decay.

    Args:
        raceline: Array of shape (N, 2) of raceline points
        closest_idx: Current closest index on raceline
        lookahead_dist: How far ahead to look (m)
        decay: Exponential decay factor (0-1). Lower = faster decay.

    Returns:
        Weighted curvature metric (higher = sharper turns ahead)
    """
    n = len(raceline)
    cumulative_dist = 0.0
    idx = closest_idx
    weighted_curv = 0.0
    weight_sum = 0.0
    step = 0

    while cumulative_dist < lookahead_dist:
        prev_idx = (idx - 1) % n
        next_idx = (idx + 1) % n

        curv = compute_curvature(raceline[prev_idx], raceline[idx], raceline[next_idx])

        # Exponential weight: nearby points weighted more heavily
        weight = decay**step
        weighted_curv += curv * weight
        weight_sum += weight

        segment_dist = np.linalg.norm(raceline[next_idx] - raceline[idx])
        cumulative_dist += segment_dist
        idx = next_idx
        step += 1

        if idx == closest_idx:
            break

    if weight_sum > 0:
        return weighted_curv / weight_sum
    return 0.0


def compute_target_velocity_for_curvature(curvature: float, v_max: float, v_min: float, k_curvature: float) -> float:
    """
    Compute target velocity based on curvature.
    Uses physics-based formula: v = sqrt(a_lat / curvature), clamped to limits.

    Args:
        curvature: Local curvature (1/m)
        v_max: Maximum velocity
        v_min: Minimum velocity
        k_curvature: Curvature scaling factor

    Returns:
        Target velocity for this curvature
    """
    if curvature < 1e-6:
        return v_max

    # v = V_MAX / (1 + K * curvature)
    v = v_max / (1.0 + k_curvature * curvature)
    return np.clip(v, v_min, v_max)


def get_braking_velocity(
    current_v: float,
    target_v: float,
    distance: float,
    max_decel: float,
    decel_factor: float = 0.85,
) -> float:
    """
    Compute what velocity we need now to reach target_v at a given distance ahead.
    Uses kinematic equation: v² = v0² + 2*a*d

    Args:
        current_v: Current velocity
        target_v: Target velocity at the point ahead
        distance: Distance to that point
        max_decel: Maximum deceleration available
        decel_factor: Fraction of max decel to use (safety margin)

    Returns:
        Maximum velocity we can be at now to reach target_v
    """
    if distance < 1e-6:
        return target_v

    # Using v² = v0² + 2*a*d, solving for v0:
    # v0² = v² - 2*a*d (where a is negative for braking)
    usable_decel = max_decel * decel_factor
    v_squared = target_v**2 + 2.0 * usable_decel * distance

    if v_squared < 0:
        return target_v

    return np.sqrt(v_squared)


def get_heading_change_ahead(
    raceline: ArrayLike,
    closest_idx: int,
    lookahead_dist: float,
) -> float:
    """
    Compute the total absolute heading change in the upcoming segment.
    This helps identify chicanes and S-curves that require steering agility.

    Args:
        raceline: Array of shape (N, 2) of raceline points
        closest_idx: Current closest index on raceline
        lookahead_dist: How far ahead to look (m)

    Returns:
        Total absolute heading change in radians
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
    decel_factor: float = 0.65,
    steer_anticipation: float = 2.5,
) -> float:
    """
    Compute the required velocity now by looking at upcoming corners
    and computing braking requirements. Also considers steering rate limits.

    Args:
        raceline: Array of shape (N, 2) of raceline points
        closest_idx: Current closest index on raceline
        lookahead_dist: How far ahead to look (m)
        v_max: Maximum velocity
        v_min: Minimum velocity
        k_curvature: Curvature scaling factor
        max_decel: Maximum deceleration
        decel_factor: Fraction of max decel to use
        steer_anticipation: How much to penalize steering changes

    Returns:
        Target velocity for current position considering upcoming corners
    """
    n = len(raceline)
    cumulative_dist = 0.0
    idx = closest_idx
    min_required_v = v_max

    # Check heading changes for chicanes - use shorter lookahead for responsiveness
    heading_change = get_heading_change_ahead(raceline, closest_idx, lookahead_dist * 0.4)

    # If there's significant heading change ahead, reduce velocity
    # This accounts for steering rate limits in chicanes
    if heading_change > 0.3:  # More than ~17 degrees of total change
        # Estimate time needed for steering changes
        # More heading change = need more time = lower velocity
        steering_penalty = 1.0 / (1.0 + steer_anticipation * heading_change)
        min_required_v = min(min_required_v, v_max * steering_penalty)

    while cumulative_dist < lookahead_dist:
        prev_idx = (idx - 1) % n
        next_idx = (idx + 1) % n

        # Get curvature at this point
        curv = compute_curvature(raceline[prev_idx], raceline[idx], raceline[next_idx])

        # Get target velocity for this curvature
        target_v_at_point = compute_target_velocity_for_curvature(curv, v_max, v_min, k_curvature)

        # What velocity do we need NOW to reach that target?
        if cumulative_dist > 0:
            required_v_now = get_braking_velocity(v_max, target_v_at_point, cumulative_dist, max_decel, decel_factor)
        else:
            required_v_now = target_v_at_point

        min_required_v = min(min_required_v, required_v_now)

        # Move to next point
        segment_dist = np.linalg.norm(raceline[next_idx] - raceline[idx])
        cumulative_dist += segment_dist
        idx = next_idx

        if idx == closest_idx:
            break

    return np.clip(min_required_v, v_min, v_max)


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
    High-level controller using Pure Pursuit for steering and physics-based velocity planning.

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
    max_decel = abs(parameters[8])  # Maximum deceleration (positive value)

    # Blend between centerline and raceline based on raceline_blend parameter
    # 0 = pure centerline (conservative), 1 = pure raceline (aggressive)
    blend = ctrl_params.raceline_blend
    raceline = (1 - blend) * racetrack.centerline + blend * racetrack.raceline

    # Find closest point on raceline (use hint for optimization)
    closest_idx = find_closest_point(position, raceline, start_idx=closest_idx_hint)

    # =========================================================================
    # PURE PURSUIT STEERING
    # =========================================================================

    # Get local curvature to detect straights vs corners
    n = len(raceline)
    prev_idx = (closest_idx - 1) % n
    next_idx = (closest_idx + 1) % n
    local_curvature = compute_curvature(raceline[prev_idx], raceline[closest_idx], raceline[next_idx])

    # Compute velocity-dependent lookahead distance
    # Use longer lookahead on straights to reduce oscillation
    base_lookahead = ctrl_params.lookahead_base + ctrl_params.lookahead_gain * abs(current_velocity)

    # On straights (low curvature), multiply lookahead to smooth path following
    # Curvature < 0.005 is roughly a straight (radius > 200m)
    straight_factor = 1.0 / (1.0 + 500.0 * local_curvature)  # Smooth transition
    lookahead_mult = 1.0 + (ctrl_params.straight_lookahead_mult - 1.0) * straight_factor
    lookahead_dist = base_lookahead * lookahead_mult

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

    desired_commands = np.array([desired_steering, desired_velocity])

    if return_closest_idx:
        return desired_commands, closest_idx
    else:
        return desired_commands
