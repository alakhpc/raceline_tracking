"""
Training simulator with checkpoint system for NEAT evolution.
Provides robust progress tracking that prevents circular motion exploits.
"""

from time import time

import numpy as np

from controller import controller, lower_controller
from racecar import RaceCar
from racetrack import RaceTrack


class TrainingSimulator:
    """
    Simulator optimized for NEAT training with checkpoint-based progress tracking.

    Unlike the basic headless simulator, this uses ordered checkpoints to ensure
    the car actually progresses around the track rather than going in circles.
    """

    def __init__(self, rt: RaceTrack, num_checkpoints: int = 20):
        """
        Initialize training simulator.

        Args:
            rt: RaceTrack object containing track geometry.
            num_checkpoints: Number of checkpoints to distribute around the track.
        """
        self.rt = rt
        self.car = RaceCar(self.rt.initial_state.T)
        self.num_checkpoints = num_checkpoints

        # Create evenly-spaced checkpoint indices along raceline
        # Start from checkpoint 1 (not 0) so first checkpoint is AHEAD of start line
        total_points = len(rt.raceline)
        self.checkpoint_indices = [int((i + 1) * total_points / (num_checkpoints + 1)) for i in range(num_checkpoints)]

        # Checkpoint state
        self.next_checkpoint = 0
        self.checkpoints_passed = 0
        self.last_checkpoint_time = 0.0
        self.last_checkpoint_idx = -1  # Track which raceline index triggered last checkpoint

        # Progress tracking
        self._last_closest_idx = 0
        self.wrong_way_count = 0
        self.max_wrong_way = 10  # Threshold for termination

        # Minimum distance (in raceline indices) between checkpoint triggers
        # Prevents triggering multiple checkpoints from same location
        self.min_checkpoint_distance = total_points // (num_checkpoints * 2)

        # Simulation state
        self.lap_time_elapsed = 0.0
        self.lap_start_time = None
        self.sim_time_elapsed = 0.0
        self.lap_finished = False
        self.lap_started = False
        self.track_limit_violations = 0
        self.currently_violating = False

        # Velocity tracking
        self.total_velocity = 0.0
        self.velocity_samples = 0

    def check_track_limits(self) -> bool:
        """
        Check if car is within track limits.

        Returns:
            True if car is on track, False if violating limits.
        """
        car_position = self.car.state[0:2]

        # Find closest centerline point
        _, closest_idx = self.rt.centerline_kdtree.query(car_position)

        to_right = self.rt.right_boundary[closest_idx] - self.rt.centerline[closest_idx]
        to_left = self.rt.left_boundary[closest_idx] - self.rt.centerline[closest_idx]
        to_car = car_position - self.rt.centerline[closest_idx]

        right_dist = np.linalg.norm(to_right)
        left_dist = np.linalg.norm(to_left)

        proj_right = np.dot(to_car, to_right) / right_dist if right_dist > 0 else 0
        proj_left = np.dot(to_car, to_left) / left_dist if left_dist > 0 else 0

        is_violating = proj_right > right_dist or proj_left > left_dist

        if is_violating and not self.currently_violating:
            self.track_limit_violations += 1
            self.currently_violating = True
        elif not is_violating:
            self.currently_violating = False

        return not is_violating

    def update_checkpoints(self) -> None:
        """
        Update checkpoint progress and detect wrong-way driving.

        Checkpoints must be passed:
        1. In order (checkpoint N before N+1)
        2. Going forward (not backward)
        3. With minimum distance traveled since last checkpoint
        """
        car_pos = self.car.state[0:2]
        _, closest_idx = self.rt.raceline_kdtree.query(car_pos)

        n_points = len(self.rt.raceline)

        # Check if we're near the next required checkpoint
        next_cp_idx = self.checkpoint_indices[self.next_checkpoint]

        # Only check forward distance - must CROSS the checkpoint going forward
        # (closest_idx should be at or just past the checkpoint)
        forward_dist = (next_cp_idx - closest_idx) % n_points

        # Tighter tolerance: 2% of track (not 5%)
        checkpoint_tolerance = max(n_points // 50, 10)

        # Check if we just crossed this checkpoint (car is just past it)
        crossed_checkpoint = forward_dist > (n_points - checkpoint_tolerance) or forward_dist == 0

        if crossed_checkpoint:
            # Verify we're going forward, not backward
            if not self._is_moving_backward(closest_idx):
                # Verify we've moved enough since last checkpoint trigger
                # This prevents triggering multiple checkpoints from same spot
                if self.last_checkpoint_idx < 0:
                    distance_since_last = self.min_checkpoint_distance + 1  # First checkpoint always OK
                else:
                    distance_since_last = (closest_idx - self.last_checkpoint_idx) % n_points

                if distance_since_last >= self.min_checkpoint_distance:
                    self.checkpoints_passed += 1
                    self.next_checkpoint = (self.next_checkpoint + 1) % self.num_checkpoints
                    self.last_checkpoint_time = self.sim_time_elapsed
                    self.last_checkpoint_idx = closest_idx

                    # Check for lap completion (passed all checkpoints and back to start)
                    if self.checkpoints_passed >= self.num_checkpoints and self.lap_started:
                        self.lap_finished = True
                        if self.lap_start_time is not None:
                            self.lap_time_elapsed = time() - self.lap_start_time

        # Detect wrong-way driving
        if self._is_moving_backward(closest_idx):
            self.wrong_way_count += 1
        else:
            self.wrong_way_count = max(0, self.wrong_way_count - 1)  # Decay

        self._last_closest_idx = closest_idx

    def _is_moving_backward(self, current_idx: int) -> bool:
        """
        Check if car is facing/moving opposite to track direction.

        Returns:
            True if car is going the wrong way.
        """
        track_heading = self.rt.raceline_headings[current_idx]
        car_heading = self.car.state[4]

        # Heading difference normalized to [-pi, pi]
        heading_diff = np.arctan2(np.sin(car_heading - track_heading), np.cos(car_heading - track_heading))

        # More than 100 degrees off = wrong way
        return abs(heading_diff) > np.pi * 0.55

    def update_lap_status(self) -> None:
        """Update lap timing and started/finished state."""
        car_pos = self.car.state[0:2]
        start_pos = self.rt.centerline[0, 0:2]
        distance_to_start_sq = np.sum((car_pos - start_pos) ** 2)

        # Start lap when car moves away from start line
        if distance_to_start_sq > 100.0 and not self.lap_started:  # 10m threshold
            self.lap_started = True
            self.sim_time_elapsed = 0.0

        # Update elapsed time
        if not self.lap_finished and self.lap_start_time is not None:
            self.lap_time_elapsed = time() - self.lap_start_time

    def run_step(self, stop_on_violation: bool = False) -> bool:
        """
        Run a single simulation step.

        Args:
            stop_on_violation: If True, stop immediately when violation occurs.

        Returns:
            True if simulation should continue, False to stop.
        """
        if self.lap_finished:
            return False

        # Check max simulation time (300 seconds)
        if self.sim_time_elapsed >= 300.0:
            return False

        # Check for persistent wrong-way driving
        if self.wrong_way_count >= self.max_wrong_way:
            return False

        # Get control output from NEAT controller
        desired = controller(self.car.state, self.car.parameters, self.rt)
        control = lower_controller(self.car.state, desired, self.car.parameters)
        self.car.update(control)

        # Update simulation time
        if self.lap_started and not self.lap_finished:
            self.sim_time_elapsed += self.car.time_step

        # Track velocity
        self.total_velocity += self.car.state[3]
        self.velocity_samples += 1

        # Update all status trackers
        self.update_lap_status()
        self.update_checkpoints()
        on_track = self.check_track_limits()

        # Stop immediately if violation occurs and stop_on_violation is True
        if stop_on_violation and not on_track:
            return False

        return True

    def run(
        self,
        max_iterations: int = 3000,
        stop_on_violation: bool = True,
        stuck_threshold: int = 500,
    ) -> dict:
        """
        Run simulation with progress tracking and early termination.

        Args:
            max_iterations: Maximum simulation steps.
            stop_on_violation: Stop immediately on track limit violation.
            stuck_threshold: Stop if no checkpoint reached for this many iterations.

        Returns:
            Dictionary containing simulation results.
        """
        self.lap_start_time = time()
        iterations = 0
        last_checkpoint_iteration = 0

        while True:
            if not self.run_step(stop_on_violation=stop_on_violation):
                break

            iterations += 1

            # Track when checkpoints are passed
            if self.checkpoints_passed > (iterations - last_checkpoint_iteration) // stuck_threshold:
                last_checkpoint_iteration = iterations

            # Early termination if stuck (no checkpoint progress)
            if iterations - last_checkpoint_iteration > stuck_threshold:
                break

            if iterations >= max_iterations:
                break

        avg_velocity = self.total_velocity / max(self.velocity_samples, 1)

        return {
            "lap_finished": self.lap_finished,
            "lap_time_elapsed": self.lap_time_elapsed,
            "sim_time_elapsed": self.sim_time_elapsed,
            "track_limit_violations": self.track_limit_violations,
            "iterations": iterations,
            "checkpoints_passed": self.checkpoints_passed,
            "total_checkpoints": self.num_checkpoints,
            "wrong_way_count": self.wrong_way_count,
            "avg_velocity": avg_velocity,
        }
