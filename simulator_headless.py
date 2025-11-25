from time import time

import numpy as np

from controller import controller, lower_controller
from racecar import RaceCar
from racetrack import RaceTrack


class HeadlessSimulator:
    def __init__(self, rt: RaceTrack):
        """
        Initialize headless simulator.

        Args:
            rt: RaceTrack object containing track geometry.
        """
        self.rt = rt
        self.car = RaceCar(self.rt.initial_state.T)

        self.lap_time_elapsed = 0
        self.lap_start_time = None
        self.sim_time_elapsed = 0
        self.lap_finished = False
        self.lap_started = False
        self.track_limit_violations = 0
        self.currently_violating = False

    def check_track_limits(self):
        """
        Check if car is within track limits.

        Args:
        """
        car_position = self.car.state[0:2]

        # Check only the closest segment (much faster than checking all points)
        centerline_distances = np.linalg.norm(self.rt.centerline - car_position, axis=1)
        closest_idx = np.argmin(centerline_distances)
        to_right = self.rt.right_boundary[closest_idx] - self.rt.centerline[closest_idx]
        to_left = self.rt.left_boundary[closest_idx] - self.rt.centerline[closest_idx]
        to_car = car_position - self.rt.centerline[closest_idx]

        right_dist = np.linalg.norm(to_right)
        left_dist = np.linalg.norm(to_left)

        # Use squared distances to avoid sqrt when possible
        proj_right = np.dot(to_car, to_right) / right_dist if right_dist > 0 else 0
        proj_left = np.dot(to_car, to_left) / left_dist if left_dist > 0 else 0

        is_violating = proj_right > right_dist or proj_left > left_dist

        if is_violating and not self.currently_violating:
            self.track_limit_violations += 1
            self.currently_violating = True
        elif not is_violating:
            self.currently_violating = False

    def update_status(self):
        # Use squared distance to avoid sqrt (10.0^2 = 100.0)
        car_pos = self.car.state[0:2]
        start_pos = self.rt.centerline[0, 0:2]
        progress_squared = np.sum((car_pos - start_pos) ** 2)

        if progress_squared > 100.0 and not self.lap_started:  # 10.0^2 = 100.0
            self.lap_started = True
            self.sim_time_elapsed = 0

        if progress_squared <= 100.0 and self.lap_started and not self.lap_finished:
            self.lap_finished = True
            self.lap_time_elapsed = time() - self.lap_start_time

        if not self.lap_finished and self.lap_start_time is not None:
            self.lap_time_elapsed = time() - self.lap_start_time

    def run_step(self, stop_on_violation: bool = False):
        """
        Run a single simulation step. Returns True if simulation should continue.

        Args:
            stop_on_violation: If True, stop simulation immediately when violation occurs.
        """
        if self.lap_finished:
            return False

        # Check if max simulation time (300 seconds) has been exceeded
        if self.sim_time_elapsed >= 300.0:
            return False

        # Get control output
        desired = controller(self.car.state, self.car.parameters, self.rt)
        cont = lower_controller(self.car.state, desired, self.car.parameters)
        self.car.update(cont)

        if self.lap_started and not self.lap_finished:
            self.sim_time_elapsed += self.car.time_step

        self.update_status()
        self.check_track_limits()

        # Stop immediately if violation occurs and stop_on_violation is True
        if stop_on_violation and self.track_limit_violations > 0:
            return False

        return True

    def run(self, max_iterations: int = None, stop_on_violation: bool = False):
        """
        Run the simulation until lap is completed, max_iterations is reached, max sim time (300s) is exceeded,
        or violation occurs (if stop_on_violation=True).

        Args:
            max_iterations: Maximum number of simulation steps. If None, runs until lap completion or max sim time.
            stop_on_violation: If True, stop simulation immediately when violation occurs.

        Returns:
            Dictionary containing simulation results:
            - lap_finished: bool
            - lap_time_elapsed: float (wall time in seconds)
            - sim_time_elapsed: float (simulation time in seconds)
            - track_limit_violations: int
            - iterations: int (number of steps taken)
        """
        self.lap_start_time = time()
        iterations = 0

        try:
            while True:
                if not self.run_step(stop_on_violation=stop_on_violation):
                    break

                iterations += 1
                if max_iterations is not None and iterations >= max_iterations:
                    break

        except KeyboardInterrupt:
            pass

        return {
            "lap_finished": self.lap_finished,
            "lap_time_elapsed": self.lap_time_elapsed,
            "sim_time_elapsed": self.sim_time_elapsed,
            "track_limit_violations": self.track_limit_violations,
            "iterations": iterations,
        }
