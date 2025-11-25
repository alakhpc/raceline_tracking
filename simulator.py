from abc import ABC, abstractmethod
from time import time

import matplotlib
import matplotlib.pyplot as plt
import numpy as np

from controller import controller, lower_controller
from racecar import RaceCar
from racetrack import RaceTrack
from report import LapReport


class SimulatorBase(ABC):
    """Base class for race simulators with shared functionality."""

    def __init__(self, rt: RaceTrack, max_time: float = 300.0):
        self.rt = rt
        self.car = RaceCar(self.rt.initial_state.T)
        self.max_time = max_time

        self.sim_time = 0.0
        self.wall_start_time: float | None = None
        self.lap_finished = False
        self.lap_started = False
        self.track_limit_violations = 0
        self.currently_violating = False
        self.steps = 0

    def check_track_limits(self, car_x: float, car_y: float) -> bool:
        """Check if car is within track boundaries. Returns True if violating."""
        centerline = self.rt.centerline
        right_boundary = self.rt.right_boundary
        left_boundary = self.rt.left_boundary

        # Find closest centerline point
        dx_arr = centerline[:, 0] - car_x
        dy_arr = centerline[:, 1] - car_y
        dist_sq = dx_arr * dx_arr + dy_arr * dy_arr
        closest_idx = np.argmin(dist_sq)

        cl = centerline[closest_idx]
        rb = right_boundary[closest_idx]
        lb = left_boundary[closest_idx]

        to_right_x, to_right_y = rb[0] - cl[0], rb[1] - cl[1]
        to_left_x, to_left_y = lb[0] - cl[0], lb[1] - cl[1]
        to_car_x, to_car_y = car_x - cl[0], car_y - cl[1]

        right_dist = (to_right_x * to_right_x + to_right_y * to_right_y) ** 0.5
        left_dist = (to_left_x * to_left_x + to_left_y * to_left_y) ** 0.5

        proj_right = (to_car_x * to_right_x + to_car_y * to_right_y) / right_dist if right_dist > 0 else 0
        proj_left = (to_car_x * to_left_x + to_car_y * to_left_y) / left_dist if left_dist > 0 else 0

        return proj_right > right_dist or proj_left > left_dist

    def update_track_violations(self, car_x: float, car_y: float):
        """Update track violation count based on current position."""
        is_violating = self.check_track_limits(car_x, car_y)

        if is_violating and not self.currently_violating:
            self.track_limit_violations += 1
            self.currently_violating = True
        elif not is_violating:
            self.currently_violating = False

    def update_lap_status(self, car_x: float, car_y: float):
        """Update lap started/finished status based on position relative to start."""
        start_x, start_y = self.rt.centerline[0, 0], self.rt.centerline[0, 1]
        dx = car_x - start_x
        dy = car_y - start_y
        progress_sq = dx * dx + dy * dy

        if progress_sq > 100.0 and not self.lap_started:  # 10^2 = 100
            self.lap_started = True
        elif progress_sq <= 1.0 and self.lap_started:
            self.lap_finished = True

    @abstractmethod
    def run(self) -> LapReport:
        """Run the simulation. Must be implemented by subclasses."""
        pass


class HeadlessSimulator(SimulatorBase):
    """Fast headless simulator for testing without visualization."""

    def run(self) -> LapReport:
        """Run the simulation until lap completion or timeout."""
        self.wall_start_time = time()

        # Cache frequently accessed values as local variables (faster than self.*)
        car = self.car
        parameters = car.parameters
        rt = self.rt
        time_step = car.time_step
        max_time = self.max_time

        # Pre-cache track data for fast access
        centerline = rt.centerline
        right_boundary = rt.right_boundary
        left_boundary = rt.left_boundary
        start_x, start_y = centerline[0, 0], centerline[0, 1]

        # Pre-allocate speed array (estimate max steps needed)
        max_steps = int(max_time / time_step) + 1
        speeds = np.empty(max_steps, dtype=np.float64)

        # Track distance
        total_distance = 0.0
        prev_x, prev_y = car.state[0], car.state[1]

        while not self.lap_finished and self.sim_time < max_time:
            # Run controller (car.state gets reassigned in update, so re-fetch it)
            desired = controller(car.state, parameters, rt)
            cont = lower_controller(car.state, desired, parameters)
            car.update(cont)

            # Get current position from updated state
            state = car.state
            car_x, car_y = state[0], state[1]
            velocity = state[3]

            # Track distance incrementally (avoid storing all positions)
            dx = car_x - prev_x
            dy = car_y - prev_y
            total_distance += (dx * dx + dy * dy) ** 0.5
            prev_x, prev_y = car_x, car_y

            # Store speed
            speeds[self.steps] = velocity
            self.sim_time += time_step
            self.steps += 1

            # Check track limits (inlined for speed - avoiding method call overhead)
            dx_arr = centerline[:, 0] - car_x
            dy_arr = centerline[:, 1] - car_y
            dist_sq = dx_arr * dx_arr + dy_arr * dy_arr
            closest_idx = np.argmin(dist_sq)

            cl = centerline[closest_idx]
            rb = right_boundary[closest_idx]
            lb = left_boundary[closest_idx]

            to_right_x, to_right_y = rb[0] - cl[0], rb[1] - cl[1]
            to_left_x, to_left_y = lb[0] - cl[0], lb[1] - cl[1]
            to_car_x, to_car_y = car_x - cl[0], car_y - cl[1]

            right_dist = (to_right_x * to_right_x + to_right_y * to_right_y) ** 0.5
            left_dist = (to_left_x * to_left_x + to_left_y * to_left_y) ** 0.5

            proj_right = (to_car_x * to_right_x + to_car_y * to_right_y) / right_dist if right_dist > 0 else 0
            proj_left = (to_car_x * to_left_x + to_car_y * to_left_y) / left_dist if left_dist > 0 else 0

            is_violating = proj_right > right_dist or proj_left > left_dist

            if is_violating and not self.currently_violating:
                self.track_limit_violations += 1
                self.currently_violating = True
            elif not is_violating:
                self.currently_violating = False

            # Check lap progress (inlined for speed)
            dx = car_x - start_x
            dy = car_y - start_y
            progress_sq = dx * dx + dy * dy

            if progress_sq > 100.0 and not self.lap_started:  # 10^2 = 100
                self.lap_started = True
            elif progress_sq <= 1.0 and self.lap_started:
                self.lap_finished = True

        wall_end = time()

        # Compile report from tracked speeds (slice to actual steps used)
        speed_slice = speeds[: self.steps] if self.steps > 0 else np.array([0.0])

        return LapReport(
            completed=self.lap_finished,
            lap_time=self.sim_time,
            track_violations=self.track_limit_violations,
            avg_speed=float(np.mean(speed_slice)),
            max_speed=float(np.max(speed_slice)),
            min_speed=float(np.min(speed_slice)),
            distance_traveled=total_distance,
            steps=self.steps,
            wall_time=wall_end - self.wall_start_time,
            dnf_reason=None if self.lap_finished else f"Timeout ({max_time}s exceeded)",
        )


class Simulator(SimulatorBase):
    """Visual simulator with matplotlib visualization."""

    def __init__(self, rt: RaceTrack, max_time: float = 300.0):
        super().__init__(rt, max_time)

        matplotlib.rcParams["figure.dpi"] = 300
        matplotlib.rcParams["font.size"] = 8

        self.figure, self.axis = plt.subplots(1, 1)
        self.axis.set_xlabel("X")
        self.axis.set_ylabel("Y")

        self.report_printed = False

        # Stats tracking (visual simulator keeps full history for plotting)
        self.speeds: list[float] = []
        self.positions: list[np.ndarray] = []

    def calculate_distance(self) -> float:
        """Calculate total distance traveled from position history."""
        if len(self.positions) < 2:
            return 0.0
        positions = np.array(self.positions)
        deltas = np.diff(positions, axis=0)
        distances = np.linalg.norm(deltas, axis=1)
        return float(np.sum(distances))

    def get_report(self) -> LapReport:
        """Generate lap report."""
        speeds = np.array(self.speeds) if self.speeds else np.array([0.0])
        wall_time = time() - self.wall_start_time if self.wall_start_time else 0.0

        return LapReport(
            completed=self.lap_finished,
            lap_time=self.sim_time,
            track_violations=self.track_limit_violations,
            avg_speed=float(np.mean(speeds)),
            max_speed=float(np.max(speeds)),
            min_speed=float(np.min(speeds)),
            distance_traveled=self.calculate_distance(),
            steps=self.steps,
            wall_time=wall_time,
        )

    def run(self) -> LapReport | None:
        try:
            if self.lap_finished:
                if not self.report_printed:
                    report = self.get_report()
                    report.print()
                    self.report_printed = True
                    return report
                return None

            self.figure.canvas.flush_events()
            self.axis.cla()

            self.rt.plot_track(self.axis)

            self.axis.set_xlim(self.car.state[0] - 200, self.car.state[0] + 200)
            self.axis.set_ylim(self.car.state[1] - 200, self.car.state[1] + 200)

            desired = controller(self.car.state, self.car.parameters, self.rt)
            cont = lower_controller(self.car.state, desired, self.car.parameters)
            self.car.update(cont)

            car_x, car_y = self.car.state[0], self.car.state[1]
            self.update_lap_status(car_x, car_y)
            self.update_track_violations(car_x, car_y)

            # Track stats
            self.steps += 1
            self.sim_time += self.car.time_step
            self.speeds.append(float(self.car.state[3]))
            self.positions.append(self.car.state[0:2].copy())

            self.axis.arrow(
                self.car.state[0],
                self.car.state[1],
                self.car.wheelbase * np.cos(self.car.state[4]),
                self.car.wheelbase * np.sin(self.car.state[4]),
            )

            self.axis.text(
                self.car.state[0] + 195,
                self.car.state[1] + 195,
                "Lap completed: " + str(self.lap_finished),
                horizontalalignment="right",
                verticalalignment="top",
                fontsize=8,
                color="Red",
            )

            self.axis.text(
                self.car.state[0] + 195,
                self.car.state[1] + 170,
                "Lap time: " + f"{self.sim_time:.2f}",
                horizontalalignment="right",
                verticalalignment="top",
                fontsize=8,
                color="Red",
            )

            self.axis.text(
                self.car.state[0] + 195,
                self.car.state[1] + 155,
                "Track violations: " + str(self.track_limit_violations),
                horizontalalignment="right",
                verticalalignment="top",
                fontsize=8,
                color="Red",
            )

            self.figure.canvas.draw()
            return None

        except KeyboardInterrupt:
            exit()

    def start(self):
        # Run the simulation loop every 1 millisecond.
        self.timer = self.figure.canvas.new_timer(interval=1)
        self.timer.add_callback(self.run)
        self.wall_start_time = time()
        self.timer.start()
