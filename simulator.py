from time import time

import matplotlib
import matplotlib.pyplot as plt
import numpy as np

from controller import neat_controller
from racecar import RaceCar
from racetrack import RaceTrack


class Simulator:
    def __init__(self, rt: RaceTrack, neat_net):
        """
        Initialize visual simulator.

        Args:
            rt: RaceTrack object containing track geometry.
            neat_net: NEAT RecurrentNetwork instance.
        """
        matplotlib.rcParams["figure.dpi"] = 300
        matplotlib.rcParams["font.size"] = 8

        self.rt = rt
        self.neat_net = neat_net
        self.figure, self.axis = plt.subplots(1, 1)

        self.axis.set_xlabel("X")
        self.axis.set_ylabel("Y")

        self.car = RaceCar(self.rt.initial_state.T)

        self.lap_time_elapsed = 0
        self.lap_start_time = None
        self.sim_time_elapsed = 0
        self.lap_finished = False
        self.lap_started = False
        self.track_limit_violations = 0
        self.currently_violating = False

        # Cache for optimization
        self._closest_idx = 0

        # Reset NEAT network state
        self.neat_net.reset()

    def check_track_limits(self, closest_idx: int = None):
        car_position = self.car.state[0:2]

        if closest_idx is None:
            centerline_distances = np.linalg.norm(self.rt.centerline - car_position, axis=1)
            closest_idx = np.argmin(centerline_distances)

        self._closest_idx = closest_idx

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

    def run(self):
        try:
            if self.lap_finished:
                self.print_results()
                exit()

            self.figure.canvas.flush_events()
            self.axis.cla()

            self.rt.plot_track(self.axis)

            self.axis.set_xlim(self.car.state[0] - 200, self.car.state[0] + 200)
            self.axis.set_ylim(self.car.state[1] - 200, self.car.state[1] + 200)

            # NEAT controller: outputs steering_rate and acceleration directly
            cont, closest_idx = neat_controller(
                self.car.state,
                self.car.parameters,
                self.rt,
                self.neat_net,
                closest_idx_hint=self._closest_idx,
                return_closest_idx=True,
            )
            self.car.update(cont)

            if self.lap_started and not self.lap_finished:
                self.sim_time_elapsed += self.car.time_step
            self.update_status()
            self.check_track_limits(closest_idx=closest_idx)

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
                "Wall time: " + f"{self.lap_time_elapsed:.2f}",
                horizontalalignment="right",
                verticalalignment="top",
                fontsize=8,
                color="Red",
            )

            self.axis.text(
                self.car.state[0] + 195,
                self.car.state[1] + 155,
                "Sim time: " + f"{self.sim_time_elapsed:.2f}",
                horizontalalignment="right",
                verticalalignment="top",
                fontsize=8,
                color="Red",
            )

            self.axis.text(
                self.car.state[0] + 195,
                self.car.state[1] + 140,
                "Track violations: " + str(self.track_limit_violations),
                horizontalalignment="right",
                verticalalignment="top",
                fontsize=8,
                color="Red",
            )

            self.figure.canvas.draw()
            return True

        except KeyboardInterrupt:
            exit()

    def update_status(self):
        progress = np.linalg.norm(self.car.state[0:2] - self.rt.centerline[0, 0:2], 2)

        if progress > 10.0 and not self.lap_started:
            self.lap_started = True
            self.sim_time_elapsed = 0

        if progress <= 10.0 and self.lap_started and not self.lap_finished:
            self.lap_finished = True
            self.lap_time_elapsed = time() - self.lap_start_time

        if not self.lap_finished and self.lap_start_time is not None:
            self.lap_time_elapsed = time() - self.lap_start_time

    def print_results(self):
        """Print simulation results to console."""
        print("\n" + "=" * 50)
        print("SIMULATION RESULTS")
        print("=" * 50)
        print(f"Lap completed:       {self.lap_finished}")
        print(f"Sim time:            {self.sim_time_elapsed:.2f}s")
        print(f"Wall time:           {self.lap_time_elapsed:.2f}s")
        print(f"Track violations:    {self.track_limit_violations}")
        print("=" * 50)

    def start(self):
        # Run the simulation loop every 1 millisecond.
        self.timer = self.figure.canvas.new_timer(interval=1)
        self.timer.add_callback(self.run)
        self.lap_start_time = time()
        self.timer.start()
