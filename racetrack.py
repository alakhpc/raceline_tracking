import matplotlib.axes as axes
import matplotlib.patches as patches
import matplotlib.path as path
import numpy as np


def resample(path_arr: np.ndarray, n: int) -> np.ndarray:
    """Resample path to have exactly n points using linear interpolation."""
    if len(path_arr) == n:
        return path_arr
    indices = np.linspace(0, len(path_arr) - 1, n)
    x = np.interp(indices, np.arange(len(path_arr)), path_arr[:, 0])
    y = np.interp(indices, np.arange(len(path_arr)), path_arr[:, 1])
    return np.column_stack((x, y))


class RaceTrack:
    def __init__(self, filepath: str, raceline_path: str):
        data = np.loadtxt(filepath, comments="#", delimiter=",")
        self.centerline = data[:, 0:2]
        self.centerline = np.vstack((self.centerline[-1], self.centerline, self.centerline[0]))

        # Load raceline
        raceline_data = np.loadtxt(raceline_path, comments="#", delimiter=",")
        self.raceline = raceline_data[:, 0:2]
        self.raceline = np.vstack((self.raceline[-1], self.raceline, self.raceline[0]))

        centerline_gradient = np.gradient(self.centerline, axis=0)
        # Unfortunate Warning Print: https://github.com/numpy/numpy/issues/26620
        centerline_cross = np.cross(centerline_gradient, np.array([0.0, 0.0, 1.0]))
        centerline_norm = centerline_cross * np.divide(1.0, np.linalg.norm(centerline_cross, axis=1))[:, None]

        centerline_norm = np.delete(centerline_norm, 0, axis=0)
        centerline_norm = np.delete(centerline_norm, -1, axis=0)

        self.centerline = np.delete(self.centerline, 0, axis=0)
        self.centerline = np.delete(self.centerline, -1, axis=0)

        # Process raceline the same way
        self.raceline = np.delete(self.raceline, 0, axis=0)
        self.raceline = np.delete(self.raceline, -1, axis=0)
        # Resample raceline to match centerline length for blending
        self.raceline = resample(self.raceline, len(self.centerline))

        # Compute track left and right boundaries
        self.right_boundary = self.centerline[:, :2] + centerline_norm[:, :2] * np.expand_dims(data[:, 2], axis=1)
        self.left_boundary = self.centerline[:, :2] - centerline_norm[:, :2] * np.expand_dims(data[:, 3], axis=1)

        # Compute initial position and heading
        self.initial_state = np.array(
            [
                self.centerline[0, 0],
                self.centerline[0, 1],
                0.0,
                0.0,
                np.arctan2(
                    self.centerline[1, 1] - self.centerline[0, 1],
                    self.centerline[1, 0] - self.centerline[0, 0],
                ),
            ]
        )

        # Matplotlib Plots
        self.code = np.empty(self.centerline.shape[0], dtype=np.uint8)
        self.code.fill(path.Path.LINETO)
        self.code[0] = path.Path.MOVETO
        self.code[-1] = path.Path.CLOSEPOLY

        self.mpl_centerline = path.Path(self.centerline, self.code)
        self.mpl_right_track_limit = path.Path(self.right_boundary, self.code)
        self.mpl_left_track_limit = path.Path(self.left_boundary, self.code)

        self.mpl_centerline_patch = patches.PathPatch(self.mpl_centerline, linestyle="-", fill=False, lw=0.3)
        self.mpl_right_track_limit_patch = patches.PathPatch(
            self.mpl_right_track_limit, linestyle="--", fill=False, lw=0.2
        )
        self.mpl_left_track_limit_patch = patches.PathPatch(
            self.mpl_left_track_limit, linestyle="--", fill=False, lw=0.2
        )

    def plot_track(self, axis: axes.Axes):
        axis.add_patch(self.mpl_centerline_patch)
        axis.add_patch(self.mpl_right_track_limit_patch)
        axis.add_patch(self.mpl_left_track_limit_patch)
