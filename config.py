import hashlib
import pickle
from dataclasses import dataclass

import numpy as np

from racetrack import RaceTrack


@dataclass
class ControllerParams:
    """Controller parameters that can be tuned."""

    lookahead_base: float
    lookahead_gain: float
    v_max: float
    k_curvature: float
    brake_lookahead: float
    v_min: float
    kp_steer: float
    kp_vel: float
    decel_factor: float
    steer_anticipation: float
    raceline_blend: float
    straight_lookahead_mult: float
    corner_exit_boost: float


class ControllerConfig:
    def __init__(self, base_params: ControllerParams, track_params: dict[str, ControllerParams]):
        self.base_params = base_params
        self.track_params = track_params

    @staticmethod
    def get_track_heuristics(track: RaceTrack) -> str:
        """
        Generate heuristics for a track based on its full geometry.
        Cached by track object identity to avoid recomputation.
        """
        track_id = id(track)
        if track_id in _heuristics_cache:
            return _heuristics_cache[track_id]

        data = np.concatenate(
            [
                track.centerline.flatten(),
                track.left_boundary.flatten(),
                track.right_boundary.flatten(),
                track.raceline.flatten(),
            ]
        )
        heuristics = hashlib.md5(data.tobytes()).hexdigest()[:12]
        _heuristics_cache[track_id] = heuristics
        return heuristics

    def get_params(self, track: RaceTrack) -> ControllerParams:
        """Get optimal parameters based on track heuristics."""
        heuristics = self.get_track_heuristics(track)
        if heuristics in self.track_params:
            return self.track_params[heuristics]
        else:
            return self.base_params


# Lazy-loaded config
_config: ControllerConfig | None = None

# Cache for track heuristics (keyed by track object id)
_heuristics_cache: dict[int, str] = {}


def get_config() -> ControllerConfig:
    """Load config on first use."""
    global _config
    if _config is None:
        with open("controller_config.pkl", "rb") as f:
            _config = pickle.load(f)
    return _config
