"""Minimal pure-pursuit follower for the locally generated overtake path.

Stateless: each call takes the ego pose, the candidate (xs, ys, v),
lookahead distance, wheelbase, and returns (steering, speed).

Kept here so the package can self-produce an AckermannDriveStamped on its
own /overtaking_spline/drive topic without depending on the external
pure_pursuit package. The real-life loop will route this via
control_gateway just like /dwa/drive.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Tuple

import numpy as np


@dataclass
class FollowerConfig:
    lookahead: float = 1.2
    wheelbase: float = 0.33
    max_steer: float = 0.4  # rad
    speed_scale: float = 1.0  # applied to reference v at the lookahead point


def pursue(ego_x: float, ego_y: float, ego_yaw: float,
           xs: np.ndarray, ys: np.ndarray, vs: np.ndarray,
           cfg: FollowerConfig) -> Tuple[float, float]:
    """Standard Ackermann pure pursuit on (xs, ys) with per-point reference speeds.

    Returns (steering_angle [rad], speed [m/s]).
    """
    if xs.size == 0:
        return 0.0, 0.0

    dx = xs - ego_x
    dy = ys - ego_y
    dist = np.hypot(dx, dy)
    # Find the first point on the path past the lookahead distance.
    ahead_mask = dist >= cfg.lookahead
    if not np.any(ahead_mask):
        idx = int(np.argmax(dist))  # farthest point on path
    else:
        idx = int(np.argmax(ahead_mask))  # first True

    tx = xs[idx] - ego_x
    ty = ys[idx] - ego_y
    # Rotate goal into vehicle frame.
    cos_y = np.cos(-ego_yaw)
    sin_y = np.sin(-ego_yaw)
    gx = cos_y * tx - sin_y * ty
    gy = sin_y * tx + cos_y * ty

    L = max(np.hypot(gx, gy), 1e-3)
    curvature = 2.0 * gy / (L * L)
    steer = float(np.arctan(curvature * cfg.wheelbase))
    steer = float(np.clip(steer, -cfg.max_steer, cfg.max_steer))

    speed = float(vs[idx]) * cfg.speed_scale if vs.size > 0 else 0.0
    return steer, speed
