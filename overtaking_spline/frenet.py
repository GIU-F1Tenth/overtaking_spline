"""Frenet-frame converter for a reference path.

Vendored from `lidar_obj_detection.frenet_utils` with two changes:
1. KD-tree replaces O(N) cdist lookup so cartesian_to_frenet is O(log N).
2. Tangents/normals are precomputed once at update_reference() so the
   hot-loop only does a tree query plus a few dot products.

No ROS dependency. Pure numpy + scipy.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Optional, Tuple

import numpy as np
from scipy.interpolate import interp1d
from scipy.spatial import cKDTree


@dataclass
class ReferencePath:
    """Arc-length-parameterized reference path."""
    s: np.ndarray  # shape (N,)
    x: np.ndarray  # shape (N,)
    y: np.ndarray  # shape (N,)
    v: np.ndarray  # shape (N,) reference velocity per point
    total_length: float


def build_reference_path(xs: np.ndarray, ys: np.ndarray,
                         vs: Optional[np.ndarray] = None) -> ReferencePath:
    """Compute arc-length parameterization from raw waypoints."""
    xs = np.asarray(xs, dtype=float)
    ys = np.asarray(ys, dtype=float)
    if xs.shape != ys.shape or xs.ndim != 1:
        raise ValueError("xs and ys must be 1-D arrays of equal length")
    if xs.size < 2:
        raise ValueError("need at least 2 waypoints")

    dx = np.diff(xs)
    dy = np.diff(ys)
    seg = np.hypot(dx, dy)
    s = np.concatenate(([0.0], np.cumsum(seg)))

    if vs is None:
        vs_arr = np.zeros_like(s)
    else:
        vs_arr = np.asarray(vs, dtype=float)
        if vs_arr.shape != xs.shape:
            raise ValueError("vs must match xs/ys length")

    return ReferencePath(s=s, x=xs, y=ys, v=vs_arr, total_length=float(s[-1]))


class FrenetConverter:
    """Converter between Cartesian (x, y) and Frenet (s, d) on a reference path."""

    def __init__(self) -> None:
        self._ref: Optional[ReferencePath] = None
        self._tree: Optional[cKDTree] = None
        self._tangents: Optional[np.ndarray] = None  # (N, 2)
        self._normals: Optional[np.ndarray] = None   # (N, 2)
        self._interp_x: Optional[interp1d] = None
        self._interp_y: Optional[interp1d] = None
        self._interp_v: Optional[interp1d] = None

    @property
    def ready(self) -> bool:
        return self._ref is not None

    @property
    def reference(self) -> Optional[ReferencePath]:
        return self._ref

    def update_reference(self, ref: ReferencePath) -> None:
        self._ref = ref
        pts = np.column_stack((ref.x, ref.y))
        self._tree = cKDTree(pts)

        # Tangent per vertex via forward difference, last point copies previous.
        tan = np.empty_like(pts)
        tan[:-1] = pts[1:] - pts[:-1]
        tan[-1] = tan[-2]
        norms = np.linalg.norm(tan, axis=1, keepdims=True)
        norms[norms == 0] = 1.0
        tan /= norms
        self._tangents = tan
        # Left-hand normal: rotate tangent 90 deg CCW.
        self._normals = np.column_stack((-tan[:, 1], tan[:, 0]))

        self._interp_x = interp1d(ref.s, ref.x, kind="cubic",
                                  fill_value="extrapolate", assume_sorted=True)
        self._interp_y = interp1d(ref.s, ref.y, kind="cubic",
                                  fill_value="extrapolate", assume_sorted=True)
        self._interp_v = interp1d(ref.s, ref.v, kind="linear",
                                  fill_value="extrapolate", assume_sorted=True)

    def cartesian_to_frenet(self, x: float, y: float) -> Tuple[float, float, int]:
        """Return (s, d, nearest_idx). d > 0 is left of the reference."""
        if self._tree is None or self._ref is None:
            raise RuntimeError("FrenetConverter has no reference")
        _, idx = self._tree.query([x, y])
        ref = self._ref
        tan = self._tangents[idx]
        nor = self._normals[idx]
        dx = x - ref.x[idx]
        dy = y - ref.y[idx]
        s_offset = dx * tan[0] + dy * tan[1]
        d = dx * nor[0] + dy * nor[1]
        s = ref.s[idx] + s_offset
        if ref.total_length > 0:
            s = s % ref.total_length
        return float(s), float(d), int(idx)

    def frenet_to_cartesian(self, s: float, d: float) -> Tuple[float, float]:
        if self._interp_x is None or self._ref is None:
            raise RuntimeError("FrenetConverter has no reference")
        total = self._ref.total_length
        if total > 0:
            s = s % total
        x_c = float(self._interp_x(s))
        y_c = float(self._interp_y(s))
        ds = 1e-2
        x_a = float(self._interp_x(s + ds))
        y_a = float(self._interp_y(s + ds))
        tx = x_a - x_c
        ty = y_a - y_c
        norm = (tx * tx + ty * ty) ** 0.5
        if norm == 0:
            return x_c, y_c
        tx /= norm
        ty /= norm
        nx = -ty
        ny = tx
        return x_c + d * nx, y_c + d * ny

    def frenet_to_cartesian_batch(self, ss: np.ndarray, ds: np.ndarray
                                  ) -> Tuple[np.ndarray, np.ndarray]:
        """Vectorized version. Returns (xs, ys)."""
        if self._interp_x is None or self._ref is None:
            raise RuntimeError("FrenetConverter has no reference")
        total = self._ref.total_length
        if total > 0:
            ss = np.mod(ss, total)
        eps = 1e-2
        x_c = self._interp_x(ss)
        y_c = self._interp_y(ss)
        x_a = self._interp_x(ss + eps)
        y_a = self._interp_y(ss + eps)
        tx = x_a - x_c
        ty = y_a - y_c
        n = np.hypot(tx, ty)
        n[n == 0] = 1.0
        tx /= n
        ty /= n
        nx = -ty
        ny = tx
        return x_c + ds * nx, y_c + ds * ny

    def velocity_at(self, s: float) -> float:
        if self._interp_v is None or self._ref is None:
            return 0.0
        total = self._ref.total_length
        if total > 0:
            s = s % total
        return float(self._interp_v(s))

    def velocity_at_batch(self, ss: np.ndarray) -> np.ndarray:
        if self._interp_v is None or self._ref is None:
            return np.zeros_like(ss)
        total = self._ref.total_length
        if total > 0:
            ss = np.mod(ss, total)
        return np.asarray(self._interp_v(ss), dtype=float)

    def wrap_delta_s(self, s_to: float, s_from: float) -> float:
        """Signed forward arc-length from s_from to s_to on a (possibly) looped track."""
        if self._ref is None or self._ref.total_length <= 0:
            return s_to - s_from
        total = self._ref.total_length
        delta = (s_to - s_from) % total
        return delta
