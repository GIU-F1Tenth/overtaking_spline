"""Werling-style quintic lateral polynomial planner in Frenet coordinates.

Generates a family of candidate paths d(s) parameterized as quintic
polynomials over a longitudinal window [s0, s0 + L]. Boundary conditions
are matched to the ego state (d, d', d'') at the branch point and to
(d_target, 0, 0) at the rejoin point, so by construction:
- position, heading, and curvature are continuous at the branch (C^2)
- the candidate flattens into the reference path at the rejoin (C^2)

Each candidate is sampled, converted back to Cartesian via the Frenet
converter, scored against:
- obstacle clearance (Frenet d distance to opponent in the longitudinal corridor)
- maximum curvature -> lateral acceleration feasibility at the reference speed
- bias toward racing-line side when clearances are comparable

No ROS dependency.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import List, Optional, Sequence, Tuple

import numpy as np

from overtaking_spline.frenet import FrenetConverter


@dataclass
class PlannerConfig:
    car_width: float = 0.30
    safety_margin: float = 0.20
    track_width: float = 1.5
    lookahead: float = 6.0  # longitudinal length of the maneuver [m]
    rejoin_distance: float = 4.0  # extra distance past opponent before re-flattening [m]
    n_samples: int = 50  # samples along the spline
    n_candidates: int = 7  # lateral-offset candidates
    max_lateral_acc: float = 6.0  # [m/s^2]
    racing_line_side: int = 0  # -1 right, +1 left, 0 no bias
    racing_line_bias: float = 0.10  # fractional clearance band for bias to apply
    w_obstacle: float = 50.0
    w_curvature: float = 1.0
    w_offset: float = 0.5
    w_bias: float = 2.0

    def usable_half_width(self) -> float:
        """Half-width available to the *vehicle center line*."""
        return max(0.0, self.track_width / 2.0 - self.car_width / 2.0 - self.safety_margin)


@dataclass
class Candidate:
    d_target: float
    s_samples: np.ndarray  # (N,)
    d_samples: np.ndarray  # (N,)
    xs: np.ndarray         # (N,)
    ys: np.ndarray         # (N,)
    max_curvature: float
    min_clearance: float   # absolute Frenet d-distance to opponent center
    feasible: bool
    cost: float


@dataclass
class PlanResult:
    chosen: Optional[Candidate]
    candidates: List[Candidate]
    side: int  # sign of d_target of chosen; 0 if none
    reason: str  # short tag explaining the choice / rejection


def quintic_coeffs(d0: float, d0p: float, d0pp: float,
                   d1: float, d1p: float, d1pp: float,
                   L: float) -> np.ndarray:
    """Solve for [a0..a5] s.t. p(0)=d0, p'(0)=d0', p''(0)=d0'',
    p(L)=d1, p'(L)=d1', p''(L)=d1''. p(u) = sum a_i u^i, u in [0, L]."""
    if L <= 0:
        raise ValueError("L must be positive")
    a0 = d0
    a1 = d0p
    a2 = 0.5 * d0pp
    # Solve 3x3 for a3, a4, a5 using the L-side conditions.
    L2 = L * L
    L3 = L2 * L
    L4 = L3 * L
    L5 = L4 * L
    A = np.array([
        [L3,        L4,        L5       ],
        [3 * L2,    4 * L3,    5 * L4   ],
        [6 * L,     12 * L2,   20 * L3  ],
    ])
    rhs = np.array([
        d1 - a0 - a1 * L - a2 * L2,
        d1p - a1 - 2 * a2 * L,
        d1pp - 2 * a2,
    ])
    a345 = np.linalg.solve(A, rhs)
    return np.array([a0, a1, a2, a345[0], a345[1], a345[2]])


def _eval_poly_and_derivs(coeffs: np.ndarray, u: np.ndarray
                          ) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    a0, a1, a2, a3, a4, a5 = coeffs
    u2 = u * u
    u3 = u2 * u
    u4 = u3 * u
    u5 = u4 * u
    p   = a0 + a1 * u + a2 * u2 + a3 * u3 + a4 * u4 + a5 * u5
    pd  = a1 + 2 * a2 * u + 3 * a3 * u2 + 4 * a4 * u3 + 5 * a5 * u4
    pdd = 2 * a2 + 6 * a3 * u + 12 * a4 * u2 + 20 * a5 * u3
    return p, pd, pdd


def _curvature_from_dprime(dp: np.ndarray, dpp: np.ndarray) -> np.ndarray:
    """First-order curvature approximation for a path parameterized by s.

    Assumes reference curvature is small relative to maneuver scale, which is
    the standard simplification used in Werling-style planners for short
    Frenet maneuvers. kappa ~ d''(s) / (1 + d'(s)^2)^1.5
    """
    return np.abs(dpp) / np.power(1.0 + dp * dp, 1.5)


class SplinePlanner:
    """Generates and scores quintic-lateral candidates."""

    def __init__(self, converter: FrenetConverter, config: PlannerConfig) -> None:
        self.converter = converter
        self.config = config

    def plan(self,
             ego_s: float, ego_d: float,
             opp_s: float, opp_d: float,
             ego_speed: float,
             ) -> PlanResult:
        cfg = self.config
        usable = cfg.usable_half_width()
        if usable <= 0:
            return PlanResult(None, [], 0, "no_drivable_width")

        # Maneuver longitudinal window: start at ego, end past the opponent.
        ds_to_opp = self.converter.wrap_delta_s(opp_s, ego_s)
        L = max(cfg.lookahead, ds_to_opp + cfg.rejoin_distance)
        s0 = ego_s

        # Candidate target offsets: spread across the usable band, pruned to feasible.
        targets = np.linspace(-usable, usable, cfg.n_candidates)

        # Boundary conditions at the rejoin point: zero offset, zero slope, zero curvature.
        d1, d1p, d1pp = 0.0, 0.0, 0.0

        # At the branch point we start exactly from the ego's current Frenet state.
        # We assume zero lateral velocity and curvature -- the controller will
        # smoothly absorb the residual; this avoids needing finite differences
        # over noisy odometry.
        d0, d0p, d0pp = ego_d, 0.0, 0.0

        # NOTE: the spline targets are absolute Frenet d offsets; if the ego is
        # slightly off-line at s0, the candidate naturally transitions from
        # ego_d to d_target without artifacts.

        # Speed for feasibility check, with a floor to avoid divide-by-zero.
        v_check = max(ego_speed, 0.5)
        kappa_max_allowed = cfg.max_lateral_acc / (v_check * v_check)

        u_samples = np.linspace(0.0, L, cfg.n_samples)
        # The Frenet "s" of each sample.
        s_samples_abs = s0 + u_samples

        # Two-segment design: branch quintic from ego state to (d_target, 0, 0)
        # at u_peak (placed near the opponent), then rejoin quintic from
        # (d_target, 0, 0) to (0, 0, 0) at u = L. Both segments are C^2 at
        # the seam by construction since the seam conditions match.
        u_peak = float(np.clip(ds_to_opp, 0.25 * L, 0.75 * L))
        L_branch = u_peak
        L_rejoin = L - u_peak

        candidates: List[Candidate] = []
        for d_target in targets:
            coeffs_branch = quintic_coeffs(
                d0, d0p, d0pp,
                d_target, 0.0, 0.0,
                L_branch,
            )
            coeffs_rejoin = quintic_coeffs(
                d_target, 0.0, 0.0,
                d1, d1p, d1pp,
                L_rejoin,
            )

            mask_branch = u_samples <= u_peak
            u_b = u_samples[mask_branch]
            u_r = u_samples[~mask_branch] - u_peak

            db, dpb, dppb = _eval_poly_and_derivs(coeffs_branch, u_b)
            dr, dpr, dppr = _eval_poly_and_derivs(coeffs_rejoin, u_r)
            d_samples = np.concatenate((db, dr))
            dp_samples = np.concatenate((dpb, dpr))
            dpp_samples = np.concatenate((dppb, dppr))

            # Hard cap: any sample outside usable corridor is infeasible.
            within_corridor = np.all(np.abs(d_samples) <= usable + 1e-6)

            xs, ys = self.converter.frenet_to_cartesian_batch(
                s_samples_abs, d_samples
            )
            kappa = _curvature_from_dprime(dp_samples, dpp_samples)
            max_kappa = float(np.max(kappa))
            feasible_curv = max_kappa <= kappa_max_allowed

            # Clearance to opponent: minimum |d(s) - opp_d| within ±1m of opp_s.
            window = np.abs(self._sample_arc_offset(s_samples_abs, opp_s)) <= 1.0
            if np.any(window):
                clearance = float(np.min(np.abs(d_samples[window] - opp_d)))
            else:
                clearance = float(np.min(np.abs(d_samples - opp_d)))

            opponent_blocked = clearance < (cfg.car_width / 2.0 + cfg.safety_margin)
            feasible = within_corridor and feasible_curv and not opponent_blocked

            # Cost composition.
            cost = (
                cfg.w_curvature * (max_kappa / max(kappa_max_allowed, 1e-6))
                + cfg.w_offset * (abs(d_target) / max(usable, 1e-6))
                - cfg.w_obstacle * clearance
            )
            if cfg.racing_line_side != 0:
                # Bias bonus if this candidate is on the racing-line side AND
                # clearance is within a small fraction of the best alternative.
                if np.sign(d_target) == cfg.racing_line_side:
                    cost -= cfg.w_bias * cfg.racing_line_bias

            candidates.append(Candidate(
                d_target=float(d_target),
                s_samples=s_samples_abs,
                d_samples=d_samples,
                xs=xs,
                ys=ys,
                max_curvature=max_kappa,
                min_clearance=clearance,
                feasible=feasible,
                cost=float(cost),
            ))

        feasible_cands = [c for c in candidates if c.feasible]
        if not feasible_cands:
            return PlanResult(None, candidates, 0, "no_feasible_candidate")

        chosen = min(feasible_cands, key=lambda c: c.cost)
        side = int(np.sign(chosen.d_target)) if chosen.d_target != 0 else 0
        return PlanResult(chosen, candidates, side, "ok")

    def _sample_arc_offset(self, s_samples_abs: np.ndarray, s_ref: float) -> np.ndarray:
        """Signed shortest arc-length offset for each sample relative to s_ref."""
        ref = self.converter.reference
        if ref is None or ref.total_length <= 0:
            return s_samples_abs - s_ref
        total = ref.total_length
        raw = (s_samples_abs - s_ref) % total
        # Map to [-total/2, total/2)
        raw = np.where(raw > total / 2, raw - total, raw)
        return raw
