"""Unit tests for the quintic spline planner."""
import time

import numpy as np
import pytest

from overtaking_spline.frenet import FrenetConverter, build_reference_path
from overtaking_spline.spline_planner import (
    PlannerConfig,
    SplinePlanner,
    _eval_poly_and_derivs,
    quintic_coeffs,
)


def _converter(length: float = 30.0, n: int = 301):
    xs = np.linspace(0.0, length, n)
    ys = np.zeros_like(xs)
    vs = 4.0 * np.ones_like(xs)
    ref = build_reference_path(xs, ys, vs)
    conv = FrenetConverter()
    conv.update_reference(ref)
    return conv


def test_quintic_satisfies_boundary_conditions():
    coeffs = quintic_coeffs(0.1, 0.2, 0.0, 0.0, 0.0, 0.0, L=5.0)
    p0, p0p, p0pp = _eval_poly_and_derivs(coeffs, np.array([0.0]))
    pL, pLp, pLpp = _eval_poly_and_derivs(coeffs, np.array([5.0]))
    assert p0[0] == pytest.approx(0.1)
    assert p0p[0] == pytest.approx(0.2)
    assert p0pp[0] == pytest.approx(0.0, abs=1e-9)
    assert pL[0] == pytest.approx(0.0, abs=1e-9)
    assert pLp[0] == pytest.approx(0.0, abs=1e-9)
    assert pLpp[0] == pytest.approx(0.0, abs=1e-9)


def test_planner_returns_feasible_with_clear_track():
    conv = _converter()
    cfg = PlannerConfig(track_width=2.0, car_width=0.3, safety_margin=0.1,
                        lookahead=6.0, rejoin_distance=3.0, n_candidates=7)
    planner = SplinePlanner(conv, cfg)
    # Opponent directly ahead on centerline: a feasible side-pass should exist.
    result = planner.plan(
        ego_s=1.0, ego_d=0.0,
        opp_s=5.0, opp_d=0.0,
        ego_speed=4.0,
    )
    assert result.chosen is not None, f"reason={result.reason}"
    assert result.side != 0
    # Chosen path should bulge off centerline somewhere.
    assert np.max(np.abs(result.chosen.d_samples)) > 0.2


def test_planner_rejects_when_no_usable_width():
    conv = _converter()
    cfg = PlannerConfig(track_width=0.4, car_width=0.3, safety_margin=0.1)
    # usable = max(0, 0.2 - 0.15 - 0.1) = 0 -> no_drivable_width
    planner = SplinePlanner(conv, cfg)
    result = planner.plan(0.0, 0.0, 3.0, 0.0, 4.0)
    assert result.chosen is None
    assert result.reason == "no_drivable_width"


def test_planner_runs_fast_enough():
    conv = _converter()
    cfg = PlannerConfig(n_candidates=7, n_samples=50)
    planner = SplinePlanner(conv, cfg)
    # Warm-up
    planner.plan(1.0, 0.0, 5.0, 0.0, 4.0)
    n_iter = 100
    t0 = time.perf_counter()
    for _ in range(n_iter):
        planner.plan(1.0, 0.0, 5.0, 0.0, 4.0)
    avg_ms = (time.perf_counter() - t0) / n_iter * 1e3
    # Generous: keep planner compute well under 4 ms to leave budget for I/O.
    assert avg_ms < 4.0, f"planner avg {avg_ms:.2f} ms exceeds 4 ms budget"


def test_planner_bias_to_racing_line_side():
    conv = _converter()
    cfg = PlannerConfig(track_width=2.0, car_width=0.3, safety_margin=0.1,
                        racing_line_side=+1, racing_line_bias=0.5,
                        w_obstacle=1.0)  # weaken obstacle so bias dominates
    planner = SplinePlanner(conv, cfg)
    result = planner.plan(0.0, 0.0, 5.0, 0.0, 4.0)
    assert result.chosen is not None
    assert result.chosen.d_target > 0, (
        f"expected left-side pass when racing_line_side=+1, "
        f"got d_target={result.chosen.d_target}"
    )
