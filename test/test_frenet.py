"""Unit tests for the Frenet converter. No ROS."""
import numpy as np
import pytest

from overtaking_spline.frenet import FrenetConverter, build_reference_path


def _straight_reference(length: float = 10.0, n: int = 101):
    xs = np.linspace(0.0, length, n)
    ys = np.zeros_like(xs)
    return build_reference_path(xs, ys)


def _circle_reference(radius: float = 5.0, n: int = 361):
    thetas = np.linspace(0.0, 2 * np.pi, n)
    xs = radius * np.cos(thetas)
    ys = radius * np.sin(thetas)
    return build_reference_path(xs, ys)


def test_straight_cartesian_to_frenet_roundtrip():
    ref = _straight_reference()
    conv = FrenetConverter()
    conv.update_reference(ref)
    for x, d in [(1.0, 0.5), (5.0, -0.3), (9.0, 0.0)]:
        s, d_out, _ = conv.cartesian_to_frenet(x, d)
        assert s == pytest.approx(x, abs=0.1)
        assert d_out == pytest.approx(d, abs=1e-6)


def test_frenet_to_cartesian_straight():
    ref = _straight_reference()
    conv = FrenetConverter()
    conv.update_reference(ref)
    x, y = conv.frenet_to_cartesian(3.0, 0.5)
    assert x == pytest.approx(3.0, abs=1e-2)
    assert y == pytest.approx(0.5, abs=1e-2)


def test_frenet_batch_matches_scalar():
    ref = _straight_reference()
    conv = FrenetConverter()
    conv.update_reference(ref)
    ss = np.linspace(0.0, 9.0, 20)
    ds = np.linspace(-0.5, 0.5, 20)
    xs_b, ys_b = conv.frenet_to_cartesian_batch(ss, ds)
    for i, (s, d) in enumerate(zip(ss, ds)):
        x, y = conv.frenet_to_cartesian(s, d)
        assert xs_b[i] == pytest.approx(x, abs=1e-2)
        assert ys_b[i] == pytest.approx(y, abs=1e-2)


def test_circle_wrap_delta_s():
    ref = _circle_reference()
    conv = FrenetConverter()
    conv.update_reference(ref)
    total = ref.total_length
    # From s near end back to s near start should wrap forward, not give a huge value.
    forward = conv.wrap_delta_s(0.1, total - 0.1)
    assert forward == pytest.approx(0.2, abs=1e-1)


def test_left_offset_is_positive_normal():
    ref = _straight_reference()
    conv = FrenetConverter()
    conv.update_reference(ref)
    # A point at (3, +0.5) lies left of the +x heading -> d > 0
    _, d, _ = conv.cartesian_to_frenet(3.0, 0.5)
    assert d > 0
    _, d2, _ = conv.cartesian_to_frenet(3.0, -0.5)
    assert d2 < 0
