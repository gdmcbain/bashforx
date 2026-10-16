import diffrax
import jax
import jax.numpy as jnp
import numpy as np
import pytest

import bashforx as bfx


def test_zero_gravity_profile_is_a_sphere():
    # With c = 0 the Young–Laplace solution is a circle of radius b through the apex.
    b = 1.3
    s = jnp.linspace(0.0, 2.5, 50)
    ys = bfx.solve_profile(b, 0.0, s[-1], saveat=diffrax.SaveAt(ts=s)).ys
    np.testing.assert_allclose(ys[:, 0], b * jnp.sin(s / b), atol=1e-8)
    np.testing.assert_allclose(ys[:, 1], b * (1 - jnp.cos(s / b)), atol=1e-8)
    np.testing.assert_allclose(ys[:, 2], s / b, atol=1e-8)


def test_profile_is_mirror_symmetric():
    s = jnp.array([-1.0, -0.3, 0.3, 1.0])
    p = bfx.profile_points(1.5, 0.15, 2.0, 1.0, s)
    np.testing.assert_allclose(p[:2, 0] - 2.0, -(p[2:, 0] - 2.0)[::-1], atol=1e-12)
    np.testing.assert_allclose(p[:2, 1], p[2:, 1][::-1], atol=1e-12)


def test_rhs_is_finite_with_finite_gradient_at_apex():
    y0 = jnp.zeros(3)
    f = bfx.young_laplace(0.0, y0, (1.5, 0.1))
    np.testing.assert_allclose(f, [1.0, 0.0, 1.0 / 1.5])
    jac = jax.jacfwd(lambda y: bfx.young_laplace(0.0, y, (1.5, 0.1)))(y0)
    assert bool(jnp.all(jnp.isfinite(jac)))


def test_forward_mode_gradient_through_solver_matches_finite_difference():
    s = jnp.linspace(0.1, 3.0, 20)

    def loss(theta):
        return jnp.sum(bfx.profile_points(theta[0], theta[1], 0.0, 0.0, s) ** 2)

    theta = jnp.array([1.6, 0.14])
    g = jax.jacfwd(loss)(theta)
    h = 1e-6
    fd = jnp.array([(loss(theta + h * e) - loss(theta - h * e)) / (2 * h) for e in jnp.eye(2)])
    np.testing.assert_allclose(g, fd, rtol=1e-5)


@pytest.mark.parametrize("seed", [0, 1, 2])
def test_fit_recovers_capillary_length(seed):
    a2_true, b_true = 7.35, 1.6
    pts, _ = bfx.synthetic_drop(b=b_true, capillary_length_sq=a2_true, seed=seed)
    r = bfx.fit_drop(pts)
    assert abs(r.capillary_length_sq - a2_true) < 4 * r.capillary_length_sq_std
    assert abs(float(r.params.b) - b_true) < 4 * r.b_std
    assert r.rms_residual < 0.01  # noise was 0.005
    # γ for water from a² in mm² should be ~72 mN/m.
    assert 0.065 < r.surface_tension(998.0) < 0.079


def test_noise_free_fit_is_essentially_exact():
    pts, _ = bfx.synthetic_drop(noise=0.0, n_points=80, seed=3)
    r = bfx.fit_drop(pts)
    assert abs(r.capillary_length_sq - 7.35) < 1e-3
    assert abs(float(r.params.b) - 1.6) < 1e-5


@pytest.mark.parametrize(
    "bad, msg",
    [
        (np.zeros((20, 3)), "shape"),
        (np.zeros((5, 2)), "at least 10"),
        (np.full((20, 2), np.nan), "NaN"),
    ],
)
def test_input_validation(bad, msg):
    with pytest.raises(ValueError, match=msg):
        bfx.fit_drop(bad)
