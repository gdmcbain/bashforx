"""Pendant-drop tensiometry à la Bashforth & Adams (1883) with Diffrax + Optimistix.

The axisymmetric Young–Laplace equation, in the arc-length form Bashforth & Adams
used, for a drop hanging from a needle with its apex at the origin and ``z``
pointing *up* (towards the needle):

    dx/ds   = cos φ
    dz/ds   = sin φ
    dφ/ds   = 2/b − c·z − sin φ / x

* ``s``  arc length measured from the apex,
* ``φ``  angle of the tangent to the horizontal,
* ``b``  radius of curvature at the apex,
* ``c = Δρ g / γ = 1/a²`` the capillary constant (``a`` = capillary length).

What the experiment really gives you is ``a² = γ / (Δρ g)``: surface tension
relative to specific gravity, which is exactly what Bashforth & Adams tabulated.
Their dimensionless shape parameter is ``β = c b²`` (the Bond number of the apex).

Fitting strategy (this is what modern ADSA does too):

* Unknowns are ``(b, c, x0, z0)`` plus one *signed* arc length ``s_i`` per observed
  edge point. Points left of the axis get ``s_i < 0``, using the symmetry
  ``x(−s) = −x(s)`` and ``z(−s) = z(s)``.
* The residual for each point is ``model(s_i) − observed_i`` (2 numbers per point).
  That is a smooth least-squares problem, so Levenberg–Marquardt applies directly.
  Taking the min-distance to the curve instead would make it non-smooth.
* The ODE is solved once per residual evaluation with dense output and then
  interpolated at ``|s_i|``. ``diffrax.ForwardMode`` makes forward-mode
  autodiff (``jac="fwd"``, Optimistix's default for least squares) go through the
  solver.
"""

from __future__ import annotations

from typing import NamedTuple

import jax

jax.config.update("jax_enable_x64", True)

import diffrax  # noqa: E402
import jax.numpy as jnp  # noqa: E402
import numpy as np  # noqa: E402
import optimistix as optx  # noqa: E402
from jax.flatten_util import ravel_pytree  # noqa: E402

# Below this |x| we take the analytic apex limit sin φ / x → 1/b instead of dividing.
_APEX_EPS = 1e-12


def young_laplace(s, y, args):
    """Right-hand side of the Bashforth–Adams system. ``y = (x, z, φ)``, ``args = (b, c)``."""
    del s
    x, z, phi = y
    b, c = args
    at_apex = jnp.abs(x) < _APEX_EPS
    # Double-where so that neither the value nor its derivative becomes NaN at x = 0.
    safe_x = jnp.where(at_apex, 1.0, x)
    azimuthal_curvature = jnp.where(at_apex, 1.0 / b, jnp.sin(phi) / safe_x)
    return jnp.stack([jnp.cos(phi), jnp.sin(phi), 2.0 / b - c * z - azimuthal_curvature])


def solve_profile(b, c, s_max, *, saveat=None, rtol=1e-8, atol=1e-10, max_steps=4096):
    """Integrate the profile from the apex (s = 0) to ``s_max``.

    Returns the ``diffrax.Solution``. With the default ``saveat`` it carries dense
    output, so ``sol.evaluate(s)`` works for any ``0 ≤ s ≤ s_max``.
    """
    if saveat is None:
        saveat = diffrax.SaveAt(dense=True)
    return diffrax.diffeqsolve(
        diffrax.ODETerm(young_laplace),
        diffrax.Tsit5(),
        t0=0.0,
        t1=s_max,
        dt0=None,
        y0=jnp.zeros(3),
        args=(b, c),
        saveat=saveat,
        stepsize_controller=diffrax.PIDController(rtol=rtol, atol=atol),
        adjoint=diffrax.ForwardMode(),
        max_steps=max_steps,
    )


def profile_points(b, c, x0, z0, s):
    """Model edge points (N, 2) at signed arc lengths ``s`` (negative = left branch)."""
    s_abs = jnp.abs(s)
    sol = solve_profile(b, c, jnp.max(s_abs))
    xz = jax.vmap(sol.evaluate)(s_abs)[:, :2]
    return jnp.stack([x0 + jnp.sign(s) * xz[:, 0], z0 + xz[:, 1]], axis=1)


# --------------------------------------------------------------------------- fitting


class DropParams(NamedTuple):
    b: jax.Array  # apex radius of curvature
    c: jax.Array  # capillary constant Δρ g / γ
    x0: jax.Array  # apex position (same units as the data)
    z0: jax.Array
    s: jax.Array  # signed arc length of each observed point, shape (N,)


class FitResult(NamedTuple):
    params: DropParams
    capillary_length_sq: float  # a² = γ / (Δρ g)
    capillary_length_sq_std: float  # 1σ from the linearised covariance
    b_std: float
    beta: float  # Bashforth–Adams shape factor c·b²
    rms_residual: float
    num_steps: int
    initial: DropParams

    def surface_tension(self, delta_rho, g=9.80665, length_unit=1e-3):
        """γ in N/m given Δρ (kg/m³). ``length_unit`` = metres per data unit (default mm)."""
        return delta_rho * g * self.capillary_length_sq * length_unit**2


def _residuals(params: DropParams, points):
    return (profile_points(params.b, params.c, params.x0, params.z0, params.s) - points).ravel()


def _validate(points) -> jnp.ndarray:
    points = jnp.asarray(points, dtype=jnp.float64)
    if points.ndim != 2 or points.shape[1] != 2:
        raise ValueError(f"points must have shape (N, 2) as (x, z); got {points.shape}")
    if points.shape[0] < 10:
        raise ValueError(f"need at least 10 edge points to fit a drop; got {points.shape[0]}")
    if not bool(jnp.all(jnp.isfinite(points))):
        raise ValueError("points contain NaN or inf")
    return points


def initial_guess(points, beta_grid=None) -> DropParams:
    """Geometric initial guess, then a 'table lookup' over β like Bashforth & Adams.

    * Apex: ``x0`` is the midpoint of the horizontal extent and ``z0`` is the lowest point.
    * ``b``: a parabola ``z − z0 ≈ (x − x0)² / (2b)`` fitted to points near the apex.
    * ``β``: the grid value whose curve best matches the data in nearest-point
      distance. Bashforth & Adams did the same comparison against printed tables.
    * ``s_i``: arc length of the closest point on that curve.
    """
    points = _validate(points)
    if beta_grid is None:
        beta_grid = jnp.linspace(0.02, 0.8, 40)
    x, z = points[:, 0], points[:, 1]
    x_mid = 0.5 * (jnp.min(x) + jnp.max(x))
    z_min = jnp.min(z)
    half_width = 0.5 * (jnp.max(x) - jnp.min(x))

    # Points on the bottom cap: near the axis and below the equator. Fit
    # z ≈ A + B·u + C·u² (u = x − x_mid) with a free offset, so the apex height is
    # not pinned to the lowest sample. Sparse data never has a sample at the apex.
    u = x - x_mid
    cap = (jnp.abs(u) < 0.5 * half_width) & (z - z_min < 0.5 * half_width)
    w = cap.astype(points.dtype)
    design = jnp.stack([jnp.ones_like(u), u, u**2], axis=1) * w[:, None]
    (A, B, C), *_ = jnp.linalg.lstsq(design, z * w)
    good = (C > 0) & (jnp.sum(w) >= 5)
    x0 = jnp.where(good, x_mid - B / (2 * C), x_mid)
    z0 = jnp.where(good, jnp.minimum(A - B**2 / (4 * C), z_min), z_min)
    b = jnp.where(good, 0.5 / C, half_width)
    dx, dz = x - x0, z - z0

    # Arc-length budget: the drop's height plus its half-width, generously padded.
    s_max = 1.5 * (jnp.max(dz) + half_width) + b
    s_dense = jnp.linspace(0.0, s_max, 600)

    def curve(beta):
        sol = solve_profile(b, beta / b**2, s_max, saveat=diffrax.SaveAt(ts=s_dense), rtol=1e-6, atol=1e-8)
        return sol.ys[:, :2]

    def misfit(beta):
        xz = curve(beta)
        d2 = (jnp.abs(dx)[:, None] - xz[None, :, 0]) ** 2 + (dz[:, None] - xz[None, :, 1]) ** 2
        return jnp.sum(jnp.min(d2, axis=1))

    beta_grid = jnp.asarray(beta_grid, dtype=jnp.float64)
    best_beta = beta_grid[jnp.argmin(jax.vmap(misfit)(beta_grid))]
    xz = curve(best_beta)
    d2 = (jnp.abs(dx)[:, None] - xz[None, :, 0]) ** 2 + (dz[:, None] - xz[None, :, 1]) ** 2
    s_init = s_dense[jnp.argmin(d2, axis=1)]
    side = jnp.where(dx < 0, -1.0, 1.0)
    return DropParams(b=b, c=best_beta / b**2, x0=x0, z0=z0, s=side * s_init)


def fit_drop(points, *, rtol=1e-6, atol=1e-6, max_steps=200, init: DropParams | None = None) -> FitResult:
    """Fit a Young–Laplace profile to edge points ``(x, z)`` with z pointing up from the apex.

    ``rtol``/``atol`` are Levenberg–Marquardt's step tolerances. Keep them well above
    the ODE solver's tolerance (1e-8): asking for more precision than the integrator
    provides makes the stopping test unreachable, and LM then hits ``max_steps``
    even though it has already converged.

    Raises ``RuntimeError`` if Levenberg–Marquardt does not converge.
    """
    points = _validate(points)
    init = initial_guess(points) if init is None else init

    solver = optx.LevenbergMarquardt(rtol=rtol, atol=atol)
    sol = optx.least_squares(
        _residuals, solver, init, args=points, max_steps=max_steps, throw=False
    )
    if sol.result != optx.RESULTS.successful:
        raise RuntimeError(f"pendant-drop fit did not converge: {sol.result}")
    p = sol.value
    if not (float(p.b) > 0 and float(p.c) > 0):
        raise RuntimeError(
            f"fit converged to an unphysical shape (b={float(p.b):.4g}, c={float(p.c):.4g}); "
            "check that z points up from the apex towards the needle, or pass a better `init`"
        )

    # Linearised uncertainty: Cov = σ² (JᵀJ)⁻¹ with σ² estimated from the residuals.
    r = _residuals(p, points)
    flat, unravel = ravel_pytree(p)
    jac = jax.jacfwd(lambda v: _residuals(unravel(v), points))(flat)
    dof = r.size - flat.size
    sigma2 = jnp.sum(r**2) / dof
    cov = sigma2 * jnp.linalg.pinv(jac.T @ jac)
    var_b, var_c = cov[0, 0], cov[1, 1]  # ravel order follows DropParams fields: b, c, ...

    a2 = 1.0 / p.c
    return FitResult(
        params=p,
        capillary_length_sq=float(a2),
        capillary_length_sq_std=float(jnp.sqrt(var_c) / p.c**2),  # |d(1/c)/dc| = 1/c²
        b_std=float(jnp.sqrt(var_b)),
        beta=float(p.c * p.b**2),
        rms_residual=float(jnp.sqrt(jnp.mean(r**2))),
        num_steps=int(sol.stats["num_steps"]),
        initial=init,
    )


# --------------------------------------------------------------------- synthetic data


def synthetic_drop(
    b=1.6, capillary_length_sq=7.35, x0=3.0, z0=1.0, needle_radius=0.9, n_points=150, noise=0.005, seed=0
):
    """Noisy edge points of a pendant drop cut off where it meets the needle.

    Defaults are roughly water in air, in mm: a² = γ/(Δρg) ≈ 72e-3/(998·9.81) m².
    Returns ``(points, s_true)``.
    """
    c = 1.0 / capillary_length_sq
    s_grid = np.linspace(0.0, 4.0 * b + 6.0, 4000)
    ys = np.asarray(solve_profile(b, c, s_grid[-1], saveat=diffrax.SaveAt(ts=jnp.asarray(s_grid))).ys)
    x, phi = ys[:, 0], ys[:, 2]
    # Past the equator (φ > π/2) the profile narrows; stop at the needle radius.
    past_equator = phi > np.pi / 2
    neck = np.nonzero(past_equator & (x <= needle_radius))[0]
    if neck.size == 0:
        raise ValueError("profile never narrows to the needle radius; pick a larger needle or different β")
    s_end = s_grid[neck[0]]

    rng = np.random.default_rng(seed)
    s_true = np.sort(rng.uniform(-s_end, s_end, n_points))
    clean = np.asarray(profile_points(b, c, x0, z0, jnp.asarray(s_true)))
    return clean + noise * rng.standard_normal(clean.shape), s_true
