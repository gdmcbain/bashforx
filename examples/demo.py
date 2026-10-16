"""Fit a synthetic water drop and plot data, initial guess and fitted Young–Laplace profile."""

from pathlib import Path

import jax.numpy as jnp
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

import bashforx as bfx  # noqa: E402

A2_TRUE, B_TRUE = 7.35, 1.6  # mm², mm  (water in air)

points, _ = bfx.synthetic_drop(b=B_TRUE, capillary_length_sq=A2_TRUE, noise=0.005, seed=0)
fit = bfx.fit_drop(points)
p, p0 = fit.params, fit.initial

print(f"apex radius b      = {float(p.b):.4f} ± {fit.b_std:.4f} mm   (true {B_TRUE})")
print(f"a² = γ/(Δρ g)      = {fit.capillary_length_sq:.3f} ± {fit.capillary_length_sq_std:.3f} mm²  (true {A2_TRUE})")
print(f"β = b²/a²          = {fit.beta:.4f}")
print(f"γ (Δρ = 998 kg/m³) = {1e3 * fit.surface_tension(998.0):.2f} mN/m")
print(f"rms residual       = {fit.rms_residual * 1e3:.2f} µm, LM steps = {fit.num_steps}")


def curve(params):
    smax = float(jnp.max(jnp.abs(params.s)))
    s = jnp.linspace(-smax, smax, 400)
    return bfx.profile_points(params.b, params.c, params.x0, params.z0, s)


fig, ax = plt.subplots(figsize=(5, 6))
ax.plot(points[:, 0], points[:, 1], ".", ms=3, color="0.4", label="edge points (noisy)")
c0 = curve(p0)
ax.plot(c0[:, 0], c0[:, 1], "--", color="tab:orange", label=f"initial guess (β-table), a²={1 / float(p0.c):.2f}")
c1 = curve(p)
ax.plot(c1[:, 0], c1[:, 1], "-", color="tab:blue", label=f"LM fit, a²={fit.capillary_length_sq:.3f} mm²")
ax.set_aspect("equal")
# z points up towards the needle, so the drop already hangs the way it looks in a photograph.
ax.set_xlabel("x / mm")
ax.set_ylabel("z / mm (needle at top)")
ax.set_title("Pendant drop: Young–Laplace fit (Diffrax + Optimistix)")
ax.legend(loc="upper center", fontsize=8)
fig.tight_layout()
out = Path(__file__).with_name("pendant_drop_fit.png")
fig.savefig(out, dpi=130)
print(f"saved {out}")
