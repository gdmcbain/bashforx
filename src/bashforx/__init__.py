"""bashforx: pendant-drop tensiometry à la Bashforth & Adams (1883) with Diffrax + Optimistix.

Fit the axisymmetric Young–Laplace profile to the edge points of a pendant drop
to infer ``a² = γ / (Δρ g)``, the surface tension relative to specific gravity.

Importing bashforx enables JAX's 64-bit mode (``jax_enable_x64``). The fit needs
double precision to resolve the arc-length parameters.
"""

from importlib.metadata import PackageNotFoundError, version

from ._core import (
    DropParams,
    FitResult,
    fit_drop,
    initial_guess,
    profile_points,
    solve_profile,
    synthetic_drop,
    young_laplace,
)

try:
    __version__ = version("bashforx")
except PackageNotFoundError:  # running from a source tree that is not installed
    __version__ = "0.0.0"

__all__ = [
    "DropParams",
    "FitResult",
    "fit_drop",
    "initial_guess",
    "profile_points",
    "solve_profile",
    "synthetic_drop",
    "young_laplace",
]
