"""
Basis function builders for DLNM package.

Implements natural cubic splines (ns), B-splines (bs), orthogonal polynomials (poly),
and linear basis, matching R's splines package where applicable.
"""

import numpy as np
from scipy.interpolate import BSpline
from numpy.polynomial.polynomial import polyvander

from .utils import to_array


def _validate_basis_output(B: np.ndarray, m: int, df: int, name: str) -> np.ndarray:
    """
    Validate basis function output.

    Parameters
    ----------
    B : np.ndarray
        Output from basis function
    m : int
        Expected number of rows (observations)
    df : int
        Expected number of columns (basis functions)
    name : str
        Name of basis for error messages

    Returns
    -------
    np.ndarray
        Validated basis matrix, converted to float64 if needed

    Raises
    ------
    ValueError
        If shape is wrong or contains non-finite values
    """
    if B.shape != (m, df):
        raise ValueError(f"Basis '{name}' returned shape {B.shape}, expected ({m}, {df})")
    if not np.issubdtype(B.dtype, np.floating):
        B = B.astype(np.float64)
    if not np.isfinite(B).all():
        raise ValueError(f"Basis '{name}' returned non-finite values")
    return B


def ns_basis(
    x: np.ndarray,
    df: int = None,
    knots: np.ndarray = None,
    boundary_knots: np.ndarray = None,
    intercept: bool = False,
    **kwargs,
) -> np.ndarray:
    """
    Natural cubic spline basis, matching R's splines::ns().

    Parameters
    ----------
    x : np.ndarray
        Input vector
    df : int, optional
        Degrees of freedom. Ignored if knots provided.
    knots : np.ndarray, optional
        Interior knot positions. If provided, df is ignored.
    boundary_knots : np.ndarray, optional
        [lower, upper] boundary knots
    intercept : bool, default False
        If False, removes the first basis function
    **kwargs
        Additional arguments (unused)

    Returns
    -------
    np.ndarray
        Natural cubic spline basis matrix
    """
    x = to_array(x)
    m = len(x)

    if boundary_knots is None:
        boundary_knots = np.array([np.nanmin(x), np.nanmax(x)], dtype=np.float64)
    else:
        boundary_knots = np.asarray(boundary_knots, dtype=np.float64)

    if knots is not None:
        interior_knots = np.asarray(knots, dtype=np.float64)
    elif df is not None:
        # n_basis_full = n_interior + 4; constraints = 2 (natural) + int(not intercept)
        # free columns = n_interior + 4 - 2 - int(not intercept) = df
        # => n_interior = df - 2 + int(not intercept) = df - 1 - int(intercept)
        n_interior = max(0, df - 1 - int(intercept))
        if n_interior > 0:
            quantiles = np.linspace(0, 1, n_interior + 2)[1:-1]
            # Use only x values within the boundary to guarantee valid knot ordering
            x_inner = x[(x >= boundary_knots[0]) & (x <= boundary_knots[1])]
            if len(x_inner) >= 2:
                interior_knots = np.nanquantile(x_inner, quantiles)
            else:
                interior_knots = np.linspace(boundary_knots[0], boundary_knots[1], n_interior + 2)[1:-1]
        else:
            interior_knots = np.array([], dtype=np.float64)
    else:
        raise ValueError("Either knots or df must be supplied")

    knot_vector = np.concatenate([
        np.repeat(boundary_knots[0], 4),
        interior_knots,
        np.repeat(boundary_knots[1], 4),
    ])
    n_basis_full = len(knot_vector) - 4

    # Build individual B-splines for basis evaluation and constraint computation
    splines = []
    for i in range(n_basis_full):
        c = np.zeros(n_basis_full)
        c[i] = 1.0
        splines.append(BSpline(knot_vector, c, k=3, extrapolate=False))

    # Evaluate full B-spline basis with linear extrapolation outside boundaries
    basis_full = np.zeros((m, n_basis_full), dtype=np.float64)
    for i, spl in enumerate(splines):
        interior_mask = (x >= boundary_knots[0]) & (x <= boundary_knots[1])
        basis_full[interior_mask, i] = spl(x[interior_mask])

        left_mask = x < boundary_knots[0]
        if left_mask.any():
            val_l = float(spl(boundary_knots[0]))
            slope_l = float(spl(boundary_knots[0], 1))
            basis_full[left_mask, i] = val_l + slope_l * (x[left_mask] - boundary_knots[0])

        right_mask = x > boundary_knots[1]
        if right_mask.any():
            val_r = float(spl(boundary_knots[1]))
            slope_r = float(spl(boundary_knots[1], 1))
            basis_full[right_mask, i] = val_r + slope_r * (x[right_mask] - boundary_knots[1])

    # Natural spline constraints: zero second derivative at both boundary knots
    const_rows = [
        np.array([float(spl(bk, 2)) for spl in splines])
        for bk in boundary_knots
    ]
    # intercept=False: exclude B-spline 0 via unit-vector constraint (R's convention)
    if not intercept:
        e0 = np.zeros(n_basis_full)
        e0[0] = 1.0
        const_rows.append(e0)

    const = np.vstack(const_rows)  # (n_constraints, n_basis_full)

    # Null space of const via QR decomposition
    n_constraints = const.shape[0]
    Q, _ = np.linalg.qr(const.T, mode='complete')
    null_space = Q[:, n_constraints:]  # (n_basis_full, df_actual)

    basis = basis_full @ null_space

    expected_df = df if df is not None else null_space.shape[1]
    return _validate_basis_output(basis, m, expected_df, "ns")


def bs_basis(
    x: np.ndarray,
    df: int = None,
    knots: np.ndarray = None,
    degree: int = 3,
    boundary_knots: np.ndarray = None,
    intercept: bool = False,
    **kwargs,
) -> np.ndarray:
    """
    B-spline basis.

    Parameters
    ----------
    x : np.ndarray
        Input vector
    df : int, optional
        Degrees of freedom. Ignored if knots provided.
    knots : np.ndarray, optional
        Interior knot positions. If provided, df is ignored.
    degree : int, default 3
        Spline degree
    boundary_knots : np.ndarray, optional
        [lower, upper] boundary knots
    intercept : bool, default False
        If False, remove first basis function
    **kwargs
        Additional arguments (unused)

    Returns
    -------
    np.ndarray
        B-spline basis matrix
    """
    x = to_array(x)
    m = len(x)

    if boundary_knots is None:
        boundary_knots = np.array([np.nanmin(x), np.nanmax(x)], dtype=np.float64)

    # Determine interior knots
    if knots is not None:
        interior_knots = np.asarray(knots, dtype=np.float64)
        expected_df = len(knots) + (degree + 1) - 1
    elif df is not None:
        n_interior = max(0, df - degree)
        if n_interior > 0:
            quantiles = np.linspace(0, 1, n_interior + 2)[1:-1]
            interior_knots = np.nanquantile(x, quantiles)
        else:
            interior_knots = np.array([], dtype=np.float64)
        expected_df = df
    else:
        interior_knots = np.array([], dtype=np.float64)
        expected_df = degree + 1

    # Build knot vector
    knot_vector = np.concatenate([
        np.repeat(boundary_knots[0], degree + 1),
        interior_knots,
        np.repeat(boundary_knots[1], degree + 1),
    ])

    n_basis_full = len(knot_vector) - degree - 1
    basis = np.zeros((m, n_basis_full), dtype=np.float64)

    for i in range(n_basis_full):
        c = np.zeros(n_basis_full)
        c[i] = 1.0
        try:
            spl = BSpline(knot_vector, c, k=degree, extrapolate=True)
            basis[:, i] = spl(x)
        except (ValueError, RuntimeError):
            pass

    if not intercept and n_basis_full > 1:
        basis = basis[:, 1:]

    return _validate_basis_output(basis, m, expected_df, "bs")


def poly_basis(x: np.ndarray, df: int, raw: bool = False, **kwargs) -> np.ndarray:
    """
    Orthogonal polynomial basis.

    Parameters
    ----------
    x : np.ndarray
        Input vector
    df : int
        Degree of polynomial (number of basis functions)
    raw : bool, default False
        If False, use orthogonal polynomials
    **kwargs
        Additional arguments (unused)

    Returns
    -------
    np.ndarray
        Polynomial basis matrix, shape (len(x), df)
    """
    x = to_array(x)
    m = len(x)
    basis = polyvander(x, df - 1)

    if not raw:
        Q, R = np.linalg.qr(basis)
        basis = Q[:, :df]
        for j in range(df):
            col_norm = np.linalg.norm(basis[:, j])
            if col_norm > 1e-10:
                basis[:, j] /= col_norm

    return _validate_basis_output(basis, m, df, "poly")


def linear_basis(x: np.ndarray, df: int = None, **kwargs) -> np.ndarray:
    """
    Linear basis (single column of x).

    Parameters
    ----------
    x : np.ndarray
        Input vector
    df : int, optional
        Ignored (always returns 1 column)
    **kwargs
        Additional arguments (unused)

    Returns
    -------
    np.ndarray
        Linear basis matrix, shape (len(x), 1)
    """
    x = to_array(x)
    m = len(x)
    basis = x.reshape(-1, 1)
    return _validate_basis_output(basis, m, 1, "linear")


_BUILTIN_BASES = {
    "ns": ns_basis,
    "bs": bs_basis,
    "poly": poly_basis,
    "linear": linear_basis,
}


def build_basis(x: np.ndarray, basis, df: int, **kwargs) -> np.ndarray:
    """
    Dispatch to built-in or user-supplied basis function.

    Parameters
    ----------
    x : np.ndarray
        Input vector
    basis : str or callable
        String key or callable basis function
    df : int
        Degrees of freedom
    **kwargs
        Additional arguments (knots, boundary_knots, etc.)

    Returns
    -------
    np.ndarray
        Basis matrix, shape (len(x), df)
    """
    if callable(basis):
        name = getattr(basis, "__name__", repr(basis))
        B = basis(x, df, **kwargs)
    elif isinstance(basis, str):
        if basis not in _BUILTIN_BASES:
            raise ValueError(
                f"Unknown basis '{basis}'. Choose from {list(_BUILTIN_BASES)} or pass a callable."
            )
        name = basis
        B = _BUILTIN_BASES[basis](x, df, **kwargs)
    else:
        raise TypeError(f"basis must be str or callable, got {type(basis)}")

    return _validate_basis_output(B, len(x), df, name)
