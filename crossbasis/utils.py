"""
Utility functions for DLNM package.
"""

import numpy as np
import pandas as pd


def to_array(x) -> np.ndarray:
    """
    Coerce input to 1-D numpy array of float64.

    Accepts numpy arrays, pandas Series, or pandas DataFrame columns.

    Parameters
    ----------
    x : np.ndarray, pd.Series, or pd.DataFrame
        Input data

    Returns
    -------
    np.ndarray
        1-D array of dtype float64
    """
    if isinstance(x, np.ndarray):
        return x.astype(np.float64)
    elif isinstance(x, pd.Series):
        return x.values.astype(np.float64)
    elif isinstance(x, pd.DataFrame):
        if x.shape[1] != 1:
            raise ValueError(f"DataFrame must have exactly 1 column, got {x.shape[1]}")
        return x.iloc[:, 0].values.astype(np.float64)
    else:
        return np.asarray(x, dtype=np.float64)


def logknots(max_lag: int, nk: int) -> np.ndarray:
    """
    Place nk interior knots on a log scale over [0, max_lag].

    Matches R's dlnm::logknots() function. Recommended default for lag dimension.

    Parameters
    ----------
    max_lag : int
        Maximum lag value
    nk : int
        Number of interior knots

    Returns
    -------
    np.ndarray
        Array of knot positions, sorted
    """
    if nk == 0:
        return np.array([])
    # Log scale from 1 to max_lag+1, place nk quantiles
    log_scale = np.logspace(0, np.log10(max_lag + 1), nk + 2)[1:-1]
    # Subtract 1 to map back to [0, max_lag]
    knots = log_scale - 1
    return np.sort(knots)


def equalknots(x: np.ndarray, nk: int) -> np.ndarray:
    """
    Place nk interior knots at equally-spaced quantiles of x.

    Parameters
    ----------
    x : np.ndarray
        Data array (may contain NaN)
    nk : int
        Number of interior knots

    Returns
    -------
    np.ndarray
        Array of knot positions at quantiles, sorted
    """
    if nk == 0:
        return np.array([])
    # Compute quantiles, ignoring NaN
    quantiles = np.linspace(0, 1, nk + 2)[1:-1]
    knots = np.nanquantile(x, quantiles)
    return np.sort(knots)


def build_lag_matrix(x: np.ndarray, max_lag: int) -> np.ndarray:
    """
    Construct the n × (max_lag + 1) lag matrix Q (Eq. 3).

    Q[t, l] = x[t - l], with np.nan where t - l < 0.

    Parameters
    ----------
    x : np.ndarray
        Exposure vector, shape (n,)
    max_lag : int
        Maximum lag

    Returns
    -------
    np.ndarray
        Lag matrix of shape (n, max_lag + 1), dtype float64
    """
    x = to_array(x)
    n = len(x)
    Q = np.full((n, max_lag + 1), np.nan, dtype=np.float64)

    for l in range(max_lag + 1):
        if l == 0:
            Q[:, l] = x
        else:
            Q[l:, l] = x[:-l]

    return Q
