"""
CrossBasis transformer for DLNM package.

Implements the sklearn-style transformer that computes the cross-basis matrix W
(Eq. 5 and Eq. 8 from Gasparrini et al. 2010).
"""

import warnings
import numpy as np
import pandas as pd

from .basis import build_basis
from .utils import to_array, build_lag_matrix, logknots, equalknots


class CrossBasis:
    """
    Distributed Lag cross-basis transformer.

    Computes the cross-basis matrix W for use in regression models. This is an
    sklearn-style transformer suitable for power users; most users should use
    CrossPred instead.

    Parameters
    ----------
    var_basis : str or callable, default "ns"
        Basis function for exposure dimension ("ns", "bs", "poly", "linear", or callable)
    var_df : int, optional
        Degrees of freedom for exposure basis
    var_knots : np.ndarray, optional
        Interior knots for exposure basis
    var_kwargs : dict, optional
        Additional arguments for exposure basis function
    lag_basis : str or callable, default "ns"
        Basis function for lag dimension
    lag_df : int, optional
        Degrees of freedom for lag basis
    lag_knots : np.ndarray, optional
        Interior knots for lag basis
    lag_kwargs : dict, optional
        Additional arguments for lag basis function
    max_lag : int, required
        Maximum lag
    lag_knot_scale : str, default "log"
        Scale for automatic lag knot placement: "log" or "equal"
    na_action : str or float, default "drop"
        Strategy for NaN rows from lag padding: "drop", "fill_zero", "fill_mean", or numeric value
    """

    def __init__(
        self,
        var_basis="ns",
        var_df=None,
        var_knots=None,
        var_kwargs=None,
        lag_basis="ns",
        lag_df=None,
        lag_knots=None,
        lag_kwargs=None,
        max_lag=None,
        lag_knot_scale="log",
        na_action="drop",
    ):
        if max_lag is None:
            raise ValueError("max_lag is required")

        if var_knots is not None and var_df is not None:
            raise ValueError("var_knots and var_df are mutually exclusive")

        if lag_knots is not None and lag_df is not None:
            raise ValueError("lag_knots and lag_df are mutually exclusive")

        self.var_basis = var_basis
        self.var_df = var_df
        self.var_knots = var_knots
        self.var_kwargs = var_kwargs or {}
        self.lag_basis = lag_basis
        self.lag_df = lag_df
        self.lag_knots = lag_knots
        self.lag_kwargs = lag_kwargs or {}
        self.max_lag = max_lag
        self.lag_knot_scale = lag_knot_scale
        self.na_action = na_action

        self.is_fitted_ = False

    @staticmethod
    def _n_interior(basis, df, kwargs):
        """Compute number of interior knots needed to achieve `df` output columns."""
        intercept = kwargs.get('intercept', False)
        if basis == "ns":
            # ns: null-space projection gives df = n_interior + 4 - 2 - int(not intercept)
            return max(0, df - 1 - int(intercept))
        else:
            # bs and others: df = n_interior + degree + int(intercept), degree defaults to 3
            degree = kwargs.get('degree', 3)
            return max(0, df - degree - int(not intercept))

    def fit(self, X):
        """
        Fit the cross-basis to training data.

        Computes knots and boundary knots for basis functions.

        Parameters
        ----------
        X : np.ndarray or pd.Series or pd.DataFrame
            Training exposure data

        Returns
        -------
        self
        """
        X = to_array(X)

        # Compute variable knots if needed
        if self.var_knots is None and self.var_df is not None:
            n_interior = self._n_interior(self.var_basis, self.var_df, self.var_kwargs)
            if n_interior > 0:
                quantiles = np.linspace(0, 1, n_interior + 2)[1:-1]
                self.var_knots_ = np.nanquantile(X, quantiles)
            else:
                self.var_knots_ = np.array([], dtype=np.float64)
            self.var_df_ = self.var_df
        else:
            self.var_knots_ = self.var_knots.copy() if self.var_knots is not None else None
            self.var_df_ = self.var_df

        # Compute lag knots if needed
        if self.lag_knots is None and self.lag_df is not None:
            n_interior = self._n_interior(self.lag_basis, self.lag_df, self.lag_kwargs)
            if self.lag_knot_scale == "log":
                if n_interior > 0:
                    self.lag_knots_ = logknots(self.max_lag, n_interior)
                else:
                    self.lag_knots_ = np.array([], dtype=np.float64)
            elif self.lag_knot_scale == "equal":
                lag_vec = np.arange(self.max_lag + 1)
                if n_interior > 0:
                    self.lag_knots_ = equalknots(lag_vec, n_interior)
                else:
                    self.lag_knots_ = np.array([], dtype=np.float64)
            else:
                raise ValueError(f"lag_knot_scale must be 'log' or 'equal', got {self.lag_knot_scale}")
            self.lag_df_ = self.lag_df
        else:
            self.lag_knots_ = self.lag_knots.copy() if self.lag_knots is not None else None
            self.lag_df_ = self.lag_df

        # Store boundary knots for extrapolation
        self.boundary_knots_ = np.array([np.nanmin(X), np.nanmax(X)], dtype=np.float64)

        self.is_fitted_ = True
        return self

    def transform(self, X):
        """
        Transform exposure data to cross-basis matrix W.

        Parameters
        ----------
        X : np.ndarray or pd.Series or pd.DataFrame
            Exposure data

        Returns
        -------
        np.ndarray or pd.DataFrame
            Cross-basis matrix W, shape (n, v_x * v_l)
        """
        if not self.is_fitted_:
            raise ValueError("CrossBasis must be fitted before transform")

        X_arr = to_array(X)
        is_dataframe = isinstance(X, (pd.DataFrame, pd.Series))

        # Build lag matrix Q
        Q = build_lag_matrix(X_arr, self.max_lag)

        # Handle NaN rows from lag padding
        nan_mask = np.isnan(Q).any(axis=1)
        nan_indices = np.where(nan_mask)[0]

        if self.na_action == "drop":
            if nan_indices.size > 0:
                warnings.warn(
                    f"CrossBasis.transform: {len(nan_indices)} rows with NaNs from lag padding "
                    f"(first {min(5, len(nan_indices))} row indices: {nan_indices[:5]}). "
                    f"These rows will produce NaN in W.",
                    UserWarning,
                )
        elif self.na_action == "fill_zero":
            Q = np.nan_to_num(Q, nan=0.0)
        elif self.na_action == "fill_mean":
            col_means = np.nanmean(Q, axis=0)
            for i in range(Q.shape[1]):
                Q[np.isnan(Q[:, i]), i] = col_means[i]
        elif isinstance(self.na_action, (int, float)):
            Q = np.nan_to_num(Q, nan=float(self.na_action))
        else:
            raise ValueError(f"Invalid na_action: {self.na_action}")

        # Build exposure basis Z
        z_kwargs = dict(boundary_knots=self.boundary_knots_, **self.var_kwargs)
        if self.var_knots_ is not None:
            z_kwargs['knots'] = self.var_knots_
        Z = build_basis(X_arr, self.var_basis, self.var_df_ or 3, **z_kwargs)

        # Build lag basis C
        lag_vector = np.arange(self.max_lag + 1, dtype=np.float64)
        c_kwargs = dict(**self.lag_kwargs)
        if self.lag_knots_ is not None:
            c_kwargs['knots'] = self.lag_knots_
        C = build_basis(lag_vector, self.lag_basis, self.lag_df_ or 3, **c_kwargs)

        # Store for later use in CrossPred
        self.C_ = C
        self.Z_shape_ = Z.shape[1]  # Store v_x

        # Compute W: row-wise Kronecker product of Z and (Q @ C)
        QC = Q @ C
        n = Z.shape[0]
        v_x = Z.shape[1]
        v_l = C.shape[1]
        W = np.zeros((n, v_x * v_l), dtype=np.float64)

        for t in range(n):
            W[t, :] = np.kron(Z[t, :], QC[t, :])

        # Apply NaN to dropped rows if na_action="drop"
        if self.na_action == "drop" and nan_indices.size > 0:
            W[nan_indices, :] = np.nan

        # Convert to DataFrame if input was
        if is_dataframe:
            col_names = [f"cb_{j}_{k}" for j in range(v_x) for k in range(v_l)]
            W = pd.DataFrame(W, columns=col_names, index=X.index)

        return W

    def fit_transform(self, X):
        """
        Fit and transform in one step.

        Parameters
        ----------
        X : np.ndarray or pd.Series or pd.DataFrame
            Training exposure data

        Returns
        -------
        np.ndarray or pd.DataFrame
            Cross-basis matrix W
        """
        return self.fit(X).transform(X)
