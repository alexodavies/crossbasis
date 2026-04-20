"""
CrossPred high-level class and PredictionResult for DLNM package.

This is the user-facing API for prediction and visualization.
"""

import warnings
from dataclasses import dataclass
from typing import Optional, Union

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D

from .crossbasis import CrossBasis
from .basis import build_basis
from .utils import to_array


@dataclass
class PredictionResult:
    """
    Result of CrossPred.predict().

    Contains predictions at each (exposure, lag) combination and aggregated
    cumulative effect, with standard errors and confidence intervals.

    Attributes
    ----------
    matfit : np.ndarray
        Log-RR at each (exposure, lag), shape (len(at), max_lag + 1), Eq. 9
    matse : np.ndarray
        SE of matfit, shape (len(at), max_lag + 1), Eq. 10
    allfit : np.ndarray
        Cumulative log-RR over all lags, shape (len(at),), Eq. 11
    allse : np.ndarray
        SE of allfit, shape (len(at),), Eq. 12
    RR : np.ndarray
        Exponentiated matfit, shape (len(at), max_lag + 1)
    RR_low : np.ndarray
        Lower 95% CI on RR
    RR_high : np.ndarray
        Upper 95% CI on RR
    allRR : np.ndarray
        Exponentiated allfit, shape (len(at),)
    allRR_low : np.ndarray
        Lower 95% CI on allRR
    allRR_high : np.ndarray
        Upper 95% CI on allRR
    predvar : np.ndarray
        Exposure values used for prediction
    predlag : np.ndarray
        Lag values [0, 1, ..., max_lag]
    cen : float
        Reference/centering value
    """

    matfit: np.ndarray
    matse: np.ndarray
    allfit: np.ndarray
    allse: np.ndarray
    RR: np.ndarray
    RR_low: np.ndarray
    RR_high: np.ndarray
    allRR: np.ndarray
    allRR_low: np.ndarray
    allRR_high: np.ndarray
    predvar: np.ndarray
    predlag: np.ndarray
    cen: float

    def to_frame(self) -> pd.DataFrame:
        """
        Convert to tidy long-format DataFrame.

        Returns
        -------
        pd.DataFrame
            Long-format DataFrame with columns: exposure, lag, logRR, se, RR, RR_low, RR_high
        """
        n_exp = len(self.predvar)
        n_lag = len(self.predlag)

        rows = []
        for i, exp in enumerate(self.predvar):
            for j, lag in enumerate(self.predlag):
                rows.append({
                    "exposure": exp,
                    "lag": lag,
                    "logRR": self.matfit[i, j],
                    "se": self.matse[i, j],
                    "RR": self.RR[i, j],
                    "RR_low": self.RR_low[i, j],
                    "RR_high": self.RR_high[i, j],
                })

        return pd.DataFrame(rows)

    def plot_3d(
        self,
        fig=None,
        ax=None,
        xlabel="Exposure",
        ylabel="Lag",
        zlabel="RR",
        title=None,
        cmap="viridis",
        **kwargs,
    ):
        """
        Plot 3D surface of RR over exposure and lag dimensions.

        Parameters
        ----------
        fig : matplotlib.figure.Figure, optional
            Existing figure; if None, creates new
        ax : mpl_toolkits.mplot3d.Axes3D, optional
            Existing 3D axes
        xlabel : str
            Label for exposure axis
        ylabel : str
            Label for lag axis
        zlabel : str
            Label for RR axis
        title : str, optional
            Plot title
        cmap : str, default "viridis"
            Colormap for surface
        **kwargs
            Additional arguments passed to plot_surface (e.g., alpha, edgecolor)

        Returns
        -------
        fig, ax
            Figure and axes objects
        """
        if fig is None:
            fig = plt.figure(figsize=(10, 8))
        if ax is None:
            ax = fig.add_subplot(111, projection="3d")

        # Create mesh grid
        X, Y = np.meshgrid(self.predvar, self.predlag)
        Z = self.RR.T  # Transpose to match X, Y shapes

        # Plot surface
        surf = ax.plot_surface(X, Y, Z, cmap=cmap, **kwargs)

        ax.set_xlabel(xlabel)
        ax.set_ylabel(ylabel)
        ax.set_zlabel(zlabel)
        if title:
            ax.set_title(title)

        fig.colorbar(surf, ax=ax, label=zlabel)

        return fig, ax

    def plot_slice(
        self,
        var: Optional[float] = None,
        lag: Optional[int] = None,
        ci: bool = True,
        reference_line: bool = True,
        fig=None,
        ax=None,
        xlabel=None,
        ylabel="RR",
        title=None,
        color="steelblue",
        ci_alpha=0.3,
        linewidth=2,
        **kwargs,
    ):
        """
        Plot lag-response or exposure-response slice.

        Parameters
        ----------
        var : float, optional
            Exposure value to fix (plot lag-response). Mutually exclusive with lag.
        lag : int, optional
            Lag value to fix (plot exposure-response). Mutually exclusive with var.
        ci : bool, default True
            Show 95% confidence interval as shaded band
        reference_line : bool, default True
            Add horizontal line at RR=1.0
        fig : matplotlib.figure.Figure, optional
        ax : matplotlib.axes.Axes, optional
        xlabel : str, optional
            Label for x-axis (defaults based on var/lag)
        ylabel : str, default "RR"
        title : str, optional
        color : str, default "steelblue"
            Line color
        ci_alpha : float, default 0.3
            Transparency of CI band
        linewidth : float, default 2
            Line width
        **kwargs
            Additional arguments passed to plot/fill_between

        Returns
        -------
        fig, ax
            Figure and axes objects

        Raises
        ------
        ValueError
            If neither or both var and lag supplied
        """
        if (var is None and lag is None) or (var is not None and lag is not None):
            raise ValueError("Exactly one of var or lag must be supplied")

        if fig is None:
            fig, ax = plt.subplots(figsize=(9, 6))
        elif ax is None:
            ax = fig.gca()

        if var is not None:
            # Find closest exposure value in predvar
            idx = np.argmin(np.abs(self.predvar - var))
            x_vals = self.predlag
            y_vals = self.RR[idx, :]
            y_low = self.RR_low[idx, :]
            y_high = self.RR_high[idx, :]
            if xlabel is None:
                xlabel = "Lag"
            if title is None:
                title = f"Lag-response at exposure={self.predvar[idx]:.2f}"
        else:
            # Find closest lag in predlag
            idx = np.argmin(np.abs(self.predlag - lag))
            x_vals = self.predvar
            y_vals = self.RR[:, idx]
            y_low = self.RR_low[:, idx]
            y_high = self.RR_high[:, idx]
            if xlabel is None:
                xlabel = "Exposure"
            if title is None:
                title = f"Exposure-response at lag={self.predlag[idx]}"

        ax.plot(x_vals, y_vals, color=color, linewidth=linewidth, **kwargs)

        if ci:
            ax.fill_between(x_vals, y_low, y_high, color=color, alpha=ci_alpha)

        if reference_line:
            ax.axhline(y=1.0, color="gray", linestyle="--", linewidth=1, alpha=0.7)

        ax.set_xlabel(xlabel)
        ax.set_ylabel(ylabel)
        if title:
            ax.set_title(title)
        ax.grid(True, alpha=0.3)

        return fig, ax

    def plot_overall(
        self,
        ci: bool = True,
        reference_line: bool = True,
        xlabel="Exposure",
        ylabel="Overall RR",
        title="Cumulative effect",
        fig=None,
        ax=None,
        color="steelblue",
        ci_alpha=0.3,
        linewidth=2,
        **kwargs,
    ):
        """
        Plot cumulative exposure-response curve (overall effect).

        Parameters
        ----------
        ci : bool, default True
            Show 95% confidence interval
        reference_line : bool, default True
            Add horizontal line at RR=1.0
        xlabel : str
        ylabel : str
        title : str
        fig : matplotlib.figure.Figure, optional
        ax : matplotlib.axes.Axes, optional
        color : str, default "steelblue"
        ci_alpha : float, default 0.3
        linewidth : float, default 2
            Line width
        **kwargs
            Additional arguments to plot/fill_between

        Returns
        -------
        fig, ax
        """
        if fig is None:
            fig, ax = plt.subplots(figsize=(9, 6))
        elif ax is None:
            ax = fig.gca()

        ax.plot(self.predvar, self.allRR, color=color, linewidth=linewidth, **kwargs)

        if ci:
            ax.fill_between(
                self.predvar, self.allRR_low, self.allRR_high, color=color, alpha=ci_alpha
            )

        if reference_line:
            ax.axhline(y=1.0, color="gray", linestyle="--", linewidth=1, alpha=0.7)

        ax.set_xlabel(xlabel)
        ax.set_ylabel(ylabel)
        if title:
            ax.set_title(title)
        ax.grid(True, alpha=0.3)

        return fig, ax


class CrossPred:
    """
    High-level user API for DLNM prediction and visualization.

    Owns a CrossBasis instance internally and manages the full prediction pipeline.

    Parameters
    ----------
    var_basis : str or callable, default "ns"
        Basis for exposure dimension
    var_df : int, optional
        Degrees of freedom for exposure basis
    var_knots : np.ndarray, optional
        Interior knots for exposure basis
    var_kwargs : dict, optional
        Additional kwargs for exposure basis
    lag_basis : str or callable, default "ns"
        Basis for lag dimension
    lag_df : int, optional
        Degrees of freedom for lag basis
    lag_knots : np.ndarray, optional
        Interior knots for lag basis
    lag_kwargs : dict, optional
        Additional kwargs for lag basis
    max_lag : int, required
        Maximum lag
    lag_knot_scale : str, default "log"
        Scale for automatic lag knot placement
    na_action : str or float, default "drop"
        Strategy for NaN rows from lag padding
    cen : float, "median", "mean", or "minimum_risk", default "median"
        Reference/centering value for predictions
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
        cen="median",
    ):
        self._crossbasis = CrossBasis(
            var_basis=var_basis,
            var_df=var_df,
            var_knots=var_knots,
            var_kwargs=var_kwargs,
            lag_basis=lag_basis,
            lag_df=lag_df,
            lag_knots=lag_knots,
            lag_kwargs=lag_kwargs,
            max_lag=max_lag,
            lag_knot_scale=lag_knot_scale,
            na_action=na_action,
        )
        self.cen = cen
        self._W = None
        self._X_train = None

    def _resolve_cen(self, cen, X):
        """
        Resolve centering value from string or numeric.

        Parameters
        ----------
        cen : float, "median", "mean", "minimum_risk", or None
            Centering specification
        X : np.ndarray
            Training data (for median/mean)

        Returns
        -------
        float or None
            Resolved centering value (None if "minimum_risk", to be resolved in predict)
        """
        if isinstance(cen, (int, float)):
            return float(cen)
        elif cen == "median":
            return float(np.nanmedian(X))
        elif cen == "mean":
            return float(np.nanmean(X))
        elif cen == "minimum_risk":
            return None  # Resolved in predict()
        else:
            raise ValueError(
                f"cen must be a float, 'median', 'mean', or 'minimum_risk'. Got: {cen!r}"
            )

    def fit(self, X):
        """
        Fit the DLNM to training data.

        Returns the cross-basis matrix W for use in GLM construction.

        Parameters
        ----------
        X : np.ndarray or pd.Series or pd.DataFrame
            Training exposure data

        Returns
        -------
        np.ndarray or pd.DataFrame
            Cross-basis matrix W, shape (n, v_x * v_l)
        """
        X_arr = to_array(X)
        self._X_train = X_arr

        self._W = self._crossbasis.fit_transform(X)

        # Count NaN rows and warn if na_action="drop"
        if isinstance(self._W, pd.DataFrame):
            nan_rows = self._W.isna().any(axis=1).sum()
        else:
            nan_rows = np.isnan(self._W).any(axis=1).sum()

        if nan_rows > 0:
            warnings.warn(
                f"CrossPred.fit: {nan_rows} rows with NaNs from lag padding. "
                "These rows will need to be dropped before GLM fitting.",
                UserWarning,
            )

        return self._W

    def predict(
        self,
        model=None,
        at: Union[np.ndarray, list] = None,
        cen: Union[float, str, None] = None,
        coef: np.ndarray = None,
        vcov: np.ndarray = None,
    ) -> PredictionResult:
        """
        Predict DLNM at specified exposure values.

        Can be called with either a statsmodels GLM result object, or with explicit
        coef and vcov arrays.

        Parameters
        ----------
        model : statsmodels result object, optional
            Fitted GLM result. Extract params and cov_params() for cross-basis terms.
        at : np.ndarray or list
            Exposure values for prediction (may be outside training range)
        cen : float, "median", "mean", "minimum_risk", or None, optional
            Centering value. If None, uses the value set at construction.
        coef : np.ndarray, optional
            Coefficient vector for cross-basis terms (alternative to model)
        vcov : np.ndarray, optional
            Covariance matrix for cross-basis terms (alternative to model)

        Returns
        -------
        PredictionResult
            Predictions at each (exposure, lag) combination
        """
        if at is None:
            raise ValueError("at must be supplied")

        at = np.asarray(at, dtype=np.float64)

        # Resolve centering value
        if cen is None:
            cen = self.cen

        # Handle "minimum_risk" by preliminary pass
        if cen == "minimum_risk":
            # Preliminary prediction over 200 equally-spaced points
            at_prelim = np.linspace(
                np.nanmin(self._X_train), np.nanmax(self._X_train), 200
            )
            cen_prelim = self._resolve_cen("median", self._X_train)  # Use median for prelim
            result_prelim = self.predict(
                model=model,
                at=at_prelim,
                cen=cen_prelim,
                coef=coef,
                vcov=vcov,
            )
            min_idx = np.argmin(result_prelim.allfit)
            cen = float(at_prelim[min_idx])
            warnings.warn(
                f"cen='minimum_risk' is approximate and may be unreliable for monotone "
                f"exposure-response relationships. The minimum was found at {cen:.2f}.",
                UserWarning,
            )
        else:
            cen = self._resolve_cen(cen, self._X_train)

        # Extract coefficients and covariance
        if model is not None:
            coef = np.asarray(model.params)
            vcov = np.asarray(model.cov_params())
        elif coef is None or vcov is None:
            raise ValueError("Either model or (coef, vcov) must be supplied")

        # Compute expected cross-basis size
        v_x = self._crossbasis.var_df_ or 3
        v_l = self._crossbasis.lag_df_ or 3
        expected_cb_size = v_x * v_l

        # Subset to cross-basis terms
        # If coef has one more element than expected, skip the first (intercept)
        if len(coef) == expected_cb_size + 1:
            g_hat = coef[1:]
            V_g_hat = vcov[1:, 1:]
        else:
            g_hat = coef
            V_g_hat = vcov

        # Build prediction basis matrix at specified exposure values
        # Grid ordering: predvar varies fastest, predlag slowest
        predvar = at
        predlag = np.arange(self._crossbasis.max_lag + 1, dtype=np.float64)

        # Prediction matrix for each (exposure, lag) pair
        varvec = np.tile(predvar, len(predlag))  # predvar repeated for each lag
        lagvec = np.repeat(predlag, len(predvar))  # each lag repeated

        # Build exposure basis Z at prediction points
        Z_pred = build_basis(
            predvar,
            self._crossbasis.var_basis,
            self._crossbasis.var_df or 3,
            knots=self._crossbasis.var_knots_,
            boundary_knots=self._crossbasis.boundary_knots_,
            **self._crossbasis.var_kwargs,
        )

        # Lag basis is on fixed lag vector
        C = self._crossbasis.C_

        v_x = Z_pred.shape[1]
        v_l = C.shape[1]
        n_pred = len(predvar)
        n_lag = len(predlag)

        # Build W_pred: row (lag_idx * n_pred + idx) = kron(Z_pred[idx], C[lag_idx])
        # predvar varies fastest (inner), predlag slowest (outer)
        m_pred = n_pred * n_lag
        W_pred = np.zeros((m_pred, v_x * v_l), dtype=np.float64)
        for lag_idx in range(n_lag):
            for idx in range(n_pred):
                flat_idx = lag_idx * n_pred + idx
                W_pred[flat_idx, :] = np.kron(Z_pred[idx, :], C[lag_idx, :])

        # Compute per-(exposure, lag) predictions and SEs
        flat_fit = W_pred @ g_hat
        diag_terms = np.sum(W_pred * (W_pred @ V_g_hat), axis=1)
        flat_se = np.sqrt(np.maximum(diag_terms, 0))

        # Reshape to (n_pred, n_lag) — predvar on rows, lag on columns
        matfit = flat_fit.reshape(n_lag, n_pred).T
        matse  = flat_se.reshape(n_lag, n_pred).T

        # Center by subtracting prediction at cen across all lags
        if cen is not None:
            Z_cen = build_basis(
                np.array([cen]),
                self._crossbasis.var_basis,
                self._crossbasis.var_df or 3,
                knots=self._crossbasis.var_knots_,
                boundary_knots=self._crossbasis.boundary_knots_,
                **self._crossbasis.var_kwargs,
            )
            W_cen = np.array([np.kron(Z_cen[0, :], C[l, :]) for l in range(n_lag)])
            cen_fit = W_cen @ g_hat          # shape (n_lag,)
            matfit = matfit - cen_fit[np.newaxis, :]
        else:
            cen = np.nan

        # Cumulative effect: sum over lags; variance uses summed W rows (Eq. 11-12)
        allfit = np.sum(matfit, axis=1)
        W_all = np.array([
            np.sum([np.kron(Z_pred[idx, :], C[l, :]) for l in range(n_lag)], axis=0)
            for idx in range(n_pred)
        ])
        if cen is not np.nan:
            W_cen_all = np.sum(W_cen, axis=0)
            W_all = W_all - W_cen_all[np.newaxis, :]
        allse = np.sqrt(np.maximum(
            np.sum(W_all * (W_all @ V_g_hat), axis=1), 0
        ))

        # Exponentiate for RR scale
        RR = np.exp(matfit)
        RR_low = np.exp(matfit - 1.96 * matse)
        RR_high = np.exp(matfit + 1.96 * matse)
        allRR = np.exp(allfit)
        allRR_low = np.exp(allfit - 1.96 * allse)
        allRR_high = np.exp(allfit + 1.96 * allse)

        return PredictionResult(
            matfit=matfit,
            matse=matse,
            allfit=allfit,
            allse=allse,
            RR=RR,
            RR_low=RR_low,
            RR_high=RR_high,
            allRR=allRR,
            allRR_low=allRR_low,
            allRR_high=allRR_high,
            predvar=predvar,
            predlag=predlag,
            cen=cen,
        )
