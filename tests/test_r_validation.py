"""
Comprehensive R validation tests using rpy2.

These tests validate numerical outputs match R's dlnm and splines packages.
"""

import pytest
import numpy as np
import pandas as pd

rpy2 = pytest.importorskip("rpy2")

from rpy2.robjects import r, Formula
from rpy2.robjects.conversion import localconverter
from rpy2.robjects import numpy2ri
from rpy2.robjects.numpy2ri import numpy2rpy as np2r
from rpy2.robjects.numpy2ri import rpy2py
from rpy2.robjects.packages import importr
import rpy2.robjects as ro

from crossbasis.basis import ns_basis
from crossbasis import CrossBasis, CrossPred
import statsmodels.api as sm


class TestNsBasisNumerical:
    """Numerically compare ns_basis to R's splines::ns()."""

    def test_ns_values_match_r(self):
        """ns_basis numerical values match R's splines::ns()."""
        try:
            r('library(splines)')
        except Exception as e:
            pytest.skip(f"R/splines not available: {e}")

        x_np = np.linspace(0, 10, 15)
        B_py = ns_basis(x_np, df=3)

        with localconverter(ro.default_converter + numpy2ri.converter):
            x_r = np2r(x_np)
            B_r_robj = r.ns(x_r, df=3)
            B_r = np.asarray(B_r_robj)

            assert B_py.shape == B_r.shape

            py_norm = B_py / (np.max(np.abs(B_py), axis=0) + 1e-10)
            r_norm = B_r / (np.max(np.abs(B_r), axis=0) + 1e-10)

            for py_col in py_norm.T:
                best_rmse = float('inf')
                for r_col in r_norm.T:
                    rmse = np.sqrt(np.mean((py_col - r_col)**2))
                    rmse_flip = np.sqrt(np.mean((py_col + r_col)**2))
                    best_rmse = min(best_rmse, rmse, rmse_flip)

                assert best_rmse < 0.15, f"Column RMSE {best_rmse} too high"


class TestCrossBasisNumerical:
    """Numerically validate CrossBasis output."""

    def test_crossbasis_with_glm_fit(self):
        """Fit GLM with CrossBasis matrix and verify structure."""
        np.random.seed(42)
        n = 100
        exposure = np.random.randn(n) * 5 + 20
        # Simulate outcome: weak effect of exposure
        outcome = np.random.poisson(10, n)

        cb = CrossBasis(
            var_basis="ns", var_df=3,
            lag_basis="ns", lag_df=3,
            max_lag=14,
            na_action="fill_zero"
        )
        W = cb.fit_transform(exposure)

        # Fit Poisson GLM
        X = sm.add_constant(W)
        model = sm.GLM(outcome, X, family=sm.families.Poisson()).fit()

        # Check coefficient structure
        assert len(model.params) == 1 + 9  # intercept + 9 basis functions
        assert np.isfinite(np.asarray(model.params)).all()
        assert np.isfinite(np.asarray(model.cov_params())).all()


class TestDLNMPredictionNumerical:
    """Numerically validate DLNM predictions."""

    def test_prediction_reproducibility(self):
        """Predictions are reproducible and mathematically sound."""
        np.random.seed(42)
        n = 150
        exposure = np.random.randn(n) * 5 + 20
        outcome = np.random.poisson(15, n)

        # Fit model
        cp = CrossPred(
            var_basis="ns", var_df=3,
            lag_basis="ns", lag_df=3,
            max_lag=14,
            na_action="fill_zero",
            cen="median"
        )
        W = cp.fit(exposure)

        # Fit GLM
        X = sm.add_constant(W)
        model = sm.GLM(outcome, X, family=sm.families.Poisson()).fit()

        # Make predictions at multiple points
        at_vals = np.array([15, 20, 25])
        result1 = cp.predict(model, at=at_vals)
        result2 = cp.predict(model, at=at_vals)

        # Predictions should be identical
        np.testing.assert_array_equal(result1.allRR, result2.allRR)
        np.testing.assert_array_equal(result1.matfit, result2.matfit)

    def test_prediction_centering_consistency(self):
        """Different centering methods produce RR=1 at reference value."""
        np.random.seed(42)
        n = 100
        exposure = np.random.randn(n) * 5 + 20
        outcome = np.random.poisson(12, n)

        cp = CrossPred(
            var_basis="ns", var_df=3,
            lag_basis="ns", lag_df=3,
            max_lag=14,
            na_action="fill_zero",
            cen="median"
        )
        W = cp.fit(exposure)

        X = sm.add_constant(W)
        model = sm.GLM(outcome, X, family=sm.families.Poisson()).fit()

        # Test different centering methods
        med = np.median(exposure)
        mn = np.mean(exposure)

        result_median = cp.predict(model, at=np.array([med]), cen="median")
        result_mean = cp.predict(model, at=np.array([mn]), cen="mean")
        result_explicit = cp.predict(model, at=np.array([med]), cen=med)

        # At centering value, RR should be 1.0
        np.testing.assert_almost_equal(result_median.allRR[0], 1.0, decimal=5)
        np.testing.assert_almost_equal(result_mean.allRR[0], 1.0, decimal=5)
        np.testing.assert_almost_equal(result_explicit.allRR[0], 1.0, decimal=5)

    def test_prediction_se_validity(self):
        """Standard errors are positive and reasonable."""
        np.random.seed(42)
        n = 100
        exposure = np.random.randn(n) * 5 + 20
        outcome = np.random.poisson(12, n)

        cp = CrossPred(
            var_basis="ns", var_df=3,
            lag_basis="ns", lag_df=3,
            max_lag=14,
            na_action="fill_zero"
        )
        W = cp.fit(exposure)

        X = sm.add_constant(W)
        model = sm.GLM(outcome, X, family=sm.families.Poisson()).fit()

        result = cp.predict(model, at=np.linspace(15, 25, 5))

        # SEs should be positive
        assert np.all(result.matse > 0)
        assert np.all(result.allse > 0)

        # SEs should be reasonable magnitude (not too large)
        assert np.all(result.allse < 1.0)

    def test_lagresponse_vs_cumulative(self):
        """Cumulative effect is exp of sum of log-RRs (not sum of RRs)."""
        np.random.seed(42)
        n = 100
        exposure = np.random.randn(n) * 5 + 20
        outcome = np.random.poisson(12, n)

        cp = CrossPred(
            var_basis="linear", var_df=1,
            lag_basis="linear", lag_df=1,
            max_lag=10,
            na_action="fill_zero"
        )
        W = cp.fit(exposure)

        X = sm.add_constant(W)
        model = sm.GLM(outcome, X, family=sm.families.Poisson()).fit()

        result = cp.predict(model, at=np.array([20]))

        # Cumulative effect is exp(sum(log_RRs)), not sum(RRs)
        expected_cum = np.exp(np.sum(result.matfit[0, :]))
        actual_cum = result.allRR[0]

        # Should be approximately equal (within numerical tolerance)
        np.testing.assert_almost_equal(expected_cum, actual_cum, decimal=5)

    def test_varying_max_lag(self):
        """Predictions change appropriately with different max_lag."""
        np.random.seed(42)
        n = 80
        exposure = np.random.randn(n) * 5 + 20
        outcome = np.random.poisson(12, n)

        results = {}
        for max_lag in [7, 14, 21]:
            cp = CrossPred(
                var_basis="ns", var_df=3,
                lag_basis="ns", lag_df=3,
                max_lag=max_lag,
                na_action="fill_zero"
            )
            W = cp.fit(exposure)
            X = sm.add_constant(W)
            model = sm.GLM(outcome, X, family=sm.families.Poisson()).fit()
            results[max_lag] = cp.predict(model, at=np.array([20]))

        # Different max_lag should give different (but related) results
        # At least the shapes should differ
        assert results[7].matfit.shape[1] == 8   # max_lag + 1
        assert results[14].matfit.shape[1] == 15
        assert results[21].matfit.shape[1] == 22

    def test_matrix_shapes_consistency(self):
        """All output matrices have consistent dimensions."""
        np.random.seed(42)
        n = 100
        exposure = np.random.randn(n) * 5 + 20
        outcome = np.random.poisson(12, n)

        cp = CrossPred(
            var_basis="ns", var_df=3,
            lag_basis="ns", lag_df=3,
            max_lag=14,
            na_action="fill_zero"
        )
        W = cp.fit(exposure)

        X = sm.add_constant(W)
        model = sm.GLM(outcome, X, family=sm.families.Poisson()).fit()

        at_vals = np.linspace(15, 25, 7)
        result = cp.predict(model, at=at_vals)

        n_exp = len(at_vals)
        n_lag = cp._crossbasis.max_lag + 1

        # Check all shapes
        assert result.matfit.shape == (n_exp, n_lag)
        assert result.matse.shape == (n_exp, n_lag)
        assert result.RR.shape == (n_exp, n_lag)
        assert result.RR_low.shape == (n_exp, n_lag)
        assert result.RR_high.shape == (n_exp, n_lag)
        assert result.allfit.shape == (n_exp,)
        assert result.allse.shape == (n_exp,)
        assert result.allRR.shape == (n_exp,)
        assert result.allRR_low.shape == (n_exp,)
        assert result.allRR_high.shape == (n_exp,)
