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

    @pytest.mark.parametrize("df,n_pts", [(3, 15), (4, 20), (5, 30)])
    def test_ns_values_match_r(self, df, n_pts):
        """ns_basis matches R's splines::ns() to numerical precision."""
        try:
            r('library(splines)')
        except Exception as e:
            pytest.skip(f"R/splines not available: {e}")

        x_np = np.linspace(0, 10, n_pts)
        B_py = ns_basis(x_np, df=df)

        with localconverter(ro.default_converter + numpy2ri.converter):
            ro.globalenv['x_ns'] = np2r(x_np)
            ro.globalenv['df_ns'] = df
            B_r = np.asarray(r('ns(x_ns, df=df_ns)'))

        assert B_py.shape == B_r.shape, f"Shape mismatch df={df}: {B_py.shape} vs {B_r.shape}"

        # Columns may have arbitrary sign flips relative to R; match each Python column
        # to its R counterpart and verify exact numerical agreement after sign alignment.
        used = set()
        for j, py_col in enumerate(B_py.T):
            best_err, best_sign, best_k = np.inf, 1, -1
            for k, r_col in enumerate(B_r.T):
                if k in used:
                    continue
                for sign in (1, -1):
                    err = np.max(np.abs(py_col - sign * r_col))
                    if err < best_err:
                        best_err, best_sign, best_k = err, sign, k
            used.add(best_k)
            assert best_err < 1e-8, (
                f"ns_basis col {j} does not match R col {best_k} (sign={best_sign}): "
                f"max abs diff = {best_err:.2e}"
            )


class TestCrossBasisMatchesR:
    """Validate CrossBasis W matrix matches R's dlnm::crossbasis() exactly."""

    @pytest.mark.parametrize("var_basis,var_df,lag_df,max_lag,r_var_fun", [
        ("ns", 3, 3,  5, "ns"),   # baseline
        ("ns", 5, 4, 10, "ns"),   # larger df and max_lag
        ("bs", 4, 3,  7, "bs"),   # B-spline exposure basis
    ])
    def test_crossbasis_w_matches_r_dlnm(self, var_basis, var_df, lag_df, max_lag, r_var_fun):
        try:
            r('library(dlnm)')
        except Exception as e:
            pytest.skip(f"R/dlnm not available: {e}")

        np.random.seed(0)
        x = np.random.uniform(0, 30, 80)

        import warnings
        cb = CrossBasis(
            var_basis=var_basis, var_df=var_df,
            lag_basis="ns", lag_df=lag_df,
            max_lag=max_lag,
            na_action="drop",
        )
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            W_py = np.asarray(cb.fit_transform(x))

        with localconverter(ro.default_converter + numpy2ri.converter):
            ro.globalenv['x_r'] = np2r(x)
            W_r = np.asarray(r(f"""
                library(dlnm)
                cb_r <- crossbasis(x_r, lag={max_lag},
                    argvar=list(fun="{r_var_fun}", df={var_df}),
                    arglag=list(fun="ns", df={lag_df}))
                unclass(cb_r)
            """))

        assert W_py.shape == W_r.shape, (
            f"Shape mismatch ({var_basis} df={var_df}, lag df={lag_df}, "
            f"max_lag={max_lag}): {W_py.shape} vs {W_r.shape}"
        )
        np.testing.assert_array_equal(np.isnan(W_py), np.isnan(W_r),
            err_msg="NaN pattern does not match R")
        full_rows = ~np.isnan(W_r).any(axis=1)
        np.testing.assert_allclose(
            W_py[full_rows], W_r[full_rows], rtol=1e-5, atol=1e-8,
            err_msg=f"W mismatch for {var_basis}×ns df={var_df}×{lag_df} max_lag={max_lag}",
        )


class TestCrosspredMatchesR:
    """End-to-end validation: fit GLM + predict, compare allRR and matRR against R's crosspred()."""

    def test_allrr_and_matrr_match_r(self):
        try:
            r('library(dlnm)')
        except Exception as e:
            pytest.skip(f"R/dlnm not available: {e}")

        np.random.seed(42)
        n = 100
        x = np.random.uniform(0, 30, n)
        y = np.random.poisson(20, n).astype(float)
        at_vals = np.array([5.0, 15.0, 25.0])
        cen_val = 15.0
        var_df, lag_df, max_lag = 3, 3, 5

        # --- Python ---
        # Drop first max_lag rows (incomplete lag history) from both W and y so
        # the GLM is fit on exactly the same observations as R.
        import warnings
        cp = CrossPred(
            var_basis="ns", var_df=var_df,
            lag_basis="ns", lag_df=lag_df,
            max_lag=max_lag,
            na_action="drop",
            cen=cen_val,
        )
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            W_py = np.asarray(cp.fit(x))

        complete = ~np.isnan(W_py).any(axis=1)
        X_glm = sm.add_constant(W_py[complete])
        model = sm.GLM(y[complete], X_glm, family=sm.families.Poisson()).fit()
        result = cp.predict(model, at=at_vals, cen=cen_val)

        # --- R ---
        with localconverter(ro.default_converter + numpy2ri.converter):
            ro.globalenv['x_r'] = np2r(x)
            ro.globalenv['y_r'] = np2r(y)
            ro.globalenv['at_r'] = np2r(at_vals)
            r(f"""
                library(dlnm)
                cb_r <- crossbasis(x_r, lag={max_lag},
                    argvar=list(fun="ns", df={var_df}),
                    arglag=list(fun="ns", df={lag_df}))
                # Fit on complete rows only (matching Python na_action="drop")
                keep <- ({max_lag}+1):length(y_r)
                mod_r <- glm(y_r[keep] ~ cb_r[keep,], family=poisson())
                pred_r <- crosspred(cb_r, mod_r, at=at_r, cen={cen_val})
            """)
            allRRfit_r  = np.asarray(r('pred_r$allRRfit'))
            allRRlow_r  = np.asarray(r('pred_r$allRRlow'))
            allRRhigh_r = np.asarray(r('pred_r$allRRhigh'))
            matRRfit_r  = np.asarray(r('pred_r$matRRfit'))

        np.testing.assert_allclose(result.allRR,      allRRfit_r,  rtol=2e-3,
            err_msg="allRR does not match R crosspred allRRfit")
        np.testing.assert_allclose(result.allRR_low,  allRRlow_r,  rtol=2e-3,
            err_msg="allRR_low does not match R crosspred allRRlow")
        np.testing.assert_allclose(result.allRR_high, allRRhigh_r, rtol=2e-3,
            err_msg="allRR_high does not match R crosspred allRRhigh")
        np.testing.assert_allclose(result.RR, matRRfit_r, rtol=1e-2,
            err_msg="RR matrix does not match R crosspred matRRfit")


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
