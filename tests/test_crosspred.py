"""
Tests for CrossPred high-level class and prediction results.
"""

import pytest
import numpy as np
import pandas as pd
import warnings
from crossbasis.crosspred import CrossPred, PredictionResult


class TestPredictionResultDataclass:
    """Tests for PredictionResult dataclass."""

    def test_prediction_result_creation(self):
        """PredictionResult can be created with all required attributes."""
        matfit = np.random.randn(10, 5)
        matse = np.abs(np.random.randn(10, 5))
        allfit = np.random.randn(10)
        allse = np.abs(np.random.randn(10))

        result = PredictionResult(
            matfit=matfit,
            matse=matse,
            allfit=allfit,
            allse=allse,
            RR=np.exp(matfit),
            RR_low=np.exp(matfit - 1.96 * matse),
            RR_high=np.exp(matfit + 1.96 * matse),
            allRR=np.exp(allfit),
            allRR_low=np.exp(allfit - 1.96 * allse),
            allRR_high=np.exp(allfit + 1.96 * allse),
            predvar=np.linspace(0, 10, 10),
            predlag=np.arange(5),
            cen=5.0,
        )

        assert result.matfit.shape == (10, 5)
        assert result.allfit.shape == (10,)
        assert result.cen == 5.0

    def test_to_frame(self):
        """to_frame() returns long-format DataFrame."""
        matfit = np.random.randn(3, 2)
        matse = np.abs(np.random.randn(3, 2))
        allfit = np.random.randn(3)
        allse = np.abs(np.random.randn(3))

        result = PredictionResult(
            matfit=matfit,
            matse=matse,
            allfit=allfit,
            allse=allse,
            RR=np.exp(matfit),
            RR_low=np.exp(matfit - 1.96 * matse),
            RR_high=np.exp(matfit + 1.96 * matse),
            allRR=np.exp(allfit),
            allRR_low=np.exp(allfit - 1.96 * allse),
            allRR_high=np.exp(allfit + 1.96 * allse),
            predvar=np.array([1.0, 2.0, 3.0]),
            predlag=np.array([0, 1]),
            cen=2.0,
        )

        df = result.to_frame()
        assert isinstance(df, pd.DataFrame)
        assert len(df) == 3 * 2  # 3 exposures * 2 lags
        assert "exposure" in df.columns
        assert "lag" in df.columns
        assert "logRR" in df.columns
        assert "se" in df.columns


class TestCrossPredInit:
    """Tests for CrossPred initialization."""

    def test_init_defaults(self):
        """CrossPred initializes with default arguments."""
        cp = CrossPred(var_df=3, lag_df=3, max_lag=14)
        assert cp.cen == "median"
        assert cp._crossbasis is not None

    def test_init_custom_cen(self):
        """CrossPred accepts custom cen argument."""
        cp = CrossPred(var_df=3, lag_df=3, max_lag=14, cen="mean")
        assert cp.cen == "mean"

    def test_init_numeric_cen(self):
        """CrossPred accepts numeric cen argument."""
        cp = CrossPred(var_df=3, lag_df=3, max_lag=14, cen=5.0)
        assert cp.cen == 5.0


class TestCrossPredFit:
    """Tests for CrossPred.fit()."""

    def test_fit_returns_W(self):
        """fit() returns cross-basis matrix W."""
        x = np.linspace(0, 10, 50)
        cp = CrossPred(var_df=3, lag_df=3, max_lag=14, na_action="fill_zero")
        W = cp.fit(x)
        assert W is not None
        assert W.shape == (50, 9)

    def test_fit_stores_W_internally(self):
        """fit() stores W internally."""
        x = np.linspace(0, 10, 50)
        cp = CrossPred(var_df=3, lag_df=3, max_lag=14, na_action="fill_zero")
        cp.fit(x)
        assert cp._W is not None

    def test_fit_stores_X_train(self):
        """fit() stores training data."""
        x = np.linspace(0, 10, 50)
        cp = CrossPred(var_df=3, lag_df=3, max_lag=14)
        cp.fit(x)
        assert cp._X_train is not None
        np.testing.assert_array_equal(cp._X_train, x)


class TestCrossPredResolveCen:
    """Tests for centering value resolution."""

    def test_resolve_cen_float(self):
        """_resolve_cen with float returns that float."""
        cp = CrossPred(var_df=3, lag_df=3, max_lag=14)
        x = np.linspace(0, 10, 50)
        cen = cp._resolve_cen(5.0, x)
        assert cen == 5.0

    def test_resolve_cen_median(self):
        """_resolve_cen with 'median' returns median of X."""
        cp = CrossPred(var_df=3, lag_df=3, max_lag=14)
        x = np.linspace(0, 10, 50)
        cen = cp._resolve_cen("median", x)
        np.testing.assert_almost_equal(cen, np.median(x))

    def test_resolve_cen_mean(self):
        """_resolve_cen with 'mean' returns mean of X."""
        cp = CrossPred(var_df=3, lag_df=3, max_lag=14)
        x = np.linspace(0, 10, 50)
        cen = cp._resolve_cen("mean", x)
        np.testing.assert_almost_equal(cen, np.mean(x))

    def test_resolve_cen_minimum_risk(self):
        """_resolve_cen with 'minimum_risk' returns None (deferred)."""
        cp = CrossPred(var_df=3, lag_df=3, max_lag=14)
        x = np.linspace(0, 10, 50)
        cen = cp._resolve_cen("minimum_risk", x)
        assert cen is None

    def test_resolve_cen_invalid_string(self):
        """_resolve_cen raises for invalid string."""
        cp = CrossPred(var_df=3, lag_df=3, max_lag=14)
        x = np.linspace(0, 10, 50)
        with pytest.raises(ValueError, match="cen must be"):
            cp._resolve_cen("invalid", x)


class TestCrossPredPredict:
    """Tests for CrossPred.predict()."""

    def create_mock_model(self, coef, vcov):
        """Create a mock statsmodels result object."""

        class MockModel:
            def __init__(self, coef, vcov):
                self.params = pd.Series(coef)
                self._vcov = vcov

            def cov_params(self):
                return pd.DataFrame(self._vcov)

        return MockModel(coef, vcov)

    def test_predict_requires_at(self):
        """predict() raises if at not supplied."""
        x = np.linspace(0, 10, 50)
        cp = CrossPred(var_df=3, lag_df=3, max_lag=14, na_action="fill_zero")
        cp.fit(x)

        coef = np.random.randn(9)
        vcov = np.eye(9) * 0.01
        model = self.create_mock_model(coef, vcov)

        with pytest.raises(ValueError, match="at must be supplied"):
            cp.predict(model=model)

    def test_predict_output_shapes(self):
        """predict() returns PredictionResult with correct shapes."""
        x = np.linspace(0, 10, 50)
        cp = CrossPred(var_df=3, lag_df=3, max_lag=14, na_action="fill_zero")
        cp.fit(x)

        at = np.linspace(0, 10, 5)
        coef = np.random.randn(9)
        vcov = np.eye(9) * 0.01
        model = self.create_mock_model(coef, vcov)

        result = cp.predict(model=model, at=at, cen=5.0)

        assert result.matfit.shape == (5, 15)  # 5 exposures, 15 lags (0-14)
        assert result.matse.shape == (5, 15)
        assert result.allfit.shape == (5,)
        assert result.allse.shape == (5,)
        assert result.RR.shape == (5, 15)

    def test_predict_cen_float(self):
        """predict() with explicit float cen produces RR=1.0 at cen."""
        x = np.linspace(0, 10, 50)
        cp = CrossPred(var_df=3, lag_df=3, max_lag=14, na_action="fill_zero")
        cp.fit(x)

        at = np.array([5.0])  # Include cen value in prediction points
        coef = np.random.randn(9)
        vcov = np.eye(9) * 0.001
        model = self.create_mock_model(coef, vcov)

        result = cp.predict(model=model, at=at, cen=5.0)

        # At cen, allRR should be 1.0
        np.testing.assert_almost_equal(result.allRR[0], 1.0, decimal=5)

    def test_predict_cen_median(self):
        """predict() with cen='median' produces RR=1.0 at median."""
        x = np.linspace(0, 10, 50)
        cp = CrossPred(var_df=3, lag_df=3, max_lag=14, na_action="fill_zero", cen="median")
        cp.fit(x)

        med = np.median(x)
        at = np.array([med])
        coef = np.random.randn(9)
        vcov = np.eye(9) * 0.001
        model = self.create_mock_model(coef, vcov)

        result = cp.predict(model=model, at=at)

        np.testing.assert_almost_equal(result.allRR[0], 1.0, decimal=5)

    def test_predict_minimum_risk_warning(self):
        """predict() with cen='minimum_risk' emits UserWarning."""
        x = np.linspace(0, 10, 50)
        cp = CrossPred(var_df=3, lag_df=3, max_lag=14, na_action="fill_zero", cen="minimum_risk")
        cp.fit(x)

        at = np.linspace(0, 10, 10)
        coef = np.random.randn(9)
        vcov = np.eye(9) * 0.01
        model = self.create_mock_model(coef, vcov)

        with warnings.catch_warnings(record=True) as w:
            warnings.simplefilter("always")
            result = cp.predict(model=model, at=at)
            assert len(w) > 0
            assert issubclass(w[-1].category, UserWarning)
            assert "minimum_risk" in str(w[-1].message).lower()

    def test_predict_coef_vcov_alternative(self):
        """predict() accepts coef and vcov directly."""
        x = np.linspace(0, 10, 50)
        cp = CrossPred(var_df=3, lag_df=3, max_lag=14, na_action="fill_zero")
        cp.fit(x)

        at = np.linspace(0, 10, 5)
        coef = np.random.randn(9)
        vcov = np.eye(9) * 0.01

        result = cp.predict(coef=coef, vcov=vcov, at=at, cen=5.0)

        assert result.matfit.shape == (5, 15)


class TestPredictionResultPlots:
    """Tests for plotting methods on PredictionResult."""

    def create_sample_result(self):
        """Create a sample PredictionResult for plotting tests."""
        matfit = np.random.randn(5, 3) * 0.1
        matse = np.abs(np.random.randn(5, 3)) * 0.05
        allfit = np.sum(matfit, axis=1)
        allse = np.sqrt(np.sum(matse**2, axis=1))

        return PredictionResult(
            matfit=matfit,
            matse=matse,
            allfit=allfit,
            allse=allse,
            RR=np.exp(matfit),
            RR_low=np.exp(matfit - 1.96 * matse),
            RR_high=np.exp(matfit + 1.96 * matse),
            allRR=np.exp(allfit),
            allRR_low=np.exp(allfit - 1.96 * allse),
            allRR_high=np.exp(allfit + 1.96 * allse),
            predvar=np.array([1.0, 2.0, 3.0, 4.0, 5.0]),
            predlag=np.array([0, 1, 2]),
            cen=3.0,
        )

    def test_plot_3d_runs(self):
        """plot_3d() runs without error."""
        result = self.create_sample_result()
        try:
            fig, ax = result.plot_3d()
            assert fig is not None
            assert ax is not None
        except Exception as e:
            pytest.fail(f"plot_3d() raised: {e}")

    def test_plot_slice_var(self):
        """plot_slice() with var argument runs without error."""
        result = self.create_sample_result()
        try:
            fig, ax = result.plot_slice(var=2.5)
            assert fig is not None
            assert ax is not None
        except Exception as e:
            pytest.fail(f"plot_slice(var=2.5) raised: {e}")

    def test_plot_slice_lag(self):
        """plot_slice() with lag argument runs without error."""
        result = self.create_sample_result()
        try:
            fig, ax = result.plot_slice(lag=1)
            assert fig is not None
            assert ax is not None
        except Exception as e:
            pytest.fail(f"plot_slice(lag=1) raised: {e}")

    def test_plot_slice_requires_var_or_lag(self):
        """plot_slice() raises if neither var nor lag supplied."""
        result = self.create_sample_result()
        with pytest.raises(ValueError, match="Exactly one"):
            result.plot_slice()

    def test_plot_slice_excludes_both_var_lag(self):
        """plot_slice() raises if both var and lag supplied."""
        result = self.create_sample_result()
        with pytest.raises(ValueError, match="Exactly one"):
            result.plot_slice(var=2.5, lag=1)

    def test_plot_overall_runs(self):
        """plot_overall() runs without error."""
        result = self.create_sample_result()
        try:
            fig, ax = result.plot_overall()
            assert fig is not None
            assert ax is not None
        except Exception as e:
            pytest.fail(f"plot_overall() raised: {e}")

    def test_plot_customization_kwargs(self):
        """Plots accept customization kwargs."""
        result = self.create_sample_result()
        try:
            # Test various kwargs
            result.plot_slice(var=2.5, color="red", ci_alpha=0.5)
            result.plot_overall(color="green", linewidth=3)
            result.plot_3d(cmap="plasma")
        except Exception as e:
            pytest.fail(f"Plot customization raised: {e}")
