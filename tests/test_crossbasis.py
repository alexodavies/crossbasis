"""
Tests for CrossBasis transformer.
"""

import pytest
import numpy as np
import pandas as pd
from crossbasis.crossbasis import CrossBasis


class TestCrossBasisInit:
    """Tests for CrossBasis initialization."""

    def test_max_lag_required(self):
        """CrossBasis raises ValueError if max_lag not supplied."""
        with pytest.raises(ValueError, match="max_lag is required"):
            CrossBasis()

    def test_mutually_exclusive_var_knots_df(self):
        """CrossBasis raises if both var_knots and var_df supplied."""
        with pytest.raises(ValueError, match="mutually exclusive"):
            CrossBasis(var_knots=np.array([3, 5]), var_df=3, max_lag=14)

    def test_mutually_exclusive_lag_knots_df(self):
        """CrossBasis raises if both lag_knots and lag_df supplied."""
        with pytest.raises(ValueError, match="mutually exclusive"):
            CrossBasis(lag_knots=np.array([3, 5]), lag_df=3, max_lag=14)

    def test_init_valid(self):
        """CrossBasis initializes with valid arguments."""
        cb = CrossBasis(var_df=3, lag_df=3, max_lag=14)
        assert cb.max_lag == 14
        assert cb.var_df == 3
        assert cb.lag_df == 3


class TestCrossBasisFit:
    """Tests for CrossBasis.fit()."""

    def test_fit_sets_is_fitted(self):
        """fit() sets is_fitted_ attribute."""
        x = np.linspace(0, 10, 50)
        cb = CrossBasis(var_df=3, lag_df=3, max_lag=14)
        assert not cb.is_fitted_
        cb.fit(x)
        assert cb.is_fitted_

    def test_fit_computes_var_knots(self):
        """fit() computes var_knots_ from var_df."""
        x = np.linspace(0, 10, 50)
        cb = CrossBasis(var_df=3, lag_df=3, max_lag=14)
        cb.fit(x)
        assert hasattr(cb, "var_knots_")
        assert cb.var_knots_ is not None

    def test_fit_computes_lag_knots_log(self):
        """fit() computes lag_knots_ using log scale by default."""
        x = np.linspace(0, 10, 50)
        cb = CrossBasis(var_df=3, lag_df=3, max_lag=14, lag_knot_scale="log")
        cb.fit(x)
        assert hasattr(cb, "lag_knots_")

    def test_fit_computes_lag_knots_equal(self):
        """fit() computes lag_knots_ using equal scale."""
        x = np.linspace(0, 10, 50)
        cb = CrossBasis(var_df=3, lag_df=3, max_lag=14, lag_knot_scale="equal")
        cb.fit(x)
        assert hasattr(cb, "lag_knots_")

    def test_fit_stores_boundary_knots(self):
        """fit() stores boundary_knots_."""
        x = np.linspace(0, 10, 50)
        cb = CrossBasis(var_df=3, lag_df=3, max_lag=14)
        cb.fit(x)
        assert hasattr(cb, "boundary_knots_")
        np.testing.assert_array_almost_equal(cb.boundary_knots_, [0, 10])

    def test_fit_returns_self(self):
        """fit() returns self for method chaining."""
        x = np.linspace(0, 10, 50)
        cb = CrossBasis(var_df=3, lag_df=3, max_lag=14)
        result = cb.fit(x)
        assert result is cb


class TestCrossBasisTransform:
    """Tests for CrossBasis.transform()."""

    def test_transform_requires_fit(self):
        """transform() raises if not fitted."""
        x = np.linspace(0, 10, 50)
        cb = CrossBasis(var_df=3, lag_df=3, max_lag=14)
        with pytest.raises(ValueError, match="must be fitted"):
            cb.transform(x)

    def test_transform_output_shape(self):
        """transform() output shape is n × (v_x * v_l)."""
        x = np.linspace(0, 10, 50)
        cb = CrossBasis(var_df=3, lag_df=4, max_lag=14)
        cb.fit(x)
        W = cb.transform(x)
        expected_cols = 3 * 4  # v_x * v_l
        assert W.shape == (50, expected_cols)

    def test_transform_nan_rows_drop(self):
        """transform() with na_action='drop' marks lag padding rows as NaN."""
        x = np.linspace(0, 10, 50)
        cb = CrossBasis(var_df=3, lag_df=4, max_lag=14, na_action="drop")
        cb.fit(x)
        W = cb.transform(x)

        # First max_lag rows should have NaN
        for i in range(cb.max_lag):
            assert np.isnan(W[i, :]).any()

    def test_transform_nan_rows_fill_zero(self):
        """transform() with na_action='fill_zero' produces no NaNs."""
        x = np.linspace(0, 10, 50)
        cb = CrossBasis(var_df=3, lag_df=4, max_lag=14, na_action="fill_zero")
        cb.fit(x)
        W = cb.transform(x)
        assert not np.isnan(W).any()

    def test_transform_nan_rows_fill_mean(self):
        """transform() with na_action='fill_mean' produces no NaNs."""
        x = np.linspace(0, 10, 50)
        cb = CrossBasis(var_df=3, lag_df=4, max_lag=14, na_action="fill_mean")
        cb.fit(x)
        W = cb.transform(x)
        assert not np.isnan(W).any()

    def test_transform_nan_rows_fill_numeric(self):
        """transform() with numeric na_action produces no NaNs."""
        x = np.linspace(0, 10, 50)
        cb = CrossBasis(var_df=3, lag_df=4, max_lag=14, na_action=0.0)
        cb.fit(x)
        W = cb.transform(x)
        assert not np.isnan(W).any()

    def test_transform_finite_values(self):
        """transform() output is finite (except for dropped NaN rows)."""
        x = np.linspace(0, 10, 50)
        cb = CrossBasis(var_df=3, lag_df=4, max_lag=14, na_action="fill_zero")
        cb.fit(x)
        W = cb.transform(x)
        assert np.isfinite(W).all()

    def test_transform_numpy_input(self):
        """transform() accepts numpy array input."""
        x = np.linspace(0, 10, 50)
        cb = CrossBasis(var_df=3, lag_df=4, max_lag=14, na_action="fill_zero")
        cb.fit(x)
        W = cb.transform(x)
        assert isinstance(W, np.ndarray)

    def test_transform_series_input(self):
        """transform() accepts pandas Series input."""
        x = pd.Series(np.linspace(0, 10, 50))
        cb = CrossBasis(var_df=3, lag_df=4, max_lag=14, na_action="fill_zero")
        cb.fit(x)
        W = cb.transform(x)
        assert isinstance(W, pd.DataFrame)

    def test_transform_dataframe_input(self):
        """transform() accepts pandas DataFrame column input."""
        x = pd.DataFrame({"temp": np.linspace(0, 10, 50)})
        cb = CrossBasis(var_df=3, lag_df=4, max_lag=14, na_action="fill_zero")
        cb.fit(x["temp"])
        W = cb.transform(x["temp"])
        assert isinstance(W, pd.DataFrame)


class TestCrossBasisFitTransform:
    """Tests for CrossBasis.fit_transform()."""

    def test_fit_transform_output_shape(self):
        """fit_transform() output shape is n × (v_x * v_l)."""
        x = np.linspace(0, 10, 50)
        cb = CrossBasis(var_df=3, lag_df=4, max_lag=14, na_action="fill_zero")
        W = cb.fit_transform(x)
        assert W.shape == (50, 12)

    def test_fit_transform_sets_fitted(self):
        """fit_transform() sets is_fitted_."""
        x = np.linspace(0, 10, 50)
        cb = CrossBasis(var_df=3, lag_df=4, max_lag=14)
        cb.fit_transform(x)
        assert cb.is_fitted_

    def test_fit_transform_matches_fit_then_transform(self):
        """fit_transform() output matches fit().transform()."""
        x = np.linspace(0, 10, 50)
        cb1 = CrossBasis(var_df=3, lag_df=4, max_lag=14, na_action="fill_zero")
        W1 = cb1.fit_transform(x)

        cb2 = CrossBasis(var_df=3, lag_df=4, max_lag=14, na_action="fill_zero")
        W2 = cb2.fit(x).transform(x)

        if isinstance(W1, pd.DataFrame):
            W1 = W1.values
        if isinstance(W2, pd.DataFrame):
            W2 = W2.values

        np.testing.assert_array_almost_equal(W1, W2)


class TestCrossBasisCustomBasis:
    """Tests for custom basis functions."""

    def test_custom_var_basis(self):
        """CrossBasis works with custom var_basis callable."""

        def my_basis(x, df, **kwargs):
            return np.tile(x.reshape(-1, 1), (1, df))

        x = np.linspace(0, 10, 50)
        cb = CrossBasis(var_basis=my_basis, var_df=2, lag_df=3, max_lag=14, na_action="fill_zero")
        W = cb.fit_transform(x)
        assert W.shape == (50, 6)

    def test_custom_lag_basis(self):
        """CrossBasis works with custom lag_basis callable."""

        def my_basis(x, df, **kwargs):
            B = np.zeros((len(x), df))
            for j in range(df):
                B[:, j] = x ** (j + 1)
            return B

        x = np.linspace(0, 10, 50)
        cb = CrossBasis(var_df=3, lag_basis=my_basis, lag_df=2, max_lag=14, na_action="fill_zero")
        W = cb.fit_transform(x)
        assert W.shape == (50, 6)
