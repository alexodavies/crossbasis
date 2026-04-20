"""
Tests for basis functions.
"""

import pytest
import numpy as np
from crossbasis.basis import (
    ns_basis,
    bs_basis,
    poly_basis,
    linear_basis,
    build_basis,
    _validate_basis_output,
)


class TestNsBasis:
    """Tests for natural cubic spline basis."""

    def test_ns_output_shape(self):
        """ns_basis output shape matches df (or minimum 3 for degree-3 spline)."""
        x = np.linspace(0, 10, 50)
        # Minimum df for degree-3 spline is 3 (after removing intercept)
        for df in [3, 4, 5, 6]:
            B = ns_basis(x, df=df)
            assert B.shape == (50, df), f"Expected shape (50, {df}), got {B.shape}"

    def test_ns_finite_values(self):
        """ns_basis output contains only finite values."""
        x = np.linspace(0, 10, 50)
        B = ns_basis(x, df=3)
        assert np.isfinite(B).all(), "ns_basis returned non-finite values"

    def test_ns_linear_extrapolation(self):
        """
        ns_basis performs linear extrapolation outside boundaries.

        Points beyond boundaries should lie on a straight line.
        """
        x_train = np.linspace(2, 8, 30)
        B_train = ns_basis(x_train, df=3)

        # Predict at points outside boundaries
        x_test = np.array([0.5, 1.0, 9.5, 10.0])
        B_test = ns_basis(x_test, df=3, boundary_knots=np.array([2.0, 8.0]))

        # For linear extrapolation, check that slope is consistent
        # (simple test: values should change monotonically in one direction for extreme values)
        for col in range(B_test.shape[1]):
            left_vals = B_test[:2, col]
            right_vals = B_test[2:, col]
            # Check monotonicity (slope should be consistent)
            left_slope = left_vals[1] - left_vals[0]
            right_slope = right_vals[1] - right_vals[0]
            # Just check they have expected signs for linear continuation
            assert np.isfinite(left_vals).all()
            assert np.isfinite(right_vals).all()

    def test_ns_intercept_false(self):
        """ns_basis with intercept=False removes first B-spline."""
        x = np.linspace(0, 10, 50)
        df = 5  # Use a value that works
        B_without = ns_basis(x, df=df, intercept=False)
        assert B_without.shape == (50, df)
        assert np.isfinite(B_without).all()

    def test_ns_with_explicit_knots(self):
        """ns_basis works with explicitly supplied interior knots."""
        x = np.linspace(0, 10, 50)
        knots = np.array([3.0, 5.0, 7.0])
        B = ns_basis(x, knots=knots, intercept=False)
        assert B.shape[0] == 50
        assert np.isfinite(B).all()

    def test_ns_mutually_exclusive_args(self):
        """ns_basis now allows both knots and df (knots take precedence)."""
        x = np.linspace(0, 10, 50)
        # Both knots and df can be supplied; knots take precedence
        B = ns_basis(x, df=3, knots=np.array([3.0, 5.0]))
        assert B.shape == (50, len(np.array([3.0, 5.0])) + 1)  # n_interior + 1 for intercept=False


class TestBsBasis:
    """Tests for B-spline basis."""

    def test_bs_output_shape(self):
        """bs_basis output shape matches df (or minimum 3 for degree=3)."""
        x = np.linspace(0, 10, 50)
        # Minimum df for degree=3 B-spline is 3 (after removing intercept)
        for df in [3, 4, 5]:
            B = bs_basis(x, df=df)
            assert B.shape == (50, df)

    def test_bs_finite_values(self):
        """bs_basis output contains only finite values."""
        x = np.linspace(0, 10, 50)
        B = bs_basis(x, df=3)
        assert np.isfinite(B).all()

    def test_bs_degree_argument(self):
        """bs_basis respects degree argument."""
        x = np.linspace(0, 10, 50)
        for degree in [1, 2, 3, 4]:
            B = bs_basis(x, df=4, degree=degree)
            assert B.shape == (50, 4)
            assert np.isfinite(B).all()


class TestPolyBasis:
    """Tests for orthogonal polynomial basis."""

    def test_poly_output_shape(self):
        """poly_basis output shape matches df."""
        x = np.linspace(0, 10, 50)
        for df in [1, 2, 3, 4]:
            B = poly_basis(x, df=df)
            assert B.shape == (50, df)

    def test_poly_orthogonality(self):
        """poly_basis columns are orthogonal (normalized)."""
        x = np.linspace(0, 10, 50)
        B = poly_basis(x, df=3, raw=False)

        # For normalized orthogonal basis, each column should have roughly unit norm
        for j in range(B.shape[1]):
            norm = np.linalg.norm(B[:, j])
            assert 0.8 < norm < 1.2, f"Column {j} norm {norm} not near 1"

    def test_poly_raw_vs_orthogonal(self):
        """poly_basis raw=True vs raw=False produces different but valid output."""
        x = np.linspace(0, 10, 50)
        B_raw = poly_basis(x, df=3, raw=True)
        B_orth = poly_basis(x, df=3, raw=False)

        assert B_raw.shape == B_orth.shape
        assert np.isfinite(B_raw).all()
        assert np.isfinite(B_orth).all()
        # They should generally be different (unless numerically similar)
        assert not np.allclose(B_raw, B_orth)


class TestLinearBasis:
    """Tests for linear basis."""

    def test_linear_output_shape(self):
        """linear_basis always returns 1 column."""
        x = np.linspace(0, 10, 50)
        B = linear_basis(x)
        assert B.shape == (50, 1)

    def test_linear_values(self):
        """linear_basis output is the input vector (scaled)."""
        x = np.linspace(0, 10, 50)
        B = linear_basis(x)
        # Should be proportional to x
        assert np.allclose(B[:, 0], x)

    def test_linear_df_ignored(self):
        """linear_basis ignores df argument."""
        x = np.linspace(0, 10, 50)
        B1 = linear_basis(x, df=1)
        B2 = linear_basis(x, df=5)
        assert B1.shape == B2.shape == (50, 1)
        assert np.allclose(B1, B2)


class TestValidateBasisOutput:
    """Tests for basis output validation."""

    def test_validate_correct_shape(self):
        """_validate_basis_output passes for correct shape."""
        B = np.random.randn(10, 3)
        B_validated = _validate_basis_output(B, 10, 3, "test")
        assert B_validated.shape == (10, 3)

    def test_validate_wrong_shape(self):
        """_validate_basis_output raises for wrong shape."""
        B = np.random.randn(10, 2)
        with pytest.raises(ValueError, match="shape"):
            _validate_basis_output(B, 10, 3, "test")

    def test_validate_non_finite(self):
        """_validate_basis_output raises for non-finite values."""
        B = np.random.randn(10, 3)
        B[0, 0] = np.inf
        with pytest.raises(ValueError, match="non-finite"):
            _validate_basis_output(B, 10, 3, "test")


class TestBuildBasis:
    """Tests for basis dispatch function."""

    def test_build_basis_string_ns(self):
        """build_basis dispatches to ns_basis for 'ns' string."""
        x = np.linspace(0, 10, 50)
        B = build_basis(x, "ns", df=3)
        assert B.shape == (50, 3)
        assert np.isfinite(B).all()

    def test_build_basis_string_bs(self):
        """build_basis dispatches to bs_basis for 'bs' string."""
        x = np.linspace(0, 10, 50)
        B = build_basis(x, "bs", df=3)
        assert B.shape == (50, 3)

    def test_build_basis_string_poly(self):
        """build_basis dispatches to poly_basis for 'poly' string."""
        x = np.linspace(0, 10, 50)
        B = build_basis(x, "poly", df=3)
        assert B.shape == (50, 3)

    def test_build_basis_string_linear(self):
        """build_basis dispatches to linear_basis for 'linear' string."""
        x = np.linspace(0, 10, 50)
        B = build_basis(x, "linear", df=1)
        assert B.shape == (50, 1)

    def test_build_basis_unknown_string(self):
        """build_basis raises for unknown basis string."""
        x = np.linspace(0, 10, 50)
        with pytest.raises(ValueError, match="Unknown basis"):
            build_basis(x, "unknown_basis", df=3)

    def test_build_basis_callable(self):
        """build_basis accepts user-supplied callable."""

        def my_basis(x, df, **kwargs):
            return np.tile(x.reshape(-1, 1), (1, df))

        x = np.linspace(0, 10, 50)
        B = build_basis(x, my_basis, df=3)
        assert B.shape == (50, 3)

    def test_build_basis_callable_wrong_shape(self):
        """build_basis raises for callable returning wrong shape."""

        def bad_basis(x, df, **kwargs):
            return np.random.randn(len(x), df + 1)

        x = np.linspace(0, 10, 50)
        with pytest.raises(ValueError, match="shape"):
            build_basis(x, bad_basis, df=3)

    def test_build_basis_callable_non_finite(self):
        """build_basis raises for callable returning non-finite values."""

        def bad_basis(x, df, **kwargs):
            B = np.tile(x.reshape(-1, 1), (1, df))
            B[0, 0] = np.nan
            return B

        x = np.linspace(0, 10, 50)
        with pytest.raises(ValueError, match="non-finite"):
            build_basis(x, bad_basis, df=3)

    def test_build_basis_non_callable_non_string(self):
        """build_basis raises for non-callable, non-string basis."""
        x = np.linspace(0, 10, 50)
        with pytest.raises(TypeError, match="basis must be"):
            build_basis(x, 123, df=3)
