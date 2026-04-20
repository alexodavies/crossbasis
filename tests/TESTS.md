# Test coverage reference

---

## test_basis.py

### `TestNsBasis` — natural cubic spline basis

| Test | What we're checking |
|------|---------------------|
| `test_ns_output_shape` | Output shape is `(n, df)` for df in [3, 4, 5, 6] |
| `test_ns_finite_values` | No NaNs or Infs in output |
| `test_ns_linear_extrapolation` | Points outside boundary knots are linearly extrapolated (not cubic-continued); values are finite |
| `test_ns_intercept_false` | With `intercept=False`, output shape is still `(n, df)` and all finite |
| `test_ns_with_explicit_knots` | Explicit interior `knots` argument accepted; shape and finiteness hold |
| `test_ns_mutually_exclusive_args` | When both `knots` and `df` are supplied, `knots` takes precedence; output columns = `n_interior + 1` (intercept=False) |

### `TestBsBasis` — B-spline basis

| Test | What we're checking |
|------|---------------------|
| `test_bs_output_shape` | Output shape is `(n, df)` for df in [3, 4, 5] |
| `test_bs_finite_values` | No NaNs or Infs in output |
| `test_bs_degree_argument` | `degree` kwarg is respected; shape and finiteness hold for degrees 1–4 |

### `TestPolyBasis` — orthogonal polynomial basis

| Test | What we're checking |
|------|---------------------|
| `test_poly_output_shape` | Output shape is `(n, df)` for df in [1, 2, 3, 4] |
| `test_poly_orthogonality` | Each column has near-unit norm (normalised basis) |
| `test_poly_raw_vs_orthogonal` | `raw=True` and `raw=False` produce different but valid outputs of the same shape |

### `TestLinearBasis` — single linear term

| Test | What we're checking |
|------|---------------------|
| `test_linear_output_shape` | Output is always `(n, 1)` |
| `test_linear_values` | Output column equals input `x` |
| `test_linear_df_ignored` | `df` argument has no effect on shape or values |

### `TestValidateBasisOutput` — validation wrapper

| Test | What we're checking |
|------|---------------------|
| `test_validate_correct_shape` | Passes silently when shape matches |
| `test_validate_wrong_shape` | Raises `ValueError` mentioning "shape" when shape mismatches |
| `test_validate_non_finite` | Raises `ValueError` mentioning "non-finite" when output contains Inf/NaN |

### `TestBuildBasis` — dispatch function

| Test | What we're checking |
|------|---------------------|
| `test_build_basis_string_ns` | `"ns"` string dispatches to `ns_basis`; shape and finiteness |
| `test_build_basis_string_bs` | `"bs"` string dispatches to `bs_basis` |
| `test_build_basis_string_poly` | `"poly"` string dispatches to `poly_basis` |
| `test_build_basis_string_linear` | `"linear"` string dispatches to `linear_basis` |
| `test_build_basis_unknown_string` | Unknown string raises `ValueError` matching "Unknown basis" |
| `test_build_basis_callable` | User-supplied callable is accepted and used |
| `test_build_basis_callable_wrong_shape` | Callable returning wrong shape raises `ValueError` |
| `test_build_basis_callable_non_finite` | Callable returning NaN raises `ValueError` |
| `test_build_basis_non_callable_non_string` | Non-string, non-callable raises `TypeError` |

---

## test_crossbasis.py

### `TestCrossBasisInit` — constructor validation

| Test | What we're checking |
|------|---------------------|
| `test_max_lag_required` | Raises `ValueError` if `max_lag` not supplied |
| `test_mutually_exclusive_var_knots_df` | Raises if both `var_knots` and `var_df` are set |
| `test_mutually_exclusive_lag_knots_df` | Raises if both `lag_knots` and `lag_df` are set |
| `test_init_valid` | Constructor succeeds and stores args correctly |

### `TestCrossBasisFit` — `fit()` method

| Test | What we're checking |
|------|---------------------|
| `test_fit_sets_is_fitted` | `is_fitted_` is False before fit, True after |
| `test_fit_computes_var_knots` | `var_knots_` is populated after fit when `var_df` given |
| `test_fit_computes_lag_knots_log` | `lag_knots_` is populated with log-scale knots |
| `test_fit_computes_lag_knots_equal` | `lag_knots_` is populated with equal-scale knots |
| `test_fit_stores_boundary_knots` | `boundary_knots_` is `[min(X), max(X)]` |
| `test_fit_returns_self` | `fit()` returns `self` for chaining |

### `TestCrossBasisTransform` — `transform()` method

| Test | What we're checking |
|------|---------------------|
| `test_transform_requires_fit` | Raises `ValueError` if called before `fit()` |
| `test_transform_output_shape` | Output is `(n, v_x * v_l)` — here 3 × 4 = 12 columns |
| `test_transform_nan_rows_drop` | With `na_action="drop"`, first `max_lag` rows contain NaN |
| `test_transform_nan_rows_fill_zero` | With `na_action="fill_zero"`, no NaNs in output |
| `test_transform_nan_rows_fill_mean` | With `na_action="fill_mean"`, no NaNs in output |
| `test_transform_nan_rows_fill_numeric` | With numeric `na_action`, no NaNs in output |
| `test_transform_finite_values` | With NaNs filled, all values are finite |
| `test_transform_numpy_input` | Accepts numpy array; returns numpy array |
| `test_transform_series_input` | Accepts pandas Series; returns DataFrame |
| `test_transform_dataframe_input` | Accepts DataFrame column; returns DataFrame |

### `TestCrossBasisFitTransform` — `fit_transform()` method

| Test | What we're checking |
|------|---------------------|
| `test_fit_transform_output_shape` | Output shape is `(n, v_x * v_l)` |
| `test_fit_transform_sets_fitted` | `is_fitted_` is True after call |
| `test_fit_transform_matches_fit_then_transform` | `fit_transform(x)` is numerically identical to `fit(x).transform(x)` |

### `TestCrossBasisCustomBasis` — user-supplied callables

| Test | What we're checking |
|------|---------------------|
| `test_custom_var_basis` | Custom callable for `var_basis` is applied; output shape correct |
| `test_custom_lag_basis` | Custom callable for `lag_basis` is applied; output shape correct |

---

## test_crosspred.py

### `TestPredictionResultDataclass` — `PredictionResult` dataclass

| Test | What we're checking |
|------|---------------------|
| `test_prediction_result_creation` | Dataclass can be instantiated; fields have expected shapes and values |
| `test_to_frame` | `to_frame()` returns a long-format DataFrame with columns `exposure`, `lag`, `logRR`, `se` and correct row count (`n_exp × n_lag`) |

### `TestCrossPredInit` — `CrossPred` constructor

| Test | What we're checking |
|------|---------------------|
| `test_init_defaults` | Default `cen="median"` and internal `_crossbasis` is set |
| `test_init_custom_cen` | String `cen` values are stored as-is |
| `test_init_numeric_cen` | Numeric `cen` values are stored as-is |

### `TestCrossPredFit` — `fit()` method

| Test | What we're checking |
|------|---------------------|
| `test_fit_returns_W` | Returns cross-basis matrix `W` with correct shape `(n, v_x * v_l)` |
| `test_fit_stores_W_internally` | `_W` attribute is set after fit |
| `test_fit_stores_X_train` | `_X_train` equals the fitted exposure array |

### `TestCrossPredResolveCen` — centering resolution

| Test | What we're checking |
|------|---------------------|
| `test_resolve_cen_float` | Float passes through unchanged |
| `test_resolve_cen_median` | `"median"` resolves to `np.median(X)` |
| `test_resolve_cen_mean` | `"mean"` resolves to `np.mean(X)` |
| `test_resolve_cen_minimum_risk` | `"minimum_risk"` returns `None` (deferred to predict) |
| `test_resolve_cen_invalid_string` | Unrecognised string raises `ValueError` matching "cen must be" |

### `TestCrossPredPredict` — `predict()` method

| Test | What we're checking |
|------|---------------------|
| `test_predict_requires_at` | Raises `ValueError` if `at` not supplied |
| `test_predict_output_shapes` | `matfit`, `matse` are `(n_exp, max_lag+1)`; `allfit`, `allse` are `(n_exp,)` |
| `test_predict_cen_float` | At the centering value, `allRR` is 1.0 |
| `test_predict_cen_median` | With `cen="median"`, `allRR` is 1.0 at the training median |
| `test_predict_minimum_risk_warning` | `cen="minimum_risk"` emits a `UserWarning` containing "minimum_risk" |
| `test_predict_coef_vcov_alternative` | `coef` + `vcov` can be passed directly instead of a statsmodels model |

### `TestPredictionResultPlots` — plot methods

| Test | What we're checking |
|------|---------------------|
| `test_plot_3d_runs` | `plot_3d()` completes without error and returns `(fig, ax)` |
| `test_plot_slice_var` | `plot_slice(var=...)` completes without error |
| `test_plot_slice_lag` | `plot_slice(lag=...)` completes without error |
| `test_plot_slice_requires_var_or_lag` | Raises `ValueError` if neither `var` nor `lag` supplied |
| `test_plot_slice_excludes_both_var_lag` | Raises `ValueError` if both `var` and `lag` supplied |
| `test_plot_overall_runs` | `plot_overall()` completes without error |
| `test_plot_customization_kwargs` | Color, alpha, linewidth kwargs accepted by all three plot methods |

---

## test_r_validation.py

All tests in this file require `rpy2` and a working R installation. The whole file is skipped automatically if `rpy2` is not importable.

### `TestNsBasisNumerical` — ns_basis vs R's `splines::ns()`

| Test | What we're checking |
|------|---------------------|
| `test_ns_values_match_r` | Column-wise RMSE (allowing sign flips and column reordering) between Python and R basis matrices is < 0.15; shapes match exactly. Skips if R/splines unavailable; fails if RMSE too high. |

### `TestCrossBasisNumerical` — CrossBasis with a real GLM

| Test | What we're checking |
|------|---------------------|
| `test_crossbasis_with_glm_fit` | CrossBasis matrix can be passed to a Poisson GLM; coefficient vector has the right length (1 intercept + v_x × v_l); all params and covariance entries are finite |

### `TestDLNMPredictionNumerical` — end-to-end prediction correctness

| Test | What we're checking |
|------|---------------------|
| `test_prediction_reproducibility` | Two identical `predict()` calls return bitwise-identical `allRR` and `matfit` |
| `test_prediction_centering_consistency` | `"median"`, `"mean"`, and explicit float centering all produce `allRR = 1.0` at their reference values |
| `test_prediction_se_validity` | All standard errors are positive; cumulative SEs are below 1.0 for a weak simulated signal |
| `test_lagresponse_vs_cumulative` | `allRR` equals `exp(sum(matfit))` for a linear × linear model (verifies Eqs. 11–12 are not summing on the RR scale) |
| `test_varying_max_lag` | `matfit.shape[1]` equals `max_lag + 1` for max_lag in [7, 14, 21] |
| `test_matrix_shapes_consistency` | All output arrays (`matfit`, `matse`, `RR`, `RR_low`, `RR_high`, `allfit`, `allse`, `allRR`, `allRR_low`, `allRR_high`) have consistent shapes `(n_exp, n_lag)` or `(n_exp,)` |
