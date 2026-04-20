# Build specification: Python DLNM package

## Background

Implement a Python package for Distributed Lag Non-linear Models (DLNMs) based on
Gasparrini, Armstrong & Kenward (2010), *Statistics in Medicine*, 29:2224–2234
(hereafter "the paper"). All equation numbers refer to that paper.

A mature functional-style Python port already exists (`dlnm` on PyPI by Victorivus).
This package differentiates itself with:
- A clean sklearn-style OOP API (`CrossBasis` transformer + `CrossPred` high-level class)
- `CrossPred` owns the full pipeline internally — users never manage `CrossBasis` directly
  unless they want to
- User-supplied callable basis functions for both exposure and lag dimensions
- DataFrame-aware inputs and outputs throughout
- Explicit, required centering argument to prevent silent reference-value errors
- User-specifiable missing value strategy

Dependencies: `numpy>=1.24`, `scipy>=1.10`, `statsmodels>=0.14`, `matplotlib>=3.7`,
`pandas>=1.5`. No R, no rpy2.

---

## Repository layout

```
pydlnm/                         # repo root (choose your own package name)
├── pydlnm/
│   ├── __init__.py             # public exports
│   ├── basis.py                # basis function builders (ns, bs, poly, linear + callable)
│   ├── crossbasis.py           # CrossBasis transformer
│   ├── crosspred.py            # CrossPred high-level class + PredictionResult dataclass
│   └── utils.py                # logknots, equalknots, input coercion, validation helpers
├── tests/
│   ├── test_basis.py
│   ├── test_crossbasis.py
│   └── test_crosspred.py
├── pyproject.toml
└── README.md
```

---

## Mathematical reference

All equation numbers below refer to Gasparrini et al. (2010).

| Symbol | Meaning |
|--------|---------|
| `n` | number of time-series observations |
| `x` | exposure vector, length n |
| `L` | maximum lag (integer) |
| `Q` | n × (L+1) matrix of lagged exposures — Eq. 3 |
| `Z` | n × v_x basis matrix for the exposure dimension — Eq. 2 |
| `C` | (L+1) × v_l basis matrix for the lag dimension — Eq. 4 |
| `W` | n × (v_x · v_l) cross-basis matrix — Eq. 5 / Eq. 8 |
| `g_hat` | fitted coefficient vector, length v_x · v_l |
| `V(g_hat)` | covariance matrix of g_hat |

### Key equations

**Eq. 2** — basis expansion for exposure:
`s(x_t; b) = z_t^T · b`
where `z_t` is the t-th row of the n × v_x basis matrix Z.

**Eq. 3** — lag matrix construction:
`q_t = [x_t, x_{t-1}, ..., x_{t-L}]^T`
producing the n × (L+1) matrix Q. Rows with insufficient lag history contain NaN.

**Eq. 4** — DLM in matrix form (special case of DLNM):
`s(x_t; g) = q_t^T · C · g`
where C is the (L+1) × v_l lag basis matrix applied to the lag vector
`λ = [0, 1, ..., L]^T`.

**Eq. 5** — transformed variable matrix:
`W = Q · C`

**Eq. 6** — back-transformation of parameters:
`b_hat = C · g_hat`
`V(b_hat) = C · V(g_hat) · C^T`

**Eq. 7** — general DLNM:
`s(x_t; g) = Σ_j Σ_k r_{tj}^T · c_{·k} · γ_{jk} = w_t^T · g`
summing over v_x exposure basis functions j and v_l lag basis functions k.

**Eq. 8** — tensor product construction of W (the core of the cross-basis):
```
A_dot = (1^T ⊗ R_dot) ⊙ (1 ⊗ P_{1,3}(C) ⊗ 1^T)
```
where ⊗ is the Kronecker product, ⊙ is the Hadamard product, and P_{1,3} permutes
array indices. W is obtained by summing A_dot along its third (lag) dimension.
In practice: for each row t, W[t, :] is the Kronecker product of Z[t, :] and
the corresponding row of (Q[t, :] @ C), reshaped to v_x · v_l.

**Eqs. 9–10** — prediction at each lag ℓ:
`e_{·ℓ} = A^p_{··} · g_hat`
`e^sd_{·ℓ} = sqrt(diag(A^p_{··} · V(g_hat) · A^{pT}_{··}))`

**Eqs. 11–12** — overall cumulative effect:
`e^tot = W^p · g_hat`
`e^sd_{tot} = sqrt(diag(W^p · V(g_hat) · W^{pT}))`

---

## Module: `basis.py`

### Basis function protocol

Every basis function — built-in or user-supplied — must satisfy:

```python
def my_basis(x: np.ndarray, df: int, **kwargs) -> np.ndarray:
    """
    Parameters
    ----------
    x : 1-D numpy array of length m (may include prediction points outside
        the training range — extrapolation behaviour matters)
    df : number of basis columns (degrees of freedom)
    **kwargs : basis-specific options (knots, degree, intercept, etc.)

    Returns
    -------
    np.ndarray of shape (m, df), dtype float64, no NaNs or Infs
    """
```

### Validation wrapper

Wrap all callable bases (built-in and user-supplied) in `_validate_basis_output`:

```python
def _validate_basis_output(B: np.ndarray, m: int, df: int, name: str) -> np.ndarray:
    if B.shape != (m, df):
        raise ValueError(f"Basis '{name}' returned shape {B.shape}, expected ({m}, {df})")
    if not np.issubdtype(B.dtype, np.floating):
        B = B.astype(np.float64)
    if not np.isfinite(B).all():
        raise ValueError(f"Basis '{name}' returned non-finite values")
    return B
```

### Built-in bases

#### `ns` — natural cubic spline

This is the most important basis. It **must** match R's `splines::ns()` exactly,
including two non-obvious behaviours documented below.

Implementation notes (critical for R compatibility):

**1. Linear extrapolation outside boundary knots**

R's `ns()` enforces zero second derivative at the boundaries, which means linear
extrapolation beyond them. `scipy.interpolate.BSpline(extrapolate=True)` continues
the cubic polynomial instead — producing completely different values outside the
training range. Since `CrossPred.predict()` commonly evaluates at exposure values
outside the training range, this must be fixed explicitly.

After computing the spline basis values, replace any points outside
`[boundary_knots[0], boundary_knots[1]]` with linear extrapolation anchored at
the boundary value and first derivative:

```python
for i in range(n_basis):
    spl_i = BSpline(...)  # i-th basis spline
    left_mask  = x < boundary_knots[0]
    right_mask = x > boundary_knots[1]

    if left_mask.any():
        val_l   = spl_i(boundary_knots[0])
        slope_l = spl_i(boundary_knots[0], 1)
        basis[left_mask, i] = val_l + slope_l * (x[left_mask] - boundary_knots[0])

    if right_mask.any():
        val_r   = spl_i(boundary_knots[1])
        slope_r = spl_i(boundary_knots[1], 1)
        basis[right_mask, i] = val_r + slope_r * (x[right_mask] - boundary_knots[1])
```

**2. `intercept=False` constraint row**

R excludes the leftmost B-spline (B-spline 0) by adding the unit vector
`e0 = [1, 0, 0, …]` as an extra constraint row before the QR null-space
projection — **not** an all-ones row. Failure to replicate this produces a
different basis matrix even within the boundary knot range.

```python
if not intercept:
    e0 = np.zeros((1, n_basis))
    e0[0, 0] = 1.0
    const = np.vstack([const, e0])
# then project out the null space of `const` via QR decomposition
```

Signature: `ns_basis(x, df=None, knots=None, boundary_knots=None, intercept=False)`

If `df` is given and `knots` is None, place `df - 1 - int(not intercept)` interior
knots at equally-spaced quantiles of x (matching R's default).

#### `bs` — B-spline

Standard B-spline via `scipy.interpolate.BSpline`. Supports `degree` (default 3),
`knots`, `df`, `intercept`.

Signature: `bs_basis(x, df=None, knots=None, degree=3, boundary_knots=None, intercept=False)`

#### `poly` — orthogonal polynomial

Use `numpy.polynomial.polynomial.polyvander` normalised to unit variance per column.

Signature: `poly_basis(x, df, raw=False)`

#### `linear` — single linear term

Returns a single-column matrix of `x` (or scaled x). `df` is ignored (always 1).

Signature: `linear_basis(x, **kwargs)`

### Dispatch function

```python
_BUILTIN_BASES = {"ns": ns_basis, "bs": bs_basis, "poly": poly_basis, "linear": linear_basis}

def build_basis(x: np.ndarray, basis, df: int, **kwargs) -> np.ndarray:
    """
    Dispatch to built-in or user-supplied basis function.

    Parameters
    ----------
    basis : str or callable
        String key from _BUILTIN_BASES, or any callable satisfying the basis protocol.
    """
    if callable(basis):
        name = getattr(basis, "__name__", repr(basis))
        B = basis(x, df, **kwargs)
    elif isinstance(basis, str):
        if basis not in _BUILTIN_BASES:
            raise ValueError(f"Unknown basis '{basis}'. Choose from {list(_BUILTIN_BASES)} or pass a callable.")
        name = basis
        B = _BUILTIN_BASES[basis](x, df, **kwargs)
    else:
        raise TypeError(f"basis must be str or callable, got {type(basis)}")
    return _validate_basis_output(B, len(x), df, name)
```

---

## Module: `utils.py`

### Input coercion

```python
def to_array(x) -> np.ndarray:
    """Accept numpy array, pandas Series, or pandas DataFrame column. Return 1-D float64 array."""
```

### Knot placement

```python
def logknots(max_lag: int, nk: int) -> np.ndarray:
    """
    Place nk interior knots on a log scale over [0, max_lag].
    Matches R dlnm's logknots(). Recommended default for lag dimension.
    """

def equalknots(x: np.ndarray, nk: int) -> np.ndarray:
    """Place nk interior knots at equally-spaced quantiles of x."""
```

### Lag matrix construction

```python
def build_lag_matrix(x: np.ndarray, max_lag: int) -> np.ndarray:
    """
    Construct the n × (max_lag + 1) matrix Q from Eq. 3.
    Q[t, l] = x[t - l], with np.nan where t - l < 0.
    """
```

---

## Module: `crossbasis.py`

### `CrossBasis`

An sklearn-style transformer. Power users who need W directly (e.g. for custom
modelling outside statsmodels) use this class. `CrossPred` uses it internally.

```python
class CrossBasis:
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
        max_lag=None,         # required
        lag_knot_scale="log", # "log" (default, per paper Section 5) or "equal"
        na_action="drop",     # "drop", "fill_zero", "fill_mean", or numeric fill value
    ):
```

**Argument rules:**
- `max_lag` is required; raise `ValueError` if None.
- `var_knots` and `var_df` are mutually exclusive; raise `ValueError` if both supplied.
- `lag_knots` and `lag_df` are mutually exclusive; raise `ValueError` if both supplied.
- `var_basis` and `lag_basis` each accept a string shortcut or any callable satisfying
  the basis protocol.

**Methods:**

`fit(X) -> self`
- Coerce X to numpy via `to_array`.
- If `var_knots` is None and `var_df` is given, compute knots from quantiles of X
  (ignoring NaNs). Store as `self.var_knots_`.
- If `lag_knots` is None and `lag_df` is given, compute lag knots via `logknots` or
  `equalknots` depending on `lag_knot_scale`. Store as `self.lag_knots_`.
- Store `self.boundary_knots_` = [min(X), max(X)] for extrapolation in predict.
- Mark `self.is_fitted_ = True`. Return `self`.

`transform(X) -> np.ndarray`
- Coerce X, build Q via `build_lag_matrix` (Eq. 3).
- Apply `na_action` to handle NaN rows from lag padding:
  - `"drop"` — identify and record row indices with any NaN in Q; these rows produce
    NaN in W and a warning is emitted with the count. The caller (CrossPred) handles
    dropping them before GLM fitting.
  - `"fill_zero"` — replace NaN with 0.
  - `"fill_mean"` — replace NaN with column mean of Q.
  - numeric — replace NaN with that value.
- Build Z via `build_basis(x, var_basis, var_df, knots=var_knots_, ...)` (Eq. 2).
- Build C via `build_basis(lag_vector, lag_basis, lag_df, knots=lag_knots_, ...)`
  where `lag_vector = np.arange(max_lag + 1)` (Eq. 4).
- Compute W as the row-wise Kronecker product of Z and (Q @ C), giving
  shape n × (v_x · v_l). This implements Eq. 8.
- Store `self.C_` and `self.Z_shape_` for use in CrossPred.
- Return W as a numpy array. If input was a DataFrame, column names are
  `["cb_{j}_{k}" for j in range(v_x) for k in range(v_l)]`.

`fit_transform(X) -> np.ndarray`
- Calls `fit(X)` then `transform(X)`.

**Attributes set after fit:**
- `self.var_knots_` — final knot positions for exposure basis
- `self.lag_knots_` — final knot positions for lag basis
- `self.boundary_knots_` — [min, max] of training exposure
- `self.C_` — the (L+1) × v_l lag basis matrix
- `self.is_fitted_`

---

## Module: `crosspred.py`

### `PredictionResult`

A dataclass holding all prediction outputs. Returned by `CrossPred.predict()`.

```python
@dataclass
class PredictionResult:
    # Core arrays
    matfit: np.ndarray    # shape (m, L+1) — log-RR at each (exposure value, lag), Eq. 9
    matse:  np.ndarray    # shape (m, L+1) — SE of matfit, Eq. 10
    allfit: np.ndarray    # shape (m,)     — cumulative log-RR over all lags, Eq. 11
    allse:  np.ndarray    # shape (m,)     — SE of allfit, Eq. 12

    # Exponentiated (relative risk scale)
    RR:     np.ndarray    # exp(matfit)
    RR_low: np.ndarray    # exp(matfit - 1.96 * matse)
    RR_high: np.ndarray   # exp(matfit + 1.96 * matse)
    allRR:      np.ndarray  # exp(allfit)
    allRR_low:  np.ndarray
    allRR_high: np.ndarray

    # Metadata
    predvar: np.ndarray   # exposure values used for prediction (the `at` argument)
    predlag: np.ndarray   # lag values [0, 1, ..., L]
    cen: float            # reference/centering value

    def to_frame(self) -> pd.DataFrame:
        """
        Return tidy long-format DataFrame with columns:
        exposure, lag, logRR, se, RR, RR_low, RR_high
        """
```

### Prediction grid construction (R compatibility critical)

The flat prediction matrix `Xpred` is built as follows — **order matters** and must
match R's convention exactly:

```python
# predvar varies fastest (inner loop), predlag slowest (outer loop)
varvec = np.tile(predvar, len(predlag))        # predvar repeated for each lag
lagvec = np.repeat(predlag, len(predvar))      # each lag repeated len(predvar) times
```

After computing `flat = Xpred @ coef`, reshape using Fortran (column-major) order:

```python
matfit = flat.reshape(len(predlag), len(predvar)).T
matse  = np.sqrt(diag_terms).reshape(len(predlag), len(predvar)).T
```

Failure to follow this ordering breaks both the allfit accumulation and the
matfit/matse reshape, producing silently incorrect results.

### `CrossPred`

The high-level user-facing class. Owns a `CrossBasis` instance internally.

```python
class CrossPred:
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
        max_lag=None,         # required
        lag_knot_scale="log",
        na_action="drop",
        cen="median",         # float, "median", "mean", or "minimum_risk"
    ):
        # Construct and store a CrossBasis instance with these arguments
        self._crossbasis = CrossBasis(...)
        self.cen = cen        # stored; resolved to float in fit() or predict()
```

**Centering parameter (`cen`) resolution**

`cen` controls the reference exposure value against which all log-RR predictions
are expressed. It can be set at construction and optionally overridden per
`predict()` call.

Accepted values:

| Value | Behaviour |
|-------|-----------|
| `float` or `int` | Used directly as the reference value |
| `"median"` | Median of the training exposure X, computed in `fit()`. **Default.** |
| `"mean"` | Mean of the training exposure X, computed in `fit()`. |
| `"minimum_risk"` | Approximate minimum of the cumulative log-RR curve — see below. |

Resolution logic:

```python
def _resolve_cen(self, cen, X):
    if isinstance(cen, (int, float)):
        return float(cen)
    elif cen == "median":
        return float(np.nanmedian(X))
    elif cen == "mean":
        return float(np.nanmean(X))
    elif cen == "minimum_risk":
        return None  # deferred — resolved inside predict() after preliminary pass
    else:
        raise ValueError(
            f"cen must be a float, 'median', 'mean', or 'minimum_risk'. Got: {cen!r}"
        )
```

**`"minimum_risk"` behaviour**

When `cen="minimum_risk"`, `predict()` performs a preliminary prediction over 200
equally-spaced points spanning the training exposure range, identifies the exposure
value minimising `allfit` (cumulative log-RR), and uses that as the centering value
for the final prediction. A `UserWarning` is always emitted:

> "cen='minimum_risk' is approximate and may be unreliable for monotone
>  exposure-response relationships. The minimum was found at {value:.2f}."

This option is documented as experimental. Users are encouraged to inspect the
overall effect curve and set `cen` explicitly if the minimum risk point matters
for their analysis.
```

**Methods:**

`fit(X) -> np.ndarray`
- Calls `self._crossbasis.fit_transform(X)`.
- Stores W internally as `self._W`.
- Emits a warning if NaN rows were produced, including how many.
- **Returns W** (the cross-basis matrix) so the user can include it in their GLM
  design matrix. This is the only time the user interacts with W directly.

Example usage:
```python
cp = CrossPred(var_basis="ns", var_df=5, lag_basis="ns", lag_df=4, max_lag=30)
W = cp.fit(temperature)  # user uses W to build their GLM design matrix
model = sm.GLM(deaths, np.column_stack([W, confounders]),
               family=sm.families.Poisson()).fit()
result = cp.predict(model, at=np.arange(-16, 34), cen=21)
```

`predict(model, at, cen=None) -> PredictionResult`
- `model`: a fitted `statsmodels` results object. Extract `model.params` (g_hat)
  and `model.cov_params()` (V(g_hat)). Only the columns/rows corresponding to the
  cross-basis terms are used — infer these from the position of W columns in the
  design matrix, or require the user to pass `coef` and `vcov` directly as an
  alternative signature.
- `at`: numpy array or list of exposure values for prediction. May include values
  outside the training range — extrapolation is handled by the basis functions.
- `cen`: float, `"median"`, `"mean"`, `"minimum_risk"`, or `None`. If `None`,
  uses the value set at construction (default `"median"`). If supplied here,
  overrides the constructor value for this prediction only. Resolved via
  `_resolve_cen()` — see centering parameter section above.
- Internally build the prediction basis matrix Z_p from `at` using the fitted
  `CrossBasis` (reusing stored knots and boundary knots).
- Build the prediction cross-basis W_p following the grid ordering rules above.
- Compute matfit, matse (Eqs. 9–10), allfit, allse (Eqs. 11–12).
- Centre all predictions by subtracting the prediction at `cen`.
- Return a `PredictionResult`.

Alternative signature for users who manage coef/vcov manually:
```python
cp.predict(coef=coef_array, vcov=vcov_matrix, at=..., cen=...)
```

**Plot methods** (on `PredictionResult`, not on `CrossPred` itself — keeps results
self-contained):

```python
result.plot_3d(
    fig=None, ax=None,       # optional existing figure/axes
    xlabel="Exposure",
    ylabel="Lag",
    zlabel="RR",
    title=None,
)

result.plot_slice(
    var=None,                # float: plot lag-response at this exposure value
    lag=None,                # int: plot exposure-response at this lag
    # exactly one of var or lag must be supplied
    ci=True,                 # show 95% CI ribbon
    reference_line=True,     # horizontal line at RR=1
    fig=None, ax=None,
)

result.plot_overall(
    ci=True,
    reference_line=True,
    xlabel="Exposure",
    ylabel="Overall RR",
    title=None,
    fig=None, ax=None,
)
```

---

## `__init__.py` public exports

```python
from .crossbasis import CrossBasis
from .crosspred import CrossPred, PredictionResult
from .basis import ns_basis, bs_basis, poly_basis, linear_basis
from .utils import logknots, equalknots
```

---

## Tests

Write tests covering:

**`test_basis.py`**
- `ns_basis` output shape matches expectation for a range of df values
- `ns_basis` linear extrapolation: values beyond boundary knots lie on a straight line
- `ns_basis` intercept=False: first column of output is not a unit vector
- `bs_basis` output shape and finite values
- `poly_basis` columns are orthogonal (dot product ≈ 0 for i ≠ j)
- User-supplied callable: valid callable passes, wrong output shape raises ValueError,
  non-finite output raises ValueError

**`test_crossbasis.py`**
- `CrossBasis` raises ValueError if `max_lag` not supplied
- `CrossBasis` raises ValueError if both `var_knots` and `var_df` supplied
- `fit_transform` output shape is n × (v_x · v_l)
- NaN rows in W correspond exactly to the first `max_lag` rows when `na_action="drop"`
- `fill_zero` produces no NaNs in W
- Accepts numpy array input
- Accepts pandas Series input
- Accepts pandas DataFrame column input

**`test_crosspred.py`**
- `CrossPred.fit` returns a matrix of the correct shape
- `CrossPred.predict` with default `cen="median"` produces RR=1.0 at the median of X
- `CrossPred.predict` with explicit float `cen` produces RR=1.0 at that value
- `CrossPred.predict` with `cen="minimum_risk"` emits a UserWarning
- `CrossPred.predict` raises ValueError if `cen` is an unrecognised string
- `PredictionResult.matfit` shape is (len(at), max_lag + 1)
- `PredictionResult.allfit` shape is (len(at),)
- RR at `cen` is 1.0 (within floating-point tolerance)
- `to_frame()` returns a DataFrame with expected columns
- Plot methods run without error on a small synthetic dataset

---

## R compatibility checklist

Before considering the implementation complete, verify numerical output against R's
`dlnm` package on the Chicago NMMAPS dataset (or any shared fixture). Key checks:

1. `ns_basis` output matches `splines::ns()` to within 1e-10, including at points
   outside the training range (linear extrapolation, not cubic continuation).

2. `ns_basis` with `intercept=False`: the constraint removes B-spline 0 via unit
   vector `e0`, not an all-ones row. Mismatched constraint produces different basis
   even within boundary knots.

3. Prediction grid ordering: `predvar` varies fastest, `predlag` slowest:
   ```python
   varvec = np.tile(predvar, len(predlag))
   lagvec = np.repeat(predlag, len(predvar))
   ```

4. `matfit`/`matse` reshape uses Fortran order to match R's column-major filling:
   ```python
   matfit = flat.reshape(len(predlag), len(predvar)).T
   matse  = np.sqrt(...).reshape(len(predlag), len(predvar)).T
   ```

Points 3 and 4 are silent bugs — the code runs and produces output, but the values
are assigned to the wrong (exposure, lag) combinations. Test explicitly.

---

## pyproject.toml

```toml
[build-system]
requires = ["setuptools>=68", "wheel"]
build-backend = "setuptools.backends.legacy:build"

[project]
name = "crossbasis"
version = "0.1.0"
requires-python = ">=3.9"
dependencies = [
    "numpy>=1.24",
    "scipy>=1.10",
    "statsmodels>=0.14",
    "matplotlib>=3.7",
    "pandas>=1.5",
]

[project.optional-dependencies]
dev = ["pytest", "pytest-cov", "ruff", "mypy"]
```

---

## Style notes for Claude Code

- Type-hint all public functions and methods.
- Docstrings on all public classes and methods; include parameter types, shapes,
  and references to equation numbers where relevant.
- Raise `ValueError` with clear messages for all invalid argument combinations.
- Internal helpers prefixed with `_`.
- No global state; all configuration flows through constructor arguments.
- Do not import R or rpy2 anywhere.
