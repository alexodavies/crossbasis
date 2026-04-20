# CrossBasis: Python DLNM Package

A pure-Python implementation of Distributed Lag Non-linear Models (DLNMs) based on Gasparrini, Armstrong & Kenward (2010), *Statistics in Medicine*, 29:2224–2234.

## Features

- **Clean sklearn-style API**: `CrossBasis` transformer + `CrossPred` high-level class
- **User-supplied basis functions**: Expose both exposure and lag dimensions to custom callars, or use built-in natural cubic splines, B-splines, polynomials, and linear bases
- **DataFrame-aware**: Full support for pandas Series and DataFrame inputs/outputs
- **Explicit centering**: Required centering argument prevents silent reference-value errors
- **Rich visualization**: 3D surfaces, lag-response slices, and cumulative effect curves
- **R compatibility**: Numerical outputs match R's `dlnm` package

## Installation

```bash
pip install crossbasis
```

## Quick Start

```python
import numpy as np
import pandas as pd
from crossbasis import CrossPred

# Create sample data
temperature = np.random.randn(365) * 10 + 15  # daily temperature
deaths = np.random.poisson(100, 365)           # daily deaths

# Fit DLNM
cp = CrossPred(
    var_basis="ns", var_df=5,      # exposure: natural cubic spline, 5 df
    lag_basis="ns", lag_df=4,      # lag: natural cubic spline, 4 df
    max_lag=30,                     # up to 30 days of lag
)

# Get cross-basis matrix for GLM
W = cp.fit(temperature)

# Fit your own GLM (here using statsmodels)
import statsmodels.api as sm

y = sm.add_constant(np.column_stack([W, confounders]))  # add intercept and confounders
model = sm.GLM(deaths, y, family=sm.families.Poisson()).fit()

# Predict at specific temperatures
result = cp.predict(
    model,
    at=np.arange(-5, 35),  # temperatures from -5 to 35°C
    cen=20,                 # reference temperature 20°C
)

# Access predictions
print(result.allRR)  # cumulative relative risk

# Visualize
result.plot_overall()                  # cumulative effect
result.plot_slice(var=25)               # lag response at 25°C
result.plot_slice(lag=7)                # exposure response at 7-day lag
result.plot_3d()                        # 3D surface
```

## Documentation

### Core Classes

#### `CrossPred` (High-level API)

Main entry point for users. Manages fitting and prediction.

```python
cp = CrossPred(
    var_basis="ns",           # "ns", "bs", "poly", "linear", or callable
    var_df=5,                 # degrees of freedom for exposure basis
    lag_basis="ns",
    lag_df=4,
    max_lag=30,               # required
    na_action="drop",         # handle lag padding: "drop", "fill_zero", "fill_mean", or number
    cen="median",             # centering: float, "median", "mean", "minimum_risk"
)

# Fit and get cross-basis matrix for GLM
W = cp.fit(exposure_data)

# Predict relative risk
result = cp.predict(model, at=exposure_values, cen=reference_value)
```

#### `PredictionResult`

Returned by `predict()`. Contains predictions and plotting methods.

```python
# Access predictions
result.matfit          # log-RR at each (exposure, lag)
result.allfit          # cumulative log-RR
result.RR              # exponentiated (relative risk scale)
result.allRR           # cumulative relative risk

# Plotting
result.plot_3d()       # 3D surface
result.plot_slice(var=20)   # lag-response at exposure=20
result.plot_overall()  # cumulative effect

# Export to DataFrame
df = result.to_frame()  # long format for ggplot or other viz
```

#### `CrossBasis` (Power Users)

Lower-level transformer if you need fine-grained control or want to use the cross-basis matrix outside statsmodels.

```python
from crossbasis import CrossBasis

cb = CrossBasis(
    var_basis="ns", var_df=5,
    lag_basis="ns", lag_df=4,
    max_lag=30,
)

# Fit and transform
W = cb.fit_transform(exposure_data)

# Use W in any regression framework
# (statsmodels, sklearn, PyMC, etc.)
```

### Built-in Basis Functions

- **`ns`** (default): Natural cubic spline, matching R's `splines::ns()`
- **`bs`**: B-spline
- **`poly`**: Orthogonal polynomial
- **`linear`**: Linear term (single column)

Or supply your own:

```python
def my_basis(x, df, **kwargs):
    # x: input vector
    # df: number of basis functions
    # return: (n, df) float64 array, no NaNs/Infs
    return basis_matrix

cp = CrossPred(var_basis=my_basis, var_df=3, ...)
```

### Centering Parameter

The `cen` argument controls the reference exposure value:

| Value | Behavior |
|-------|----------|
| `float` | Use that value as reference |
| `"median"` (default) | Median of training exposure |
| `"mean"` | Mean of training exposure |
| `"minimum_risk"` | Approximate minimum of cumulative effect (experimental) |

## Dependencies

Required:
- `numpy>=1.24`
- `scipy>=1.10`
- `statsmodels>=0.14` (for fitting; not needed if you use coef/vcov directly)
- `matplotlib>=3.7` (for plotting)
- `pandas>=1.5` (for DataFrame support)

Optional (dev):
- `pytest`, `pytest-cov` (testing)
- `rpy2` (R validation, if comparing against R's dlnm package)

## Testing

```bash
pytest tests/
```

Tests cover basis functions, cross-basis matrix construction, and prediction output shapes and centering behavior.

## References

Gasparrini A, Armstrong B, Kenward MG (2010). Distributed lag non-linear models. *Statistics in Medicine* 29:2224–2234.
https://doi.org/10.1002/sim.3940

## License

MIT
