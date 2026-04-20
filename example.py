"""
Synthetic temperature-mortality DLNM example.

Generates three years of daily data for a hypothetical city, simulates
Poisson mortality with a J-shaped temperature-mortality relationship,
then fits a DLNM and plots the overall exposure-response risk curve and
lag-specific effects at extreme temperatures.
"""

import numpy as np
import pandas as pd
import statsmodels.api as sm
import matplotlib.pyplot as plt

from crossbasis import CrossBasis, CrossPred

rng = np.random.default_rng(42)

# ── Simulate data ─────────────────────────────────────────────────────────────

n = 3 * 365  # three years of daily observations

day = np.arange(n)
temperature = 15 + 12 * np.sin(2 * np.pi * (day - 60) / 365) + rng.normal(0, 3, n)

# True instantaneous log-RR: J-shaped, centred at 18 °C
# Cold increases risk steeply; heat increases risk moderately
cen = 18.0
def true_log_rr(temp):
    delta = temp - cen
    return np.where(delta < 0, 0.003 * delta**2, 0.001 * delta**2)

# Base rate with weak direct effect + Poisson noise; no lag structure in DGP
# (the DLNM is deliberately misspecified to show what it recovers)
log_mu = np.log(40) + true_log_rr(temperature)
deaths = rng.poisson(np.exp(log_mu))

# ── Fit DLNM ─────────────────────────────────────────────────────────────────

cp = CrossPred(
    var_basis="ns", var_df=5,
    lag_basis="ns", lag_df=4,
    max_lag=14,
    na_action="fill_zero",
    cen="median",
)
W = cp.fit(temperature)

# Seasonal confounders: sine/cosine of day-of-year
doy = day % 365
season = np.column_stack([
    np.sin(2 * np.pi * doy / 365),
    np.cos(2 * np.pi * doy / 365),
])

X = sm.add_constant(np.column_stack([W, season]))
model = sm.GLM(deaths, X, family=sm.families.Poisson()).fit()

# ── Predict ───────────────────────────────────────────────────────────────────

at_temps = np.linspace(-5, 38, 100)
cen_val = float(np.median(temperature))

# Slice out only the cross-basis coefficients (model also has intercept + 2 season terms)
n_cb = W.shape[1]
g_hat  = np.asarray(model.params)[1:n_cb + 1]
V_g_hat = np.asarray(model.cov_params())[1:n_cb + 1, 1:n_cb + 1]

result = cp.predict(coef=g_hat, vcov=V_g_hat, at=at_temps, cen=cen_val)

# ── Plot ──────────────────────────────────────────────────────────────────────

fig, axes = plt.subplots(1, 2, figsize=(13, 5))
fig.suptitle(
    "DLNM: temperature–mortality risk (synthetic city, 3 years daily data)",
    fontsize=12,
)

# Left: overall cumulative RR
ax = axes[0]
ax.fill_between(at_temps, result.allRR_low, result.allRR_high,
                alpha=0.25, color="steelblue", label="95% CI")
ax.plot(at_temps, result.allRR, color="steelblue", lw=2, label="Cumulative RR")
ax.axhline(1.0, color="black", lw=0.8, ls="--")
ax.axvline(cen_val, color="grey", lw=0.8, ls=":",
           label=f"Reference ({cen_val:.1f} °C)")
ax.set_xlabel("Temperature (°C)")
ax.set_ylabel("Relative risk (RR)")
ax.set_title("Overall cumulative effect (lags 0–14 days)")
ax.set_ylim(bottom=0)
ax.legend(fontsize=9)
ax.grid(True, alpha=0.3)

# Right: lag-response at a hot day vs a cold day
ax2 = axes[1]
lags = np.arange(result.matfit.shape[1])

for target_temp, color, label in [(-2.0, "#2166ac", "Cold (−2 °C)"),
                                   (34.0, "#d6604d", "Hot (34 °C)")]:
    idx = int(np.argmin(np.abs(at_temps - target_temp)))
    rr      = result.RR[idx, :]
    rr_low  = result.RR_low[idx, :]
    rr_high = result.RR_high[idx, :]
    ax2.fill_between(lags, rr_low, rr_high, alpha=0.2, color=color)
    ax2.plot(lags, rr, color=color, lw=2, label=label)

ax2.axhline(1.0, color="black", lw=0.8, ls="--")
ax2.set_xlabel("Lag (days)")
ax2.set_ylabel("Relative risk (RR)")
ax2.set_title("Lag-specific effect at extreme temperatures")
ax2.set_ylim(bottom=0)
ax2.legend(fontsize=9)
ax2.grid(True, alpha=0.3)

plt.tight_layout()
plt.savefig("example_risk_curve.png", dpi=150, bbox_inches="tight")
print("Saved example_risk_curve.png")
plt.show()
