"""Exercises for 05_time_series_research/04_classical_time_series_stats.ipynb"""
import numpy as np
import pandas as pd

COMMON_SETUP = """
import numpy as np
import pandas as pd
"""

QUESTIONS = [
    dict(
        prompt="Compute the lag-1 autocorrelation of `x` with the **standard estimator**: centre x with its overall mean, numerator = Σ xc[t]·xc[t−1] over t=1..n−1, denominator = Σ xc[t]² over all t.",
        setup="x = pd.Series([1.0, 3.0, 5.0, 7.0, 5.0, 3.0])",
        solution="xc = x - x.mean()\nnum = (xc.values[1:] * xc.values[:-1]).sum()\nden = (xc.values ** 2).sum()\nanswer = float(num / den)",
        hint="Centre once with the overall mean; multiply neighbours; divide by the total sum of squares (not by corrcoef).",
    ),
    dict(
        prompt="Estimate the partial autocorrelation at lag 2 of `x`: regress x[t] on an intercept, x[t−1] and x[t−2] with `np.linalg.lstsq` and assign the coefficient on x[t−2].",
        setup="x = pd.Series([1.0, 3.0, 5.0, 7.0, 5.0, 3.0, 1.0, 3.0, 5.0, 7.0])",
        solution="y = x.values[2:]\nX = np.column_stack([np.ones(len(y)), x.values[1:-1], x.values[:-2]])\nbeta = np.linalg.lstsq(X, y, rcond=None)[0]\nanswer = float(beta[2])",
        hint="Rows t=2..n−1; columns [1, x[t-1], x[t-2]]; lstsq returns (beta, ...). The last coefficient is the PACF(2).",
        tol=1e-6,
    ),
    dict(
        prompt="Assign the first difference of `x` (x[t] − x[t−1]) as a Series with the same index; the first value is NaN.",
        setup="x = pd.Series([1.0, 3.0, 5.0, 7.0, 5.0, 3.0])",
        solution="answer = x.diff()",
        hint="x.diff() or x - x.shift(1).",
    ),
    dict(
        prompt="Assign the seasonal difference of `x` with period 4 (x[t] − x[t−4]); the first four values are NaN.",
        setup="x = pd.Series([1.0, 3.0, 5.0, 7.0, 2.0, 4.0, 6.0, 8.0])",
        solution="answer = x.diff(4)",
        hint="diff takes a period argument: x.diff(4).",
    ),
    dict(
        prompt="Assign the centred 3-point moving average of `x` (each value averaged with its neighbours on both sides; NaN at the ends).",
        setup="x = pd.Series([1.0, 3.0, 5.0, 7.0, 2.0, 4.0, 6.0, 8.0, 3.0, 5.0, 7.0, 9.0])",
        solution="answer = x.rolling(3, center=True).mean()",
        hint="rolling(3, center=True).mean(). Without center=True the label sits at the end of the window.",
    ),
    dict(
        prompt="`x` has period 4. Assign the seasonal means by position: a Series indexed 0,1,2,3 with the mean of the values at positions t where t mod 4 equals that index.",
        setup="x = pd.Series([1.0, 3.0, 5.0, 7.0, 2.0, 4.0, 6.0, 8.0, 3.0, 5.0, 7.0, 9.0])",
        solution="answer = x.groupby(np.arange(len(x)) % 4).mean()",
        hint="groupby with the key np.arange(len(x)) % 4.",
    ),
    dict(
        prompt="Fit an AR(1) by OLS: regress x[t] on an intercept and x[t−1] with `np.linalg.lstsq`. Assign the slope coefficient φ.",
        setup="x = pd.Series([8.0, 4.0, 2.0, 1.0, 0.5, 0.25, 0.125, 0.0625])",
        solution="y = x.values[1:]\nX = np.column_stack([np.ones(len(y)), x.values[:-1]])\nbeta = np.linalg.lstsq(X, y, rcond=None)[0]\nanswer = float(beta[1])",
        hint="Each value is half the previous one, so φ should come out as 0.5 exactly.",
        tol=1e-6,
    ),
    dict(
        prompt="Simple exponential smoothing with α = 0.5: start with level = x[0], then for each later value update level = α·x[t] + (1−α)·level. Assign the level after processing **all** values of `x`.",
        setup="x = pd.Series([10.0, 20.0, 10.0, 20.0, 10.0, 20.0])\nalpha = 0.5",
        solution="level = x.iloc[0]\nfor v in x.iloc[1:]:\n    level = alpha * v + (1 - alpha) * level\nanswer = float(level)",
        hint="Five updates: 15, 12.5, 16.25, 13.125, 16.5625.",
        tol=1e-6,
    ),
    dict(
        prompt="`y` is `x` moved one step later (y[t+1] = x[t]). Assign `ccf_plus1` = corr(x[t], y[t+1]) and `ccf_minus1` = corr(x[t+1], y[t]) using `np.corrcoef` on the aligned slices.",
        setup="x = pd.Series([1.0, 5.0, 2.0, 8.0, 3.0, 6.0, 4.0])\ny = x.shift(1).fillna(0.0)",
        solution="ccf_plus1 = float(np.corrcoef(x.values[:-1], y.values[1:])[0, 1])\nccf_minus1 = float(np.corrcoef(x.values[1:], y.values[:-1])[0, 1])",
        hint="corr(x[t], y[t+1]) pairs x[:-1] with y[1:]; it should be exactly 1 because y is x delayed by one. The other direction is much lower.",
        answer_var=["ccf_plus1", "ccf_minus1"],
        tol=1e-6,
    ),
    dict(
        prompt="Variance ratio at k = 2: var(x[t] − x[t−2]) / (2 · var(x[t] − x[t−1])), both variances with ddof=1 over the available differences.",
        setup="x = pd.Series([1.0, 2.0, 4.0, 7.0, 11.0, 16.0, 22.0, 29.0])",
        solution="d1 = x.diff(1).dropna()\nd2 = x.diff(2).dropna()\nanswer = float(d2.var(ddof=1) / (2 * d1.var(ddof=1)))",
        hint="x.diff(2) and x.diff(1), dropna, .var(ddof=1). Near 1 for a random walk, above 1 for a trending series.",
        tol=1e-6,
    ),
    dict(
        prompt="Real data: load `../data/hourly_power_clean.csv` and assign the lag-24 autocorrelation of `consumption_mwh` using pandas `Series.autocorr`.",
        setup='df = pd.read_csv("../data/hourly_power_clean.csv", parse_dates=["time"])',
        solution='answer = float(df["consumption_mwh"].autocorr(24))',
        hint="df['consumption_mwh'].autocorr(24).",
        tol=1e-4,
    ),
]
