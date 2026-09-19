"""Exercises for 01_numpy/03_stats_linalg_random.ipynb"""
import numpy as np
import pandas as pd

COMMON_SETUP = """
import numpy as np
import pandas as pd
"""

QUESTIONS = [
    dict(
        prompt="Compute the **sample** standard deviation of `x` (divide by n − 1, i.e. the same number pandas gives).",
        setup="x = np.array([2.0, 4.0, 6.0, 8.0])",
        solution="answer = np.std(x, ddof=1)",
        hint="np.std defaults to ddof=0 (divide by n). Pass ddof=1.",
    ),
    dict(
        prompt="`x` contains a NaN. Compute the mean of the non-missing values, and the count of NaN values.",
        setup="x = np.array([1.0, np.nan, 3.0, 5.0])",
        solution="mean_ok = np.nanmean(x)\nn_nan = int(np.isnan(x).sum())",
        hint="np.nanmean ignores NaN; np.isnan(x).sum() counts them (never use x == np.nan).",
        answer_var=["mean_ok", "n_nan"],
    ),
    dict(
        prompt="Compute the 10th and 90th percentiles of `x` as a 2-element array.",
        setup="x = np.array([1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0, 10.0])",
        solution="answer = np.percentile(x, [10, 90])",
        hint="np.percentile(x, [10, 90]) accepts a list of percentiles.",
    ),
    dict(
        prompt="Compute the Pearson correlation between `temp` and `load` as a single number.",
        setup="temp = np.array([0.0, 5.0, 10.0, 15.0, 20.0])\nload = np.array([40.0, 36.0, 31.0, 25.0, 22.0])",
        solution="answer = np.corrcoef(temp, load)[0, 1]",
        hint="np.corrcoef returns a 2 × 2 matrix; take the off-diagonal element.",
    ),
    dict(
        prompt="Fit a straight line `load ≈ slope * temp + intercept` with `np.polyfit` and assign the slope and intercept.",
        setup="temp = np.array([0.0, 5.0, 10.0, 15.0, 20.0])\nload = np.array([40.0, 36.0, 31.0, 25.0, 22.0])",
        solution="slope, intercept = np.polyfit(temp, load, 1)",
        hint="np.polyfit(x, y, 1) returns [slope, intercept] for degree 1.",
        answer_var=["slope", "intercept"],
    ),
    dict(
        prompt="For the 2 × 2 matrices `A` and `B`, compute the element-wise product and the matrix product.",
        setup="A = np.array([[1, 2],\n              [3, 4]])\nB = np.array([[1, 0],\n              [0, 2]])",
        solution="elementwise = A * B\nmatrix = A @ B",
        hint="`*` is element-wise; `@` (or np.dot) is the matrix product.",
        answer_var=["elementwise", "matrix"],
    ),
    dict(
        prompt="Solve the linear system `A @ x = b` for `x` using `np.linalg.solve`.",
        setup="A = np.array([[2.0, 1.0],\n              [1.0, 3.0]])\nb = np.array([3.0, 5.0])",
        solution="answer = np.linalg.solve(A, b)",
        hint="np.linalg.solve(A, b); avoid inv(A) @ b.",
    ),
    dict(
        prompt="Estimate the OLS coefficients of `y ≈ b0 + b1 * x` by hand: build the design matrix `X` with a column of ones and a column `x`, then use `np.linalg.lstsq`. Assign the 2-element coefficient array (b0, b1).",
        setup="x = np.array([0.0, 1.0, 2.0, 3.0])\ny = np.array([1.0, 3.0, 5.0, 7.0])",
        solution="X = np.column_stack([np.ones(len(x)), x])\nanswer = np.linalg.lstsq(X, y, rcond=None)[0]",
        hint="np.column_stack([np.ones(n), x]) for X; lstsq returns a tuple whose first element is the coefficient vector. Expected: intercept 1, slope 2.",
    ),
    dict(
        prompt="Compute the same OLS coefficients with the normal equations `(XᵀX)⁻¹ Xᵀ y`, using `np.linalg.solve` rather than `inv`.",
        setup="x = np.array([0.0, 1.0, 2.0, 3.0])\ny = np.array([1.0, 3.0, 5.0, 7.0])\nX = np.column_stack([np.ones(len(x)), x])",
        solution="answer = np.linalg.solve(X.T @ X, X.T @ y)",
        hint="solve(X.T @ X, X.T @ y).",
    ),
    dict(
        prompt="Draw 5 standard-normal numbers with a generator seeded with 42 (`np.random.default_rng(42)`), so the result is reproducible.",
        setup="",
        solution="rng = np.random.default_rng(42)\nanswer = rng.normal(size=5)",
        hint="rng = np.random.default_rng(42); rng.normal(size=5).",
    ),
    dict(
        prompt="Simulate an AR(1) series of length 5: `x[0] = 0` and `x[t] = 0.8 * x[t-1] + eps[t]` for t = 1..4, using the given `eps`.",
        setup="eps = np.array([0.0, 1.0, 0.5, -1.0, 0.2])",
        solution="x = np.zeros(5)\nfor t in range(1, 5):\n    x[t] = 0.8 * x[t - 1] + eps[t]\nanswer = x",
        hint="A short loop: x[t] = 0.8 * x[t-1] + eps[t]. Expected [0, 1, 1.3, 0.04, 0.232].",
    ),
    dict(
        prompt="`s1` and `s2` are pandas Series with the **same labels in different order**. Assign to `aligned` the sum `s1 + s2` (pandas aligns on labels) and to `positional` the sum of their `.values` (NumPy adds by position).",
        setup='s1 = pd.Series([1, 2, 3], index=["a", "b", "c"])\ns2 = pd.Series([30, 10, 20], index=["c", "a", "b"])',
        solution="aligned = s1 + s2\npositional = s1.values + s2.values",
        hint="Aligned gives a=11, b=22, c=33; positional gives [31, 12, 23].",
        answer_var=["aligned", "positional"],
    ),
    dict(
        prompt="On the real data: compute the correlation between temperature and consumption over all hours (a single number).",
        setup='df = pd.read_csv("../data/hourly_power_clean.csv")\ntemp = df["temp_c"].to_numpy()\ncons = df["consumption_mwh"].to_numpy()',
        solution="answer = np.corrcoef(temp, cons)[0, 1]",
        hint="np.corrcoef(temp, cons)[0, 1].",
    ),
    dict(
        prompt="On the real data: fit consumption on temperature by OLS with an intercept (`cons ≈ b0 + b1 * temp`) using `lstsq`, and assign the slope `b1` (MWh per °C).",
        setup='df = pd.read_csv("../data/hourly_power_clean.csv")\ntemp = df["temp_c"].to_numpy()\ncons = df["consumption_mwh"].to_numpy()',
        solution="X = np.column_stack([np.ones(len(temp)), temp])\ncoef = np.linalg.lstsq(X, cons, rcond=None)[0]\nanswer = coef[1]",
        hint="Same recipe as task 8; the slope is the second coefficient (it should be negative: warmer → less heating).",
    ),
]
