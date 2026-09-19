"""Exercises for 04_sklearn/04_model_diagnostics_and_interpretation.ipynb"""
import numpy as np
import pandas as pd

COMMON_SETUP = """
import numpy as np
import pandas as pd
from sklearn.linear_model import LinearRegression, Ridge
from sklearn.metrics import mean_squared_error, r2_score
from sklearn.inspection import permutation_importance
"""

QUESTIONS = [
    dict(
        prompt="Build an error table from `y` and `pred`: a DataFrame with columns `y`, `pred`, `error` (= y − pred). Then assign the **mean error** (bias, float) to `bias` and the table to `answer`.",
        setup="y = pd.Series([10.0, 12.0, 11.0, 15.0, 14.0, 16.0])\npred = pd.Series([11.0, 11.0, 12.0, 14.0, 15.0, 15.0])",
        solution="answer = pd.DataFrame({\"y\": y, \"pred\": pred})\nanswer[\"error\"] = answer[\"y\"] - answer[\"pred\"]\nbias = float(answer[\"error\"].mean())",
        hint="error = y - pred; bias is its mean (zero on average can still hide large signed errors by group).",
        answer_var=["answer", "bias"],
        tol=1e-9,
    ),
    dict(
        prompt="From `err` (errors over time), compute the **rolling RMSE with a window of 3**: square, rolling mean, square root. Assign the Series to `answer`.",
        setup="err = pd.Series([1.0, -1.0, 2.0, -2.0, 3.0, -3.0])",
        solution="answer = np.sqrt((err ** 2).rolling(3).mean())",
        hint="(err ** 2).rolling(3).mean() then np.sqrt. The first two rows are NaN.",
        tol=1e-9,
    ),
    dict(
        prompt="Fit `LinearRegression` on the first 6 rows of `X`, `y` and compute RMSE on train (rows 0–5) and on test (rows 6–9). Assign both floats to `rmse_train` and `rmse_test`.",
        setup="rng = np.random.default_rng(0)\nX = pd.DataFrame({\"x\": np.arange(10.0)})\ny = 2 * X[\"x\"] + rng.normal(0, 1, 10)",
        solution="m = LinearRegression().fit(X.iloc[:6], y.iloc[:6])\nrmse_train = float(np.sqrt(mean_squared_error(y.iloc[:6], m.predict(X.iloc[:6]))))\nrmse_test = float(np.sqrt(mean_squared_error(y.iloc[6:], m.predict(X.iloc[6:]))))",
        hint="Fit on iloc[:6]; predict both slices; RMSE each.",
        answer_var=["rmse_train", "rmse_test"],
        tol=1e-6,
    ),
    dict(
        prompt="Coefficient stability: fit `LinearRegression` on `X`, `y` separately for `period == 'A'` and `period == 'B'`. Assign a DataFrame with index `['x1', 'x2']` and columns `A`, `B` holding the coefficients to `answer`.",
        setup="X = pd.DataFrame({\"x1\": [1, 2, 3, 4, 1, 2, 3, 4], \"x2\": [1, 1, 2, 2, 2, 2, 1, 1]})\ny = pd.Series([3, 5, 8, 10, 4, 6, 7, 9])\nperiod = pd.Series([\"A\"] * 4 + [\"B\"] * 4)",
        solution="cols = {}\nfor p in [\"A\", \"B\"]:\n    m = LinearRegression().fit(X[period == p], y[period == p])\n    cols[p] = m.coef_\nanswer = pd.DataFrame(cols, index=[\"x1\", \"x2\"])",
        hint="Mask rows with period == 'A', fit, take coef_; same for 'B'; pd.DataFrame({'A': ..., 'B': ...}, index=['x1', 'x2']).",
        tol=1e-6,
    ),
    dict(
        prompt="Compute the VIF of column `x2` in `X` by hand: regress `x2` on the other columns, get R², VIF = 1 / (1 − R²). Assign the float to `answer`.",
        setup="rng = np.random.default_rng(0)\nx1 = rng.normal(size=50)\nX = pd.DataFrame({\"x1\": x1, \"x2\": x1 + rng.normal(0, 0.1, 50), \"x3\": rng.normal(size=50)})",
        solution="others = X.drop(columns=\"x2\")\nr2 = LinearRegression().fit(others, X[\"x2\"]).score(others, X[\"x2\"])\nanswer = float(1 / (1 - r2))",
        hint="LinearRegression().fit(others, X['x2']).score(others, X['x2']) is R²; VIF = 1/(1-R²).",
        tol=1e-4,
    ),
    dict(
        prompt="Fit OLS and `Ridge(alpha=1.0)` on the collinear `X` (x2 ≈ x1). Assign the OLS coefficient array to `ols` and the Ridge one to `ridge`. Ridge should spread the weight across x1 and x2.",
        setup="rng = np.random.default_rng(0)\nx1 = rng.normal(size=50)\nX = pd.DataFrame({\"x1\": x1, \"x2\": x1 + rng.normal(0, 0.1, 50)})\ny = 2 * x1 + rng.normal(0, 0.5, 50)",
        solution="ols = LinearRegression().fit(X, y).coef_\nridge = Ridge(alpha=1.0).fit(X, y).coef_",
        hint="Two fits on the same X, y; take .coef_ from each.",
        answer_var=["ols", "ridge"],
        tol=1e-6,
    ),
    dict(
        prompt="Permutation importance by hand for `x1`: fit `LinearRegression` on `X`, `y`, record R² on `X`; then shuffle column `x1` with `rng.permutation` (rng seeded 0, as given) and recompute R². Assign the **drop** in R² (baseline − shuffled, float) to `answer`.",
        setup="rng_data = np.random.default_rng(1)\nX = pd.DataFrame({\"x1\": rng_data.normal(size=40), \"x2\": rng_data.normal(size=40)})\ny = 3 * X[\"x1\"] + 0.1 * X[\"x2\"]\nrng = np.random.default_rng(0)",
        solution="m = LinearRegression().fit(X, y)\nbase = m.score(X, y)\nXs = X.copy()\nXs[\"x1\"] = rng.permutation(Xs[\"x1\"].values)\nanswer = float(base - m.score(Xs, y))",
        hint="Copy X, replace x1 with rng.permutation(X['x1'].values), call m.score again on the shuffled copy.",
        tol=1e-6,
    ),
    dict(
        prompt="Use sklearn's `permutation_importance` on the fitted model with `n_repeats=5, random_state=0` and assign the `importances_mean` array to `answer`.",
        setup="rng_data = np.random.default_rng(1)\nX = pd.DataFrame({\"x1\": rng_data.normal(size=40), \"x2\": rng_data.normal(size=40)})\ny = 3 * X[\"x1\"] + 0.1 * X[\"x2\"]\nm = LinearRegression().fit(X, y)",
        solution="answer = permutation_importance(m, X, y, n_repeats=5, random_state=0).importances_mean",
        hint="permutation_importance(model, X, y, n_repeats=5, random_state=0).importances_mean",
        tol=1e-6,
    ),
    dict(
        prompt="Partial dependence by hand: for `x1` in [−1, 0, 1], set every row's `x1` to that value, keep `x2` at its **median**, and predict with `m`. Assign the array of three mean predictions to `answer`.",
        setup="rng_data = np.random.default_rng(1)\nX = pd.DataFrame({\"x1\": rng_data.normal(size=40), \"x2\": rng_data.normal(size=40)})\ny = 3 * X[\"x1\"] + 0.1 * X[\"x2\"] + 1\nm = LinearRegression().fit(X, y)",
        solution="out = []\nfor v in [-1, 0, 1]:\n    Xg = X.copy()\n    Xg[\"x1\"] = v\n    Xg[\"x2\"] = X[\"x2\"].median()\n    out.append(m.predict(Xg).mean())\nanswer = np.array(out)",
        hint="Loop over the grid; copy X; overwrite x1 with the grid value and x2 with its median; predict; mean.",
        tol=1e-6,
    ),
    dict(
        prompt="Residual autocorrelation: assign the lag-1 autocorrelation of `resid` (float) to `answer`.",
        setup="resid = pd.Series([1.0, 0.8, 0.5, -0.2, -0.6, -0.4, 0.1, 0.5])",
        solution="answer = float(resid.autocorr(lag=1))",
        hint="Series.autocorr(lag=1), or corr of resid with resid.shift(1).",
        tol=1e-6,
    ),
    dict(
        prompt="The leaked-feature detector. `X` contains a column `roll2` computed as `y.rolling(2).mean()` **without** shifting (it contains y itself) and `lag1`. Fit `LinearRegression` on the rows without NaN and assign the coefficient on `roll2` (float) and the in-sample R² (float) to `coef_roll2` and `r2`.",
        setup="y = pd.Series([10.0, 12.0, 11.0, 15.0, 14.0, 16.0, 18.0, 17.0])\nX = pd.DataFrame({\"roll2\": y.rolling(2).mean(), \"lag1\": y.shift(1)})\nmask = X.notna().all(axis=1)",
        solution="m = LinearRegression().fit(X[mask], y[mask])\ncoef_roll2 = float(m.coef_[0])\nr2 = float(m.score(X[mask], y[mask]))",
        hint="Fit on X[mask], y[mask]; coef_[0] is roll2. Expect coef 2 on roll2, −1 on lag1 and R² = 1: the model reconstructs y exactly.",
        answer_var=["coef_roll2", "r2"],
        tol=1e-6,
    ),
    dict(
        prompt="Real data. Fit `Ridge(alpha=1.0)` on the given lag model with a chronological split, then compute the **mean error by hour of day** on the test set. Assign the hour (int) with the largest **absolute** mean error to `answer`.",
        setup="df = pd.read_csv(\"../data/hourly_power_clean.csv\", parse_dates=[\"time\"]).sort_values(\"time\").reset_index(drop=True)\ny = df[\"consumption_mwh\"]\nX = pd.DataFrame({\"lag1\": y.shift(1), \"lag24\": y.shift(24), \"temp\": df[\"temp_c\"]}).dropna()\ny = y.loc[X.index]\nhour = df.loc[X.index, \"time\"].dt.hour\nsplit = int(len(X) * 0.8)",
        solution="m = Ridge(alpha=1.0).fit(X.iloc[:split], y.iloc[:split])\nerr = y.iloc[split:] - m.predict(X.iloc[split:])\nby_hour = err.groupby(hour.iloc[split:]).mean()\nanswer = int(by_hour.abs().idxmax())",
        hint="err = y_test - pred; err.groupby(hour_test).mean(); .abs().idxmax().",
        check_fn=lambda got, exp: (int(got) == int(exp), "" if int(got) == int(exp) else f"expected hour {exp}"),
    ),
]
