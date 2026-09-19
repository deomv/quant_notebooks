"""Exercises for 04_sklearn/01_regression_workflow.ipynb"""
import numpy as np
import pandas as pd

COMMON_SETUP = """
import numpy as np
import pandas as pd
from sklearn.linear_model import LinearRegression, Ridge, Lasso
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
"""

QUESTIONS = [
    dict(
        prompt="Fit a `LinearRegression` on `X` and `y` (y is exactly 2·x1 + 3). Assign the fitted coefficient array to `coef` and the intercept (a float) to `intercept`.",
        setup="X = pd.DataFrame({\"x1\": [1, 2, 3, 4, 5, 6]})\ny = pd.Series([5, 7, 9, 11, 13, 15])",
        solution="model = LinearRegression().fit(X, y)\ncoef = model.coef_\nintercept = float(model.intercept_)",
        hint="LinearRegression().fit(X, y) then .coef_ (array) and .intercept_.",
        answer_var=["coef", "intercept"],
        tol=1e-6,
    ),
    dict(
        prompt="Using `y_true` and `pred`, compute the RMSE **by hand** (no sklearn): mean of squared errors, then square root. Assign the float to `answer`.",
        setup="y_true = np.array([10.0, 20.0, 30.0, 40.0])\npred = np.array([12.0, 18.0, 33.0, 37.0])",
        solution="answer = float(np.sqrt(np.mean((y_true - pred) ** 2)))",
        hint="errors = y_true - pred; square them; take the mean; np.sqrt.",
        tol=1e-6,
    ),
    dict(
        prompt="Compute R² of `pred` against `y_true` with sklearn. Be careful with the argument order (true values first). Assign the float to `answer`.",
        setup="y_true = np.array([10.0, 20.0, 30.0, 40.0])\npred = np.array([12.0, 18.0, 33.0, 37.0])",
        solution="answer = float(r2_score(y_true, pred))",
        hint="r2_score(y_true, y_pred). Swapping the arguments gives a different number because R² is not symmetric.",
        tol=1e-6,
    ),
    dict(
        prompt="Fit `LinearRegression` on the two-feature `X` and `y` (y = 3·x1 − 2·x2 + 1) and build a Series of coefficients indexed by feature name, sorted from smallest to largest.",
        setup="X = pd.DataFrame({\"x1\": [1, 2, 3, 4, 5, 6], \"x2\": [2, 1, 4, 3, 6, 5]})\ny = pd.Series(3 * X[\"x1\"] - 2 * X[\"x2\"] + 1)",
        solution="model = LinearRegression().fit(X, y)\nanswer = pd.Series(model.coef_, index=X.columns).sort_values()",
        hint="pd.Series(model.coef_, index=X.columns).sort_values()",
        tol=1e-6,
    ),
    dict(
        prompt="`df` has 8 hourly rows. Do a **chronological** split with the first 6 rows as train and the last 2 as test. Assign the list of test-set timestamps (as strings 'YYYY-MM-DD HH:MM') to `answer`.",
        setup="df = pd.DataFrame({\"time\": pd.date_range(\"2023-01-01\", periods=8, freq=\"h\"), \"y\": [1, 2, 3, 4, 5, 6, 7, 8]})",
        solution="train = df.iloc[:6]\ntest = df.iloc[6:]\nanswer = list(test[\"time\"].dt.strftime(\"%Y-%m-%d %H:%M\"))",
        hint="iloc[:6] and iloc[6:] on a time-sorted frame; then .dt.strftime on the time column.",
    ),
    dict(
        prompt="Fit OLS and `Ridge(alpha=10)` on the same `X`, `y`. Assign the OLS coefficient array to `ols_coef` and the Ridge coefficient array to `ridge_coef`. (Ridge should be visibly smaller in absolute value.)",
        setup="X = pd.DataFrame({\"x1\": [1, 2, 3, 4, 5, 6], \"x2\": [2, 1, 4, 3, 6, 5]})\ny = pd.Series(3 * X[\"x1\"] - 2 * X[\"x2\"] + 1)",
        solution="ols_coef = LinearRegression().fit(X, y).coef_\nridge_coef = Ridge(alpha=10).fit(X, y).coef_",
        hint="Ridge(alpha=10).fit(X, y).coef_ — same API as LinearRegression.",
        answer_var=["ols_coef", "ridge_coef"],
        tol=1e-6,
    ),
    dict(
        prompt="One-hot encode the `hour` column of `X` with `pd.get_dummies`, dropping the first category, and keep the numeric columns as integers 0/1. Assign the resulting DataFrame's column names as a list to `answer`.",
        setup="X = pd.DataFrame({\"temp\": [5, 6, 7, 8], \"hour\": [0, 1, 2, 3]})",
        solution="X_oh = pd.get_dummies(X, columns=[\"hour\"], drop_first=True, dtype=int)\nanswer = list(X_oh.columns)",
        hint="pd.get_dummies(X, columns=['hour'], drop_first=True, dtype=int). Column names look like 'hour_1'.",
    ),
    dict(
        prompt="Compute the naive 'same value one step earlier' baseline for the last 4 rows of `y` (the test part): prediction for row t is y at row t−1. Assign the RMSE of that baseline on the 4 test rows to `answer` (float).",
        setup="y = pd.Series([10.0, 12.0, 11.0, 15.0, 14.0, 16.0, 18.0, 17.0])",
        solution="naive = y.shift(1).iloc[4:]\nanswer = float(np.sqrt(mean_squared_error(y.iloc[4:], naive)))",
        hint="y.shift(1) is the previous value; slice both y and the shifted series with .iloc[4:]; then RMSE.",
        tol=1e-6,
    ),
    dict(
        prompt="`out` has true values, predictions and an `hour` column. Compute the **mean error** (true − pred) per hour as a Series indexed by hour.",
        setup="out = pd.DataFrame({\"hour\": [0, 1, 0, 1, 0, 1], \"y\": [10, 20, 12, 22, 11, 21], \"pred\": [11, 18, 11, 19, 12, 18]})",
        solution="out[\"error\"] = out[\"y\"] - out[\"pred\"]\nanswer = out.groupby(\"hour\")[\"error\"].mean()",
        hint="Add an error column, then groupby('hour')['error'].mean().",
        tol=1e-9,
    ),
    dict(
        prompt="Real data. `df`, `features`, `target` and the chronological `split` are given. Fit `LinearRegression` on the first `split` rows and assign the test-set RMSE (float, rounded to 1 decimal) to `answer`.",
        setup="df = pd.read_csv(\"../data/hourly_power_clean.csv\", parse_dates=[\"time\"]).sort_values(\"time\").reset_index(drop=True)\ndf[\"hour\"] = df[\"time\"].dt.hour\ndf[\"weekend\"] = (df[\"time\"].dt.dayofweek >= 5).astype(int)\ndf[\"hdd\"] = np.clip(15 - df[\"temp_c\"], 0, None)\nfeatures = [\"temp_c\", \"hdd\", \"wind_ms\", \"solar_wm2\", \"hour\", \"weekend\"]\ntarget = \"consumption_mwh\"\nsplit = int(len(df) * 0.8)",
        solution="train, test = df.iloc[:split], df.iloc[split:]\nmodel = LinearRegression().fit(train[features], train[target])\npred = model.predict(test[features])\nanswer = round(float(np.sqrt(mean_squared_error(test[target], pred))), 1)",
        hint="train = df.iloc[:split], test = df.iloc[split:]; fit on train[features]; predict test; RMSE = sqrt(mean_squared_error).",
        tol=0.2,
    ),
    dict(
        prompt="Real data, continued. Compute the RMSE of the naive **same hour yesterday** baseline (`shift(24)`) on the same test rows. Assign a boolean `answer`: does the model from the previous task beat this baseline (model RMSE < naive RMSE)?",
        setup="df = pd.read_csv(\"../data/hourly_power_clean.csv\", parse_dates=[\"time\"]).sort_values(\"time\").reset_index(drop=True)\ndf[\"hour\"] = df[\"time\"].dt.hour\ndf[\"weekend\"] = (df[\"time\"].dt.dayofweek >= 5).astype(int)\ndf[\"hdd\"] = np.clip(15 - df[\"temp_c\"], 0, None)\nfeatures = [\"temp_c\", \"hdd\", \"wind_ms\", \"solar_wm2\", \"hour\", \"weekend\"]\ntarget = \"consumption_mwh\"\nsplit = int(len(df) * 0.8)\ntrain, test = df.iloc[:split], df.iloc[split:]\nmodel_rmse = float(np.sqrt(mean_squared_error(test[target], LinearRegression().fit(train[features], train[target]).predict(test[features]))))",
        solution="naive24 = df[target].shift(24).iloc[split:]\nnaive_rmse = float(np.sqrt(mean_squared_error(test[target], naive24)))\nanswer = bool(model_rmse < naive_rmse)",
        hint="df[target].shift(24).iloc[split:] lines up with test[target]; compare the two RMSEs.",
        note="With hour as a plain number the model does NOT beat 'same hour yesterday'. The cheat sheet shows one-hot hours fix that.",
    ),
]
