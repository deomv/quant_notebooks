"""Exercises for 04_sklearn/03_cross_validation_timeseries.ipynb"""
import numpy as np
import pandas as pd

COMMON_SETUP = """
import numpy as np
import pandas as pd
from sklearn.model_selection import KFold, TimeSeriesSplit, cross_val_score, GridSearchCV
from sklearn.linear_model import Ridge, LinearRegression
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import mean_squared_error
"""

QUESTIONS = [
    dict(
        prompt="`X` has 8 rows. Use `TimeSeriesSplit(n_splits=3)` and assign a list with the **test** indices of each fold (each as a Python list) to `answer`.",
        setup="X = np.arange(8).reshape(-1, 1)",
        solution="tscv = TimeSeriesSplit(n_splits=3)\nanswer = [list(test_idx) for _, test_idx in tscv.split(X)]",
        hint="for train_idx, test_idx in TimeSeriesSplit(n_splits=3).split(X): collect list(test_idx).",
        check_fn=lambda got, exp: ([list(map(int, g)) for g in got] == [list(map(int, e)) for e in exp], "" if [list(map(int, g)) for g in got] == [list(map(int, e)) for e in exp] else f"expected {exp}"),
    ),
    dict(
        prompt="Same 8 rows, `TimeSeriesSplit(n_splits=3, gap=1)`. Assign the **train** indices of the **last** fold as a list to `answer`.",
        setup="X = np.arange(8).reshape(-1, 1)",
        solution="tscv = TimeSeriesSplit(n_splits=3, gap=1)\nanswer = [list(tr) for tr, _ in tscv.split(X)][-1]",
        hint="With gap=1 the train set stops one row before the test set starts.",
        check_fn=lambda got, exp: (list(map(int, got)) == list(map(int, exp)), "" if list(map(int, got)) == list(map(int, exp)) else f"expected {list(exp)}"),
    ),
    dict(
        prompt="Use `KFold(n_splits=4, shuffle=True, random_state=0)` on the 8 rows and assign the test indices of the **first** fold as a sorted list to `answer`. (Notice they are not contiguous.)",
        setup="X = np.arange(8).reshape(-1, 1)",
        solution="kf = KFold(n_splits=4, shuffle=True, random_state=0)\nanswer = sorted(int(i) for i in next(kf.split(X))[1])",
        hint="next(kf.split(X)) gives (train_idx, test_idx) of the first fold.",
        check_fn=lambda got, exp: (sorted(map(int, got)) == sorted(map(int, exp)), "" if sorted(map(int, got)) == sorted(map(int, exp)) else f"expected {list(exp)}"),
    ),
    dict(
        prompt="Walk-forward by hand on `y` (9 values, feature = previous value). For folds t = 3, 6: train on rows [0, t), predict rows [t, t+3) with `LinearRegression` on `lag1`. Concatenate the out-of-sample predictions into one array `answer` (5 values: the last fold only has rows 6 and 7).",
        setup="y = pd.Series([1.0, 2.0, 3.0, 5.0, 8.0, 13.0, 21.0, 34.0, 55.0])\ndf = pd.DataFrame({\"y\": y, \"lag1\": y.shift(1)}).dropna().reset_index(drop=True)   # 8 rows",
        solution="preds = []\nfor t in [3, 6]:\n    train = df.iloc[:t]\n    test = df.iloc[t:t + 3]\n    m = LinearRegression().fit(train[[\"lag1\"]], train[\"y\"])\n    preds.append(m.predict(test[[\"lag1\"]]))\nanswer = np.concatenate(preds)",
        hint="Loop over t in [3, 6]; df.iloc[:t] is train, df.iloc[t:t+3] is test; fit, predict, collect; np.concatenate.",
        tol=1e-6,
        note="df has 8 rows after dropping the NaN lag, so the second fold predicts rows 6 and 7 only (5 predictions in total is correct).",
    ),
    dict(
        prompt="`cross_val_score` with `scoring='neg_root_mean_squared_error'` returns negative numbers. Run it for `Ridge(alpha=1.0)` on `X`, `y` with `TimeSeriesSplit(n_splits=3)` and assign the **mean RMSE as a positive float** to `answer`.",
        setup="rng = np.random.default_rng(0)\nX = np.arange(20).reshape(-1, 1).astype(float)\ny = 2 * X[:, 0] + rng.normal(0, 1, 20)",
        solution="scores = cross_val_score(Ridge(alpha=1.0), X, y, cv=TimeSeriesSplit(n_splits=3), scoring=\"neg_root_mean_squared_error\")\nanswer = float(-scores.mean())",
        hint="Negate the mean of the scores.",
        tol=1e-6,
    ),
    dict(
        prompt="Run `GridSearchCV` over `alpha` in [0.1, 1, 10, 100] for a `Pipeline` of StandardScaler + Ridge with `TimeSeriesSplit(n_splits=3)` and `scoring='neg_root_mean_squared_error'`. Assign the best alpha (float) to `answer`. The parameter name must reach the ridge step.",
        setup="rng = np.random.default_rng(1)\nX = rng.normal(size=(30, 3))\ny = X @ np.array([1.0, 0.5, -0.5]) + rng.normal(0, 0.1, 30)",
        solution="pipe = Pipeline([(\"scale\", StandardScaler()), (\"ridge\", Ridge())])\ngs = GridSearchCV(pipe, {\"ridge__alpha\": [0.1, 1, 10, 100]}, cv=TimeSeriesSplit(n_splits=3), scoring=\"neg_root_mean_squared_error\").fit(X, y)\nanswer = float(gs.best_params_[\"ridge__alpha\"])",
        hint="Parameter grid key is 'ridge__alpha' (step name, double underscore, parameter).",
        tol=1e-9,
    ),
    dict(
        prompt="From the fitted grid search of the previous task, build a DataFrame with columns `alpha` (the parameter values) and `rmse` (positive mean test RMSE), one row per alpha, in grid order. Assign it to `answer`.",
        setup="rng = np.random.default_rng(1)\nX = rng.normal(size=(30, 3))\ny = X @ np.array([1.0, 0.5, -0.5]) + rng.normal(0, 0.1, 30)\npipe = Pipeline([(\"scale\", StandardScaler()), (\"ridge\", Ridge())])\ngs = GridSearchCV(pipe, {\"ridge__alpha\": [0.1, 1, 10, 100]}, cv=TimeSeriesSplit(n_splits=3), scoring=\"neg_root_mean_squared_error\").fit(X, y)",
        solution="res = pd.DataFrame(gs.cv_results_)\nanswer = pd.DataFrame({\"alpha\": res[\"param_ridge__alpha\"].astype(float), \"rmse\": -res[\"mean_test_score\"]})",
        hint="pd.DataFrame(gs.cv_results_) has 'param_ridge__alpha' and 'mean_test_score'; negate the score.",
        tol=1e-6,
    ),
    dict(
        prompt="Compare an honest and a leaky evaluation on autocorrelated `y`. Assign to `rmse_shuffled` the mean RMSE from `KFold(5, shuffle=True, random_state=0)` and to `rmse_ts` the mean RMSE from `TimeSeriesSplit(5)`, both for `LinearRegression` on `X` (lag features). Positive floats.",
        setup="rng = np.random.default_rng(2)\nn = 200\ne = np.zeros(n)\nfor t in range(1, n):\n    e[t] = 0.95 * e[t - 1] + rng.normal()\ny = pd.Series(e)\nX = pd.DataFrame({\"lag1\": y.shift(1), \"lag2\": y.shift(2)}).dropna()\ny = y.loc[X.index]",
        solution="s1 = cross_val_score(LinearRegression(), X, y, cv=KFold(5, shuffle=True, random_state=0), scoring=\"neg_root_mean_squared_error\")\ns2 = cross_val_score(LinearRegression(), X, y, cv=TimeSeriesSplit(5), scoring=\"neg_root_mean_squared_error\")\nrmse_shuffled = float(-s1.mean())\nrmse_ts = float(-s2.mean())",
        hint="Two cross_val_score calls with different cv objects; negate the means.",
        answer_var=["rmse_shuffled", "rmse_ts"],
        tol=1e-6,
    ),
    dict(
        prompt="Given a target that is `h = 2` steps ahead, what `gap` should `TimeSeriesSplit` use so that no training row's target overlaps the test window? Assign the integer to `answer`.",
        setup="h = 2",
        solution="answer = h",
        hint="The last h training rows contain targets from the first h test rows.",
        check_fn=lambda got, exp: (int(got) == int(exp), "" if int(got) == int(exp) else "think about how many training rows have a target inside the test window"),
    ),
    dict(
        prompt="Real data. With the given lag features and a chronological 80/20 split, fit `Ridge(alpha=1.0)` on train and assign the test RMSE (float, rounded to 1 decimal) to `answer`.",
        setup="df = pd.read_csv(\"../data/hourly_power_clean.csv\", parse_dates=[\"time\"]).sort_values(\"time\").reset_index(drop=True)\ny = df[\"consumption_mwh\"]\nX = pd.DataFrame({\"lag24\": y.shift(24), \"lag168\": y.shift(168), \"temp\": df[\"temp_c\"]}).dropna()\ny = y.loc[X.index]\nsplit = int(len(X) * 0.8)",
        solution="m = Ridge(alpha=1.0).fit(X.iloc[:split], y.iloc[:split])\npred = m.predict(X.iloc[split:])\nanswer = round(float(np.sqrt(mean_squared_error(y.iloc[split:], pred))), 1)",
        hint="X.iloc[:split] / X.iloc[split:] and the same for y; fit; predict; RMSE.",
        tol=0.2,
    ),
]
