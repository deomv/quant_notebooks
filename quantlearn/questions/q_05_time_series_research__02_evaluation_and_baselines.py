"""Exercises for 05_time_series_research/02_evaluation_and_baselines.ipynb"""
import numpy as np
import pandas as pd

COMMON_SETUP = """
import numpy as np
import pandas as pd
"""

QUESTIONS = [
    dict(
        prompt="Compute the RMSE of `pred` against `y` by hand (no sklearn): square root of the mean squared error.",
        setup="y = np.array([10.0, 20.0, 30.0, 40.0])\npred = np.array([12.0, 18.0, 33.0, 37.0])",
        solution="answer = float(np.sqrt(np.mean((y - pred) ** 2)))",
        hint="errors = y - pred; square, mean, sqrt.",
    ),
    dict(
        prompt="Compute the MAE (mean absolute error) of `pred` against `y`.",
        setup="y = np.array([10.0, 20.0, 30.0, 40.0])\npred = np.array([12.0, 18.0, 33.0, 37.0])",
        solution="answer = float(np.mean(np.abs(y - pred)))",
        hint="Absolute errors are 2, 2, 3, 3.",
    ),
    dict(
        prompt="Compute MAPE and sMAPE as **fractions** (not percent). MAPE = mean(|y−pred| / |y|); sMAPE = mean(2·|y−pred| / (|y|+|pred|)). Assign `mape` and `smape`.",
        setup="y = np.array([10.0, 20.0, 30.0, 40.0])\npred = np.array([12.0, 18.0, 33.0, 37.0])",
        solution="mape = float(np.mean(np.abs(y - pred) / np.abs(y)))\nsmape = float(np.mean(2 * np.abs(y - pred) / (np.abs(y) + np.abs(pred))))",
        hint="Element-wise division, then the mean. Both are numbers between 0 and 1 here.",
        answer_var=["mape", "smape"],
    ),
    dict(
        prompt="Compute MASE: the MAE of `pred` divided by the MAE of the in-sample naive one-step forecast, i.e. divided by the mean absolute difference between consecutive values of `y`.",
        setup="y = np.array([10.0, 20.0, 30.0, 40.0])\npred = np.array([12.0, 18.0, 33.0, 37.0])",
        solution="answer = float(np.mean(np.abs(y - pred)) / np.mean(np.abs(np.diff(y))))",
        hint="np.diff(y) gives the consecutive differences (all 10 here), so the denominator is 10.",
    ),
    dict(
        prompt="Compute R² by hand: 1 − SS_res / SS_tot, where SS_res = Σ(y−pred)² and SS_tot = Σ(y−mean(y))².",
        setup="y = np.array([10.0, 20.0, 30.0, 40.0])\npred = np.array([12.0, 18.0, 33.0, 37.0])",
        solution="ss_res = np.sum((y - pred) ** 2)\nss_tot = np.sum((y - y.mean()) ** 2)\nanswer = float(1 - ss_res / ss_tot)",
        hint="SS_tot uses the mean of y (25), not zero.",
    ),
    dict(
        prompt="`y` repeats with period 4. Compute the RMSE of the naive forecast 'same value 4 rows earlier' over the rows where that forecast exists.",
        setup="y = pd.Series([10, 20, 30, 40, 12, 18, 33, 37])",
        solution="naive = y.shift(4)\nerr = (y - naive).dropna()\nanswer = float(np.sqrt((err ** 2).mean()))",
        hint="y.shift(4) is the forecast; drop the 4 NaN rows before averaging the squared errors.",
    ),
    dict(
        prompt="Walk-forward split of `n` rows: the first train set has `initial` rows, each test block has `step` rows, and the train set grows to include everything before the block. Assign a list of `(train_end, test_end)` tuples (both end positions exclusive), one per test block.",
        setup="n = 12\ninitial = 6\nstep = 2",
        solution="answer = [(train_end, train_end + step) for train_end in range(initial, n, step)]",
        hint="train_end runs 6, 8, 10; test_end is train_end + 2. So [(6, 8), (8, 10), (10, 12)].",
    ),
    dict(
        prompt="`ev` has an `hour` column and an `error` column. Assign the mean error per hour (a Series indexed by hour).",
        setup="""ev = pd.DataFrame({
    "hour":  [0, 0, 0, 18, 18, 18],
    "error": [-2, -4, -3, 5, 6, 7],
})""",
        solution='answer = ev.groupby("hour")["error"].mean()',
        hint="groupby('hour') then the mean of the error column.",
    ),
    dict(
        prompt="Compute the rolling RMSE of `err` with a window of 3 rows: the square root of the rolling mean of squared errors. Keep NaN where the window is incomplete.",
        setup="err = pd.Series([1.0, 2.0, 2.0, 10.0, 1.0, 1.0])",
        solution="answer = np.sqrt((err ** 2).rolling(3).mean())",
        hint="Square first, then rolling(3).mean(), then np.sqrt.",
    ),
    dict(
        prompt="Diebold–Mariano-style test. `err_a` and `err_b` are paired errors of two models. Let d = err_a² − err_b². Assign the t-statistic of the mean of d: mean(d) / (std(d, ddof=1) / √n).",
        setup="err_a = np.array([3.0, -2.0, 4.0, -3.0, 2.0])\nerr_b = np.array([1.0, -1.0, 2.0, -2.0, 1.0])",
        solution="d = err_a ** 2 - err_b ** 2\nanswer = float(d.mean() / (d.std(ddof=1) / np.sqrt(len(d))))",
        hint="d is the per-period difference in squared error; n = 5; use ddof=1 for the standard deviation.",
    ),
    dict(
        prompt="Empirical 90% prediction interval. From the training residuals `resid`, assign `lower` = 5% quantile and `upper` = 95% quantile (np.quantile with the default interpolation). Then assign `coverage` = the fraction of test rows where `y_test` lies within `[pred_test + lower, pred_test + upper]`.",
        setup="resid = np.array([-5.0, -3.0, -2.0, -1.0, 0.0, 1.0, 2.0, 3.0, 4.0, 6.0])\ny_test = np.array([10.0, 20.0, 30.0, 40.0, 50.0])\npred_test = np.array([12.0, 18.0, 25.0, 41.0, 45.0])",
        solution="lower = float(np.quantile(resid, 0.05))\nupper = float(np.quantile(resid, 0.95))\ninside = (y_test >= pred_test + lower) & (y_test <= pred_test + upper)\ncoverage = float(inside.mean())",
        hint="np.quantile(resid, 0.05) and 0.95; then a boolean mask on the test rows and its mean.",
        answer_var=["lower", "upper", "coverage"],
    ),
    dict(
        prompt="`res` holds the RMSE of models A and B per month. Assign the number of months in which B has a **strictly lower** RMSE than A.",
        setup="""res = pd.DataFrame({
    "month":  [1, 2, 3, 4, 5, 6],
    "rmse_a": [100, 110, 105, 120, 98, 130],
    "rmse_b": [102, 108, 106, 90, 99, 100],
})""",
        solution='answer = int((res["rmse_b"] < res["rmse_a"]).sum())',
        hint="A boolean comparison of the two columns, summed.",
    ),
]
