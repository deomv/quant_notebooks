"""Exercises for 02_pandas/08_debugging_toolkit.ipynb"""
import numpy as np
import pandas as pd

COMMON_SETUP = """
import numpy as np
import pandas as pd
"""

QUESTIONS = [
    dict(
        prompt="Vital signs of `df`: assign a tuple `(n_rows, n_cols, n_nan, n_duplicate_rows)` of four ints.",
        setup='df = pd.DataFrame({"a": [1, 2, 2, np.nan], "b": ["x", "y", "y", "z"]})',
        solution="answer = (len(df), df.shape[1], int(df.isna().sum().sum()), int(df.duplicated().sum()))",
        hint="len, shape[1], isna().sum().sum(), duplicated().sum()",
    ),
    dict(
        prompt="Is the index of `df` both unique and sorted ascending? Assign a single boolean.",
        setup='df = pd.DataFrame({"v": [1, 2, 3]}, index=pd.to_datetime(["2023-01-01 02:00", "2023-01-01 00:00", "2023-01-01 01:00"]))',
        solution="answer = bool(df.index.is_unique and df.index.is_monotonic_increasing)",
        hint="index.is_unique and index.is_monotonic_increasing",
    ),
    dict(
        prompt="`new` should equal `old` but one cell differs. Return the DataFrame that shows only the differing cells (pandas has a method for this).",
        setup='old = pd.DataFrame({"a": [1, 2, 3], "b": [10, 20, 30]})\nnew = pd.DataFrame({"a": [1, 2, 3], "b": [10, 25, 30]})',
        solution="answer = old.compare(new)",
        hint="old.compare(new)",
    ),
    dict(
        prompt="Outer-merge `left` and `right` on `key` with `indicator=True` and assign a dict counting the `_merge` categories, e.g. `{'both': .., 'left_only': .., 'right_only': ..}`.",
        setup='left = pd.DataFrame({"key": ["a", "b", "c"], "v": [1, 2, 3]})\nright = pd.DataFrame({"key": ["b", "c", "d"], "w": [20, 30, 40]})',
        solution='m = left.merge(right, on="key", how="outer", indicator=True)\nanswer = {str(k): int(v) for k, v in m["_merge"].value_counts().items()}',
        hint='value_counts() of the "_merge" column, converted to a dict of ints.',
    ),
    dict(
        prompt="List the columns of `df` that are stored as `object` dtype (a sorted list of column names). Those are the ones that will break .mean().",
        setup='df = pd.DataFrame({"price": ["1.0", "2.0"], "mw": [1.0, 2.0], "site": ["A", "B"], "flag": [True, False]})',
        solution='answer = sorted(df.columns[df.dtypes == "object"])',
        hint='df.dtypes == "object" is a boolean mask over columns.',
    ),
    dict(
        prompt="List the columns of `df` that are constant (only one distinct value, NaN counted as a value) — a sorted list of names.",
        setup='df = pd.DataFrame({"region": ["GB", "GB", "GB"], "mw": [1, 2, 3], "unit": ["MW", "MW", "MW"], "flag": [np.nan, np.nan, np.nan]})',
        solution="answer = sorted(c for c in df.columns if df[c].nunique(dropna=False) == 1)",
        hint="nunique(dropna=False) == 1 per column.",
    ),
    dict(
        prompt="Leakage test. `f` is a feature and `y` the target. Compute the correlation of `f` with `y.shift(-1)`, `y` and `y.shift(1)` and assign a dict `{'lead': .., 'same': .., 'lag': ..}` of floats. A feature that correlates most with the *future* target is leaking.",
        setup="y = pd.Series([1.0, 3.0, 2.0, 5.0, 4.0, 6.0, 5.0, 8.0])\nf = y.shift(-1)   # suspicious feature",
        solution='answer = {"lead": float(f.corr(y.shift(-1))), "same": float(f.corr(y)), "lag": float(f.corr(y.shift(1)))}',
        hint="Series.corr() ignores NaN pairs; shift(-1) is the future.",
    ),
    dict(
        prompt="Recompute `fast` the slow way: build a Series with the same index whose value at position i is the mean of y[i-2], y[i-1], y[i] (NaN for the first two rows), using a plain Python loop. If it matches `fast`, the vectorised version is right.",
        setup="y = pd.Series([2.0, 4.0, 6.0, 8.0, 10.0])\nfast = y.rolling(3).mean()",
        solution="vals = []\nfor i in range(len(y)):\n    if i < 2:\n        vals.append(np.nan)\n    else:\n        vals.append((y.iloc[i-2] + y.iloc[i-1] + y.iloc[i]) / 3)\nanswer = pd.Series(vals, index=y.index)",
        hint="Loop over positions with .iloc; append NaN for i < 2.",
    ),
    dict(
        prompt="Show the rows of `df` from one hour before to one hour after the bad timestamp `t` (inclusive).",
        setup='df = pd.DataFrame({"v": range(6)}, index=pd.date_range("2023-01-01", periods=6, freq="h"))\nt = pd.Timestamp("2023-01-01 03:00")',
        solution='answer = df.loc[t - pd.Timedelta("1h"): t + pd.Timedelta("1h")]',
        hint="df.loc[t - pd.Timedelta('1h') : t + pd.Timedelta('1h')] — label slices are inclusive.",
    ),
    dict(
        prompt="Are `a` and `b` equal if you ignore the dtype difference (ints vs floats)? Use `pd.testing.assert_frame_equal(..., check_dtype=False)` inside try/except and assign True if it passes, False if it raises.",
        setup='a = pd.DataFrame({"x": [1, 2, 3]})\nb = pd.DataFrame({"x": [1.0, 2.0, 3.0]})',
        solution="try:\n    pd.testing.assert_frame_equal(a, b, check_dtype=False)\n    answer = True\nexcept AssertionError:\n    answer = False",
        hint="assert_frame_equal returns None when equal and raises AssertionError otherwise.",
    ),
    dict(
        prompt="Trace the shape through a pipeline: assign a list of three tuples — the shape of `df`, the shape after dropping rows with any NaN, and the shape after then keeping only rows with `mw > 1`.",
        setup='df = pd.DataFrame({"mw": [1.0, 2.0, np.nan, 3.0], "site": ["A", "B", "C", "D"]})',
        solution='step1 = df.dropna()\nstep2 = step1[step1["mw"] > 1]\nanswer = [df.shape, step1.shape, step2.shape]',
        hint="Keep each intermediate frame in its own variable and collect .shape.",
    ),
    dict(
        prompt="Real data: load `../data/hourly_power_clean.csv`. Assign the R² of the **persistence** forecast (predict each hour with the previous hour's consumption) as a float, using `sklearn.metrics.r2_score` on the rows where both exist.",
        setup='from sklearn.metrics import r2_score\ndf = pd.read_csv("../data/hourly_power_clean.csv", parse_dates=["time"])',
        solution='y = df["consumption_mwh"]\npred = y.shift(1)\nmask = pred.notna()\nanswer = float(r2_score(y[mask], pred[mask]))',
        hint="pred = y.shift(1); drop the first row (NaN) before scoring.",
        tol=1e-4,
    ),
]
