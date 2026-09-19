"""Exercises for 02_pandas/07_pitfalls_and_debugging.ipynb — every task is "fix this"."""
import numpy as np
import pandas as pd

COMMON_SETUP = """
import numpy as np
import pandas as pd
"""

QUESTIONS = [
    dict(
        prompt="The rows of `df` are not in time order. Add a column `lag` holding the value of `v` from the **previous hour** (in time order), but keep the rows of `df` in their current order. A colleague wrote `df['lag'] = df.sort_index()['v'].shift(1).values` — that is wrong. Do it correctly.",
        setup='idx = pd.to_datetime(["2023-01-01 02:00", "2023-01-01 00:00", "2023-01-01 01:00"])\ndf = pd.DataFrame({"v": [30, 10, 20]}, index=idx)',
        solution='lag = df.sort_index()["v"].shift(1)\nanswer = df.assign(lag=lag)',
        hint="Compute the lag on the sorted frame and assign the *Series* (pandas aligns by index); .values assigns by position.",
    ),
    dict(
        prompt="`s` has unsorted timestamps. Return the hour-over-hour difference, with the result sorted by time.",
        setup='s = pd.Series([30, 10, 20], index=pd.to_datetime(["2023-01-01 02:00", "2023-01-01 00:00", "2023-01-01 01:00"]))',
        solution="answer = s.sort_index().diff()",
        hint="Sort first; diff() works on row order.",
    ),
    dict(
        prompt="`s` has 01:00 twice. Keep the **last** value for each timestamp, then compute the previous-row lag with shift(1).",
        setup='s = pd.Series([10, 11, 99, 13], index=pd.to_datetime(["2023-01-01 00:00", "2023-01-01 01:00", "2023-01-01 01:00", "2023-01-01 02:00"]))',
        solution='answer = s[~s.index.duplicated(keep="last")].shift(1)',
        hint="Mask with ~s.index.duplicated(keep='last'), then shift.",
    ),
    dict(
        prompt="The `price` column was read as text and contains the word 'missing'. Assign the mean of the numeric prices (a float), ignoring the bad value.",
        setup='df = pd.DataFrame({"price": ["70.0", "80.0", "missing", "90.0"]})',
        solution='answer = float(pd.to_numeric(df["price"], errors="coerce").mean())',
        hint="pd.to_numeric(..., errors='coerce') then .mean().",
    ),
    dict(
        prompt="Daily totals of `s`: the second day has no data at all, so its total must be NaN, not 0.",
        setup='s = pd.Series([1.0, 2.0, np.nan, np.nan], index=pd.to_datetime(["2023-01-01 00:00", "2023-01-01 12:00", "2023-01-02 00:00", "2023-01-02 12:00"]))',
        solution='answer = s.resample("D").sum(min_count=1)',
        hint="sum(min_count=1) returns NaN when there is nothing to sum.",
    ),
    dict(
        prompt="Set `x` to 0 wherever `flag` is True. `df[df.flag]['x'] = 0` does nothing — write the version that works, returning a **new** frame (do not mutate `df`).",
        setup='df = pd.DataFrame({"x": [5, 6, 7], "flag": [True, False, True]})',
        solution='answer = df.copy()\nanswer.loc[answer["flag"], "x"] = 0',
        hint="answer = df.copy(); answer.loc[mask, 'x'] = 0",
    ),
    dict(
        prompt="`right` has duplicated keys, so a plain merge multiplies the rows of `left`. Deduplicate `right` on `key` (keep the first), then left-merge so the result has exactly as many rows as `left`.",
        setup='left = pd.DataFrame({"key": ["a", "b", "c"], "v": [1, 2, 3]})\nright = pd.DataFrame({"key": ["a", "a", "b"], "w": [10, 11, 20]})',
        solution='answer = left.merge(right.drop_duplicates("key"), on="key", how="left")',
        hint="right.drop_duplicates('key') before merging.",
    ),
    dict(
        prompt="`df.time` is tz-naive but the values are UTC; `cutoff` is tz-aware. Count the rows with time >= cutoff (an int). Comparing them directly raises, so localise first.",
        setup='df = pd.DataFrame({"time": pd.to_datetime(["2023-01-01 00:00", "2023-01-01 06:00", "2023-01-01 12:00"])})\ncutoff = pd.Timestamp("2023-01-01 06:00", tz="UTC")',
        solution='t = df["time"].dt.tz_localize("UTC")\nanswer = int((t >= cutoff).sum())',
        hint=".dt.tz_localize('UTC') on the naive column, then compare.",
    ),
    dict(
        prompt="Fill the gaps in `s` using **only past** information (a leak-free fill).",
        setup='s = pd.Series([10.0, np.nan, np.nan, 40.0, np.nan], index=pd.date_range("2023-01-01", periods=5, freq="h"))',
        solution="answer = s.ffill()",
        hint="ffill() copies the last seen value forward; bfill() would use the future.",
    ),
    dict(
        prompt="The dates in `raw` are written day-first ('03/04/2023' is 3 April). Parse them into a datetime Series correctly.",
        setup='raw = pd.Series(["03/04/2023", "15/04/2023", "01/05/2023"])',
        solution="answer = pd.to_datetime(raw, dayfirst=True)",
        hint="pd.to_datetime(..., dayfirst=True) (or format='%d/%m/%Y').",
    ),
    dict(
        prompt="`a` and `b` should be equal but differ by floating-point noise. Count how many pairs are equal within 1e-9 (an int). `(a == b).sum()` gives the wrong answer.",
        setup="a = pd.Series([0.1 + 0.2, 1.0, 2.5])\nb = pd.Series([0.3, 1.0, 2.5000001])",
        solution="answer = int(np.isclose(a, b, atol=1e-9).sum())",
        hint="np.isclose(a, b, atol=1e-9)",
    ),
    dict(
        prompt="Reindexing `s` to labels 0..3 introduces a NaN and silently turns the integers into floats. Return the reindexed Series with the nullable integer dtype `Int64` so the existing values stay integers.",
        setup="s = pd.Series([1, 2, 3])",
        solution='answer = s.reindex([0, 1, 2, 3]).astype("Int64")',
        hint='.reindex([0, 1, 2, 3]).astype("Int64")',
        check_fn=lambda got, exp: (isinstance(got, pd.Series) and str(got.dtype) == "Int64" and got.equals(exp),
                                   "" if isinstance(got, pd.Series) and str(got.dtype) == "Int64" else "dtype must be Int64"),
    ),
]
