"""Exercises for 02_pandas/06_time_series.ipynb"""
import numpy as np
import pandas as pd

COMMON_SETUP = """
import numpy as np
import pandas as pd
"""

GAPPY = 'idx = pd.to_datetime(["2023-01-01 00:00", "2023-01-01 01:00", "2023-01-01 03:00", "2023-01-01 04:00", "2023-01-01 05:00"])\ns = pd.Series([10, 11, 13, 14, 15], index=idx)   # 02:00 is missing'

QUESTIONS = [
    dict(
        prompt="`df` has a `time` column stored as text, in random order. Return a frame indexed by a real DatetimeIndex, sorted by time.",
        setup='df = pd.DataFrame({"time": ["2023-01-01 02:00", "2023-01-01 00:00", "2023-01-01 01:00"], "mw": [12, 10, 11]})',
        solution='answer = df.assign(time=pd.to_datetime(df["time"])).set_index("time").sort_index()',
        hint="pd.to_datetime the column, set_index, sort_index.",
    ),
    dict(
        prompt="`s72` is 72 hourly values over three days. Select every row of the second day (2023-01-02) using a partial date string.",
        setup='s72 = pd.Series(range(72), index=pd.date_range("2023-01-01", periods=72, freq="h"))',
        solution='answer = s72.loc["2023-01-02"]',
        hint='A DatetimeIndex accepts s.loc["YYYY-MM-DD"] and returns the whole day.',
    ),
    dict(
        prompt="From the `time` column of `df`, build a DataFrame with two integer columns `hour` and `dow` (Monday = 0).",
        setup='df = pd.DataFrame({"time": pd.to_datetime(["2023-01-01 07:00", "2023-01-02 18:00", "2023-01-07 23:00"])})',
        solution='answer = pd.DataFrame({"hour": df["time"].dt.hour, "dow": df["time"].dt.dayofweek})',
        hint="The .dt accessor: .dt.hour and .dt.dayofweek.",
    ),
    dict(
        prompt="It is 12:00 local time in London on 2023-07-01 (British Summer Time). What is the hour in UTC? Assign the integer.",
        setup="",
        solution='answer = int(pd.Timestamp("2023-07-01 12:00", tz="Europe/London").tz_convert("UTC").hour)',
        hint='pd.Timestamp("...", tz="Europe/London").tz_convert("UTC").hour',
    ),
    dict(
        prompt="`s` is hourly but 02:00 is missing. Reindex it to the complete hourly grid from its first to its last timestamp (assign to `on_grid`) and count the missing hours (assign the int to `n_missing`).",
        setup=GAPPY,
        solution='grid = pd.date_range(s.index.min(), s.index.max(), freq="h")\non_grid = s.reindex(grid)\nn_missing = int(on_grid.isna().sum())',
        hint='pd.date_range(start, end, freq="h") then reindex; missing rows are NaN.',
        answer_var=["on_grid", "n_missing"],
    ),
    dict(
        prompt="Aggregate `s` into 2-hour totals with the default settings (each bar labelled by its **start**).",
        setup=GAPPY,
        solution='answer = s.resample("2h").sum()',
        hint='resample("2h").sum()',
    ),
    dict(
        prompt="Aggregate `s` into 2-hour totals where each bar is labelled by its **end** timestamp and includes that end timestamp (`closed='right'`).",
        setup=GAPPY,
        solution='answer = s.resample("2h", label="right", closed="right").sum()',
        hint='resample("2h", label="right", closed="right")',
    ),
    dict(
        prompt="Move every value of `s` one hour **later by timestamp** (not by row), so that the value for 01:00 lands on 02:00 even though 02:00 was not in the index.",
        setup=GAPPY,
        solution='answer = s.shift(freq="1h")',
        hint='shift(freq="1h") shifts the index labels; shift(1) shifts by rows.',
    ),
    dict(
        prompt="Compute the trailing mean of `s` over a **3-hour time window** (so the gap at 02:00 shortens the window), not over 3 rows.",
        setup=GAPPY,
        solution='answer = s.rolling("3h").mean()',
        hint='rolling("3h") is a time-based window; rolling(3) counts rows.',
    ),
    dict(
        prompt="Build a leak-free feature: for each hour, the mean of the **previous three** values of `y` (not including the current one).",
        setup='y = pd.Series([10, 20, 30, 40, 50, 60], index=pd.date_range("2023-01-01", periods=6, freq="h"))',
        solution="answer = y.shift(1).rolling(3).mean()",
        hint="shift(1) first so the window ends at the previous row, then rolling(3).mean().",
    ),
    dict(
        prompt="Select the rows of `s72` whose time of day is between 17:00 and 19:00 inclusive.",
        setup='s72 = pd.Series(range(72), index=pd.date_range("2023-01-01", periods=72, freq="h"))',
        solution='answer = s72.between_time("17:00", "19:00")',
        hint='between_time("17:00", "19:00")',
    ),
    dict(
        prompt="Real data: load `../data/hourly_power_clean.csv` and compute the average `consumption_mwh` for each hour of the day (a Series indexed 0..23).",
        setup='df = pd.read_csv("../data/hourly_power_clean.csv", parse_dates=["time"])',
        solution='answer = df.groupby(df["time"].dt.hour)["consumption_mwh"].mean()',
        hint='groupby(df["time"].dt.hour)["consumption_mwh"].mean()',
    ),
    dict(
        prompt="Real data: load `../data/hourly_power_raw.csv` (unsorted, with duplicates). Parse `time` as UTC, drop duplicate timestamps, and count how many hours are missing between the first and last timestamp. Assign the int.",
        setup='raw = pd.read_csv("../data/hourly_power_raw.csv")',
        solution='t = pd.to_datetime(raw["time"], utc=True)\nr = raw.assign(time=t).drop_duplicates("time").set_index("time").sort_index()\ngrid = pd.date_range(r.index.min(), r.index.max(), freq="h")\nanswer = int(r["consumption_mwh"].reindex(grid).isna().sum())',
        hint="to_datetime(utc=True) → drop_duplicates('time') → set_index → reindex to a date_range grid → count NaN.",
    ),
]
