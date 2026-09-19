"""Exercises for 02_pandas/02_clean_missing_dtypes.ipynb"""
import numpy as np
import pandas as pd

COMMON_SETUP = """
import numpy as np
import pandas as pd

raw = pd.DataFrame({
    "time": ["2023-01-01 02:00", "2023-01-01 00:00", "2023-01-01 01:00", "2023-01-01 02:00", "2023-01-01 04:00"],
    "cons": [30.0, 10.0, 20.0, 31.0, 50.0],
    "temp": [5.0, -999.0, 6.0, 5.5, -999.0],
    "price": ["70.1", "missing", "23.2", "70.5", "115.8"],
    "region": ["GB", "GB", "GB", "GB", "GB"],
})
"""

QUESTIONS = [
    dict(
        prompt="Convert `raw['time']` to timezone-aware UTC timestamps. Assign the Series.",
        setup="",
        solution='answer = pd.to_datetime(raw["time"], utc=True)',
        hint="pd.to_datetime(..., utc=True)",
        check_fn=lambda got, exp: (isinstance(got, pd.Series) and str(got.dtype) == "datetime64[ns, UTC]" and got.equals(exp),
                                   "expected dtype datetime64[ns, UTC] (pass utc=True)"),
    ),
    dict(
        prompt="Parse the time column (UTC), sort the frame by time and reset the index so it runs 0..n-1. Assign the DataFrame.",
        setup="",
        solution='df = raw.copy()\ndf["time"] = pd.to_datetime(df["time"], utc=True)\nanswer = df.sort_values("time").reset_index(drop=True)',
        hint="sort_values('time').reset_index(drop=True)",
    ),
    dict(
        prompt="`df` is sorted. Two rows share the timestamp 02:00 with slightly different values. Keep only the **last** row for each time. Assign the DataFrame (reset the index).",
        setup='df = raw.copy()\ndf["time"] = pd.to_datetime(df["time"], utc=True)\ndf = df.sort_values("time").reset_index(drop=True)',
        solution='answer = df.drop_duplicates("time", keep="last").reset_index(drop=True)',
        hint="drop_duplicates(subset, keep='last') — the default keep='first' would keep the wrong row.",
    ),
    dict(
        prompt="Convert `df['price']` (text) to numbers so that bad values become NaN. Assign the numeric Series to `answer` and the number of NaN it contains to `n_bad`.",
        setup="df = raw.copy()",
        solution='answer = pd.to_numeric(df["price"], errors="coerce")\nn_bad = int(answer.isna().sum())',
        hint="pd.to_numeric(..., errors='coerce')",
        answer_var=["answer", "n_bad"],
    ),
    dict(
        prompt="`df['temp']` uses -999 as a sentinel. Assign a Series where every value below -100 is NaN and everything else is unchanged.",
        setup="df = raw.copy()",
        solution='answer = df["temp"].mask(df["temp"] <= -100)',
        hint="mask(condition) replaces where True; where(condition) keeps where True.",
    ),
    dict(
        prompt="Assign a **sorted list** of the columns of `df` that are constant (should be dropped).",
        setup="df = raw.copy()",
        solution="answer = sorted(df.columns[df.nunique() == 1].tolist())",
        hint="nunique() == 1",
    ),
    dict(
        prompt="`s` is hourly consumption indexed by time, with 03:00 missing. Reindex it to the complete hourly grid from 00:00 to 04:00 and assign the reindexed Series to `answer` and the number of missing hours to `n_missing`.",
        setup='idx = pd.to_datetime(["2023-01-01 00:00", "2023-01-01 01:00", "2023-01-01 02:00", "2023-01-01 04:00"], utc=True)\ns = pd.Series([10.0, 20.0, 30.0, 50.0], index=idx)',
        solution='grid = pd.date_range(s.index.min(), s.index.max(), freq="h")\nanswer = s.reindex(grid)\nn_missing = int(answer.isna().sum())',
        hint="pd.date_range(start, end, freq='h') then s.reindex(grid).",
        answer_var=["answer", "n_missing"],
    ),
    dict(
        prompt="Forward-fill `s`, but never fill more than **one** consecutive gap (the second NaN in a run must stay NaN).",
        setup="s = pd.Series([10.0, np.nan, np.nan, 40.0, np.nan, 60.0])",
        solution="answer = s.ffill(limit=1)",
        hint="ffill(limit=...)",
    ),
    dict(
        prompt="Fill the gaps in `s` by linear interpolation. Assign the Series.",
        setup="s = pd.Series([10.0, np.nan, np.nan, 40.0, np.nan, 60.0])",
        solution="answer = s.interpolate()",
        hint="interpolate() with default method='linear'.",
    ),
    dict(
        prompt="Drop the rows of `df` where `cons` **or** `temp` is NaN (a NaN in `note` alone must not drop the row). Assign the DataFrame.",
        setup='df = pd.DataFrame({"cons": [1.0, np.nan, 3.0, 4.0], "temp": [5.0, 6.0, np.nan, 8.0], "note": [np.nan, "a", "b", np.nan]})',
        solution='answer = df.dropna(subset=["cons", "temp"])',
        hint="dropna(subset=[...])",
    ),
    dict(
        prompt="Fill the NaN in `df['value']` with the **median of the same hour** (group by `hour`). Assign the filled Series.",
        setup='df = pd.DataFrame({"hour": [0, 0, 0, 1, 1, 1], "value": [10.0, np.nan, 30.0, 100.0, 120.0, np.nan]})',
        solution='answer = df["value"].fillna(df.groupby("hour")["value"].transform("median"))',
        hint="groupby('hour')['value'].transform('median') gives a full-length Series to fillna with.",
    ),
    dict(
        prompt="Clip `s` to its 5th and 95th percentiles (values below the 5th become the 5th, above the 95th become the 95th). Assign the Series.",
        setup="s = pd.Series([1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0, 100.0])",
        solution="lo, hi = s.quantile(0.05), s.quantile(0.95)\nanswer = s.clip(lower=lo, upper=hi)",
        hint="s.quantile(0.05), s.quantile(0.95), then s.clip(lower=..., upper=...)",
    ),
    dict(
        prompt="Clean `region`: strip spaces, make the first letter upper-case and the rest lower-case. Assign the Series.",
        setup='region = pd.Series([" london", "WALES ", "North", "scotland"])',
        solution="answer = region.str.strip().str.capitalize()",
        hint=".str.strip() then .str.capitalize() (or .str.title() here).",
    ),
    dict(
        prompt="Real data: clean `../data/hourly_power_raw.csv` enough to count the missing hours: parse `time` as UTC, drop duplicate times, and reindex to the full hourly grid from the first to the last timestamp. Assign the number of missing hours as an integer.",
        setup='raw_file = pd.read_csv("../data/hourly_power_raw.csv")',
        solution='t = pd.to_datetime(raw_file["time"], utc=True)\ns = raw_file.set_index(t)["consumption_mwh"]\ns = s[~s.index.duplicated()].sort_index()\ngrid = pd.date_range(s.index.min(), s.index.max(), freq="h")\nanswer = int(s.reindex(grid).isna().sum())',
        hint="Dedupe the index first (reindex refuses duplicates), then date_range + reindex + isna().sum(). The answer is 78.",
    ),
]
