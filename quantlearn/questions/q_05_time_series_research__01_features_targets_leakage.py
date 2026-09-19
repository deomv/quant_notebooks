"""Exercises for 05_time_series_research/01_features_targets_leakage.ipynb"""
import numpy as np
import pandas as pd

COMMON_SETUP = """
import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler
"""

INTRO = "Convention used throughout: at time t you may use values up to and including t, never later."

QUESTIONS = [
    dict(
        prompt="`y` is an hourly series. Build the target for a forecast **2 hours ahead**: at each timestamp the value that will be observed 2 hours later. The last two entries must be NaN.",
        setup="y = pd.Series([10, 20, 30, 40, 50, 60, 70, 80], index=pd.date_range('2023-01-01', periods=8, freq='h'))",
        solution="answer = y.shift(-2)",
        hint="A future value is a negative shift. shift(+2) would give the past.",
    ),
    dict(
        prompt="Build a DataFrame with two lag features of `y`: `lag1` (value 1 hour earlier) and `lag2` (value 2 hours earlier). Keep the same index as `y`.",
        setup="y = pd.Series([10, 20, 30, 40, 50, 60, 70, 80], index=pd.date_range('2023-01-01', periods=8, freq='h'))",
        solution='answer = pd.DataFrame({"lag1": y.shift(1), "lag2": y.shift(2)})',
        hint="Past values are positive shifts; put both Series in a dict inside pd.DataFrame.",
    ),
    dict(
        prompt="Build the **honest** rolling feature: the mean of the 3 values **before** t (not including t). At 03:00 it must equal the mean of 00:00, 01:00 and 02:00.",
        setup="y = pd.Series([10, 20, 30, 40, 50, 60, 70, 80], index=pd.date_range('2023-01-01', periods=8, freq='h'))",
        solution="answer = y.shift(1).rolling(3).mean()",
        hint="rolling(3) ends at the current row. Shift by one first so the window ends at t-1.",
    ),
    dict(
        prompt="Compute the difference between the **leaky** rolling feature `y.rolling(3).mean()` and the honest one from the previous task, at 03:00. Assign that single number.",
        setup="y = pd.Series([10, 20, 30, 40, 50, 60, 70, 80], index=pd.date_range('2023-01-01', periods=8, freq='h'))",
        solution="leaky = y.rolling(3).mean()\nhonest = y.shift(1).rolling(3).mean()\nanswer = float(leaky.loc['2023-01-01 03:00'] - honest.loc['2023-01-01 03:00'])",
        hint="Leaky at 03:00 = mean(20, 30, 40); honest = mean(10, 20, 30). The difference is 10.",
    ),
    dict(
        prompt="Build one frame with column `lag1 = y.shift(1)` and column `target = y.shift(-1)`, drop the rows with any NaN **once**, and assign the **index** of the resulting frame.",
        setup="y = pd.Series([10, 20, 30, 40, 50, 60, 70, 80], index=pd.date_range('2023-01-01', periods=8, freq='h'))",
        solution='frame = pd.DataFrame({"lag1": y.shift(1), "target": y.shift(-1)}).dropna()\nanswer = frame.index',
        hint="Put both columns in one DataFrame, call dropna() on the frame, then take .index. The first and last hour disappear.",
    ),
    dict(
        prompt="Encode the hour of day cyclically. From `hours` build a DataFrame with columns `hour_sin` = sin(2π·hour/24) and `hour_cos` = cos(2π·hour/24).",
        setup="hours = pd.Series([0, 6, 12, 18])",
        solution='answer = pd.DataFrame({"hour_sin": np.sin(2 * np.pi * hours / 24), "hour_cos": np.cos(2 * np.pi * hours / 24)})',
        hint="np.sin and np.cos work on a whole Series; the angle is 2·π·hour/24.",
        tol=1e-6,
    ),
    dict(
        prompt=(
            "Forecasts for 15:00 and 16:00 were issued at two origins (00:00 and 12:00). "
            "Your decision is made at 06:00, so only forecasts with `origin_datetime <= decision_time` may be used. "
            "Assign a Series of the usable `temp_forecast_c` values indexed by `forecast_datetime` (one value per target hour, the latest usable origin)."
        ),
        setup="""fc = pd.DataFrame({
    "origin_datetime":   pd.to_datetime(["2023-01-01 00:00", "2023-01-01 00:00", "2023-01-01 12:00", "2023-01-01 12:00"]),
    "forecast_datetime": pd.to_datetime(["2023-01-01 15:00", "2023-01-01 16:00", "2023-01-01 15:00", "2023-01-01 16:00"]),
    "temp_forecast_c":   [5.0, 6.0, 7.0, 8.0],
})
decision_time = pd.Timestamp("2023-01-01 06:00")""",
        solution="""usable = fc[fc["origin_datetime"] <= decision_time].sort_values("origin_datetime")
answer = usable.groupby("forecast_datetime")["temp_forecast_c"].last()""",
        hint="Filter on origin_datetime first, then keep the last origin per forecast_datetime. Taking the latest origin without the filter gives 7 and 8, which were not known at 06:00.",
    ),
    dict(
        prompt="Fit a `StandardScaler` on `train` **only** and transform `test`. Assign the transformed test array (shape (2, 1)).",
        setup="train = np.array([[1.0], [2.0], [3.0], [4.0]])\ntest = np.array([[5.0], [6.0]])",
        solution="scaler = StandardScaler().fit(train)\nanswer = scaler.transform(test)",
        hint="scaler.fit(train) then scaler.transform(test). fit_transform on the concatenation would leak the test mean.",
        tol=1e-6,
    ),
    dict(
        prompt="`df` has a `group`, a value `y` and a flag `is_train`. Build the group-mean feature using **train rows only**: every row (train or test) gets the mean of `y` over the train rows of its group. Assign the resulting Series (same index as `df`).",
        setup="""df = pd.DataFrame({
    "group":    ["A", "A", "B", "B", "A", "B"],
    "y":        [10, 20, 100, 200, 30, 300],
    "is_train": [True, True, True, True, False, False],
})""",
        solution='means = df[df["is_train"]].groupby("group")["y"].mean()\nanswer = df["group"].map(means)',
        hint="Compute the group means on df[df.is_train], then map the group column through them. groupby().transform on the full frame would include the test rows.",
    ),
    dict(
        prompt="`s` has gaps. Fill them using **only past** values (a gap takes the last observed value before it).",
        setup="s = pd.Series([1.0, np.nan, np.nan, 4.0, np.nan, 6.0], index=pd.date_range('2023-01-01', periods=6, freq='h'))",
        solution="answer = s.ffill()",
        hint="ffill copies the previous value forward; bfill and interpolate use values from the future.",
    ),
    dict(
        prompt="Aggregate the half-hourly values `hh` into hourly **sums** where each hour is labelled by its **start** time (00:00 covers 00:00 and 00:30).",
        setup="hh = pd.Series([1, 2, 3, 4], index=pd.date_range('2023-01-01 00:00', periods=4, freq='30min'))",
        solution='answer = hh.resample("h").sum()',
        hint="resample('h').sum() with the default label='left', closed='left'. label='right' would stamp the hour with its end.",
    ),
    dict(
        prompt=(
            "Real data. Load `../data/weather_forecasts.csv` (parse both datetime columns). For every hour of 2023-03-01 (24 target hours, UTC), "
            "the decision is made at 2023-02-28 12:00 UTC. Pick, per target hour, the forecast with the **latest origin_datetime that is <= the decision time**. "
            "Assign a Series of `horizon_h` indexed by `forecast_datetime` for those 24 hours (sorted by time)."
        ),
        setup='fc = pd.read_csv("../data/weather_forecasts.csv", parse_dates=["origin_datetime", "forecast_datetime"])\ndecision_time = pd.Timestamp("2023-02-28 12:00", tz="UTC")\ntargets = pd.date_range("2023-03-01 00:00", periods=24, freq="h", tz="UTC")',
        solution="""usable = fc[(fc["origin_datetime"] <= decision_time) & fc["forecast_datetime"].isin(targets)]
usable = usable.sort_values("origin_datetime")
answer = usable.groupby("forecast_datetime")["horizon_h"].last().sort_index()""",
        hint="Filter origin_datetime <= decision_time and forecast_datetime in targets, sort by origin, groupby forecast_datetime and take last. Horizons should run 12..35.",
    ),
]
