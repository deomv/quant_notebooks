"""Exercises for 05_time_series_research/03_end_to_end_template.ipynb"""
import numpy as np
import pandas as pd

COMMON_SETUP = """
import numpy as np
import pandas as pd
from sklearn.linear_model import Ridge
from sklearn.metrics import mean_squared_error

raw = pd.read_csv("../data/hourly_power_raw.csv")
"""

INTRO = "These tasks walk through the raw hourly file `hourly_power_raw.csv` in the order you would in the interview: inspect, clean, define, feature, split, baseline, model, diagnose."

CLEAN = """
t = pd.to_datetime(raw["time"], utc=True)
df = raw.assign(time=t).sort_values("time").drop_duplicates("time", keep="last").set_index("time")
df["price_eur_mwh"] = pd.to_numeric(df["price_eur_mwh"], errors="coerce")
df.loc[df["temp_c"] <= -100, "temp_c"] = np.nan
full_idx = pd.date_range(df.index.min(), df.index.max(), freq="h")
df = df.reindex(full_idx)
df.index.name = "time"
"""

FEATURES = CLEAN + """
d = pd.DataFrame(index=df.index)
d["y"] = df["consumption_mwh"].shift(-24)
d["lag24"] = df["consumption_mwh"].shift(24)
d["lag168"] = df["consumption_mwh"].shift(168)
d["roll24"] = df["consumption_mwh"].shift(1).rolling(24).mean()
d["temp_c"] = df["temp_c"]
d["hour"] = d.index.hour
d["dow"] = d.index.dayofweek
d = d.dropna()
split = int(len(d) * 0.8)
train, test = d.iloc[:split], d.iloc[split:]
features = ["lag24", "lag168", "roll24", "temp_c", "hour", "dow"]
"""

QUESTIONS = [
    dict(
        prompt="Parse the `time` column of `raw` as UTC timestamps and assign the number of **duplicated timestamps** (rows whose timestamp already appeared earlier).",
        setup="",
        solution='t = pd.to_datetime(raw["time"], utc=True)\nanswer = int(t.duplicated().sum())',
        hint="pd.to_datetime(..., utc=True), then .duplicated().sum().",
    ),
    dict(
        prompt="Assign the number of `temp_c` values equal to the sentinel -999.",
        setup="",
        solution='answer = int((raw["temp_c"] == -999).sum())',
        hint="A boolean comparison summed. describe() would have shown the -999 as the min.",
    ),
    dict(
        prompt="`price_eur_mwh` is text. Convert it to numbers so that bad values become NaN, and assign how many NaN that produces.",
        setup="",
        solution='price = pd.to_numeric(raw["price_eur_mwh"], errors="coerce")\nanswer = int(price.isna().sum())',
        hint="pd.to_numeric(errors='coerce') then isna().sum().",
    ),
    dict(
        prompt="Sort by time, drop duplicated timestamps (keep the last), set `time` as the index, reindex to the complete hourly grid from the first to the last timestamp, and assign the number of hours whose `consumption_mwh` is missing.",
        setup='t = pd.to_datetime(raw["time"], utc=True)',
        solution="""df = raw.assign(time=t).sort_values("time").drop_duplicates("time", keep="last").set_index("time")
full_idx = pd.date_range(df.index.min(), df.index.max(), freq="h")
df = df.reindex(full_idx)
answer = int(df["consumption_mwh"].isna().sum())""",
        hint="pd.date_range(min, max, freq='h') is the grid; reindex to it and count NaN in consumption. Expect a whole day plus scattered hours.",
    ),
    dict(
        prompt=(
            "`df` is the cleaned hourly frame on the full grid. The problem: at hour t predict consumption at t+24. "
            "Build `d` with columns `y` = consumption 24 h ahead, `lag24`, `lag168`, `roll24` = mean of the 24 hours before t (shift first), `temp_c`, `hour`, `dow`; drop rows with any NaN once. "
            "Assign the first timestamp of `d`."
        ),
        setup=CLEAN,
        solution="""d = pd.DataFrame(index=df.index)
d["y"] = df["consumption_mwh"].shift(-24)
d["lag24"] = df["consumption_mwh"].shift(24)
d["lag168"] = df["consumption_mwh"].shift(168)
d["roll24"] = df["consumption_mwh"].shift(1).rolling(24).mean()
d["temp_c"] = df["temp_c"]
d["hour"] = d.index.hour
d["dow"] = d.index.dayofweek
d = d.dropna()
answer = d.index[0]""",
        hint="lag168 needs 168 earlier rows, so the first valid row is 168 hours after the start (later if a temp_c NaN sits there).",
        check_fn=lambda got, exp: (pd.Timestamp(got) == exp, "" if pd.Timestamp(got) == exp else f"expected {exp}"),
    ),
    dict(
        prompt="Chronological split: the first 80% of the rows of `d` (`split = int(len(d) * 0.8)`) are train, the rest test. Assign `n_test` = number of test rows and `naive_rmse` = RMSE on the test rows of the naive forecast `lag24` against `y`.",
        setup=FEATURES,
        solution='n_test = int(len(test))\nnaive_rmse = float(np.sqrt(np.mean((test["y"] - test["lag24"]) ** 2)))',
        hint="test = d.iloc[split:]. The naive prediction is the lag24 column itself.",
        answer_var=["n_test", "naive_rmse"],
        tol=1e-3,
    ),
    dict(
        prompt="Fit `Ridge(alpha=1.0)` on the train rows using `features`, predict the test rows, and assign the test RMSE.",
        setup=FEATURES,
        solution='model = Ridge(alpha=1.0).fit(train[features], train["y"])\npred = model.predict(test[features])\nanswer = float(np.sqrt(mean_squared_error(test["y"], pred)))',
        hint="Ridge(alpha=1.0).fit(train[features], train['y']); predict on test[features]; RMSE.",
        tol=1e-3,
    ),
    dict(
        prompt="Using the fitted Ridge model `model` and its test predictions `pred`, assign the **hour of day** (0-23) with the largest mean absolute error on the test set.",
        setup=FEATURES + 'model = Ridge(alpha=1.0).fit(train[features], train["y"])\npred = model.predict(test[features])',
        solution='ev = pd.DataFrame({"abs_err": np.abs(test["y"] - pred), "hour": test.index.hour})\nanswer = int(ev.groupby("hour")["abs_err"].mean().idxmax())',
        hint="Put |error| and hour in a frame, groupby hour, mean, idxmax.",
    ),
    dict(
        prompt="Assign the name (string) of the feature with the largest **absolute** Ridge coefficient.",
        setup=FEATURES + 'model = Ridge(alpha=1.0).fit(train[features], train["y"])',
        solution="coefs = pd.Series(model.coef_, index=features)\nanswer = str(coefs.abs().idxmax())",
        hint="pd.Series(model.coef_, index=features).abs().idxmax(). (Unscaled coefficients: this is the largest number, not the most important feature.)",
    ),
    dict(
        prompt="Assign the lag-1 autocorrelation of the test residuals (`y − pred`) using pandas `autocorr(1)`.",
        setup=FEATURES + 'model = Ridge(alpha=1.0).fit(train[features], train["y"])\npred = model.predict(test[features])',
        solution='resid = test["y"] - pred\nanswer = float(resid.autocorr(1))',
        hint="resid is a Series; Series.autocorr(lag=1). A high value means structure the model misses.",
        tol=1e-4,
    ),
]
