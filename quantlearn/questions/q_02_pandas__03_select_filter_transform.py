"""Exercises for 02_pandas/03_select_filter_transform.ipynb"""
import numpy as np
import pandas as pd

COMMON_SETUP = """
import numpy as np
import pandas as pd

df = pd.DataFrame({
    "region": ["N", "S", "N", "S", "N", "S"],
    "temp":   [2.0, 8.0, 15.0, 20.0, 25.0, 5.0],
    "cons":   [50, 40, 30, 25, 35, 45],
}, index=["r0", "r1", "r2", "r3", "r4", "r5"])
"""

QUESTIONS = [
    dict(
        prompt="Select rows `r1` and `r3` and only the columns `region` and `cons`, using labels.",
        setup="",
        solution='answer = df.loc[["r1", "r3"], ["region", "cons"]]',
        hint="df.loc[row_labels, column_labels]",
    ),
    dict(
        prompt="Select the first three rows and the last column **by position**.",
        setup="",
        solution="answer = df.iloc[:3, -1]",
        hint="df.iloc[rows, cols]; the result is a Series because you picked one column.",
    ),
    dict(
        prompt="Keep the rows where `region` is 'N' **and** `temp` is above 10.",
        setup="",
        solution='answer = df[(df["region"] == "N") & (df["temp"] > 10)]',
        hint="Parentheses around each comparison, combined with &.",
    ),
    dict(
        prompt="Keep the rows whose `temp` is between 5 and 20 inclusive, using `between`.",
        setup="",
        solution='answer = df[df["temp"].between(5, 20)]',
        hint="Series.between(low, high) is inclusive by default.",
    ),
    dict(
        prompt="Do the same selection as the previous task with `df.query`, using a query string.",
        setup="",
        solution='answer = df.query("5 <= temp <= 20")',
        hint="df.query('5 <= temp <= 20') — chained comparisons work inside query strings.",
    ),
    dict(
        prompt="Rename the column `cons` to `consumption_mwh` and `temp` to `temp_c`. Assign the renamed DataFrame.",
        setup="",
        solution='answer = df.rename(columns={"cons": "consumption_mwh", "temp": "temp_c"})',
        hint="rename(columns={old: new, ...})",
    ),
    dict(
        prompt="Add a column `hdd` (heating degrees) equal to max(15 − temp, 0) for each row, without modifying `df`. Assign the new DataFrame.",
        setup="",
        solution='answer = df.assign(hdd=(15 - df["temp"]).clip(lower=0))',
        hint="df.assign(hdd=...) returns a new frame; (15 - df['temp']).clip(lower=0) is the vectorised max(…, 0).",
    ),
    dict(
        prompt="Create a Series `label` that is 'cold' when temp < 10, 'hot' when temp > 20 and 'mild' otherwise, indexed like `df`.",
        setup="",
        solution='answer = pd.Series(np.select([df["temp"] < 10, df["temp"] > 20], ["cold", "hot"], default="mild"), index=df.index)',
        hint="np.select([cond1, cond2], [value1, value2], default=...) then wrap in pd.Series with df.index.",
    ),
    dict(
        prompt="Bin `temp` into three fixed bins with edges 0, 10, 20, 30 labelled 'low', 'mid', 'high'. Assign the Series (as strings, i.e. call `.astype(str)` at the end).",
        setup="",
        solution='answer = pd.cut(df["temp"], bins=[0, 10, 20, 30], labels=["low", "mid", "high"]).astype(str)',
        hint="pd.cut(series, bins=[...], labels=[...])",
    ),
    dict(
        prompt="Assign the percentage change of `cons` from one row to the next (first row NaN).",
        setup="",
        solution='answer = df["cons"].pct_change()',
        hint="pct_change() = value / previous value − 1.",
    ),
    dict(
        prompt="Sort `df` by `region` ascending and, within region, by `cons` **descending**.",
        setup="",
        solution='answer = df.sort_values(["region", "cons"], ascending=[True, False])',
        hint="sort_values([...], ascending=[True, False])",
    ),
    dict(
        prompt="Assign the two rows with the largest `cons` (largest first) using `nlargest`.",
        setup="",
        solution='answer = df.nlargest(2, "cons")',
        hint="df.nlargest(n, column)",
    ),
    dict(
        prompt="Assign the **row label** where `cons` is highest.",
        setup="",
        solution='answer = df["cons"].idxmax()',
        hint="idxmax() gives the label; argmax() gives the position.",
    ),
    dict(
        prompt="Make a copy of `df` and set `cons` to 0 for every row where `region` is 'S', using the `.loc[mask, col] = value` pattern. Assign the modified copy.",
        setup="",
        solution='answer = df.copy()\nanswer.loc[answer["region"] == "S", "cons"] = 0',
        hint="answer = df.copy(); answer.loc[answer['region'] == 'S', 'cons'] = 0. Chained df[mask]['cons'] = 0 would not modify anything.",
    ),
    dict(
        prompt="Real data: load `../data/hourly_power_clean.csv` (parse `time`) and assign a DataFrame with the 3 hours of highest `price_eur_mwh`, keeping only the columns `time` and `price_eur_mwh`, highest first.",
        setup='power = pd.read_csv("../data/hourly_power_clean.csv", parse_dates=["time"])',
        solution='answer = power.nlargest(3, "price_eur_mwh")[["time", "price_eur_mwh"]]',
        hint="nlargest then select the two columns.",
    ),
]
