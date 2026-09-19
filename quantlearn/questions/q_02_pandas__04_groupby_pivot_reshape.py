"""Exercises for 02_pandas/04_groupby_pivot_reshape.ipynb"""
import numpy as np
import pandas as pd

COMMON_SETUP = """
import numpy as np
import pandas as pd

df = pd.DataFrame({
    "region": ["N", "N", "S", "S", "S", "N"],
    "tariff": ["A", "B", "A", "A", "B", "A"],
    "kwh":    [10.0, 20.0, 30.0, 40.0, 50.0, np.nan],
})
"""

QUESTIONS = [
    dict(
        prompt="Total `kwh` per `region` as a Series indexed by region.",
        setup="",
        solution='answer = df.groupby("region")["kwh"].sum()',
        hint="groupby(key)[column].sum()",
    ),
    dict(
        prompt="Per region, assign the number of **rows** to `n_rows` and the number of **non-missing kwh values** to `n_valid` (both Series indexed by region).",
        setup="",
        solution='n_rows = df.groupby("region").size()\nn_valid = df.groupby("region")["kwh"].count()',
        hint="size() counts rows (NaN included); count() counts non-null values of a column.",
        answer_var=["n_rows", "n_valid"],
    ),
    dict(
        prompt="Group by `region` and `tariff` and produce a DataFrame with two columns: `total` (sum of kwh) and `avg` (mean of kwh), using named aggregation.",
        setup="",
        solution='answer = df.groupby(["region", "tariff"]).agg(total=("kwh", "sum"), avg=("kwh", "mean"))',
        hint="agg(new_name=(column, func), ...)",
    ),
    dict(
        prompt="Assign a Series, aligned with `df`, giving each row's `kwh` as a share of its region's total.",
        setup="",
        solution='answer = df["kwh"] / df.groupby("region")["kwh"].transform("sum")',
        hint="transform('sum') broadcasts the group total back to every row; then divide.",
    ),
    dict(
        prompt="Keep only the rows belonging to (region, tariff) groups that have **at least 2 rows**. Assign the DataFrame.",
        setup="",
        solution='answer = df.groupby(["region", "tariff"]).filter(lambda g: len(g) >= 2)',
        hint="groupby(...).filter(lambda g: len(g) >= 2)",
    ),
    dict(
        prompt="`ts` has a `time` column. Assign the mean of `value` by **hour of day** as a Series indexed by hour.",
        setup='ts = pd.DataFrame({"time": pd.to_datetime(["2023-01-01 06:00", "2023-01-01 18:00", "2023-01-02 06:00", "2023-01-02 18:00"]), "value": [10, 30, 20, 50]})',
        solution='answer = ts.groupby(ts["time"].dt.hour)["value"].mean()',
        hint="groupby(ts['time'].dt.hour)",
    ),
    dict(
        prompt="Build a pivot table of the **sum** of `kwh` with `region` as rows and `tariff` as columns, including the `All` margins.",
        setup="",
        solution='answer = pd.pivot_table(df, index="region", columns="tariff", values="kwh", aggfunc="sum", margins=True)',
        hint="pd.pivot_table(df, index=, columns=, values=, aggfunc='sum', margins=True). The default aggfunc is mean.",
    ),
    dict(
        prompt="`wide` has one column per tariff. Melt it into long format with columns `region`, `tariff`, `kwh`.",
        setup='wide = pd.DataFrame({"region": ["N", "S"], "A": [10.0, 70.0], "B": [20.0, 50.0]})',
        solution='answer = wide.melt(id_vars="region", var_name="tariff", value_name="kwh")',
        hint="melt(id_vars=..., var_name=..., value_name=...)",
    ),
    dict(
        prompt="`g` is a Series with a two-level index (region, tariff). Unstack the `tariff` level into columns. Assign the DataFrame.",
        setup='g = df.groupby(["region", "tariff"])["kwh"].sum()',
        solution='answer = g.unstack("tariff")',
        hint="unstack(level)",
    ),
    dict(
        prompt="Assign a crosstab counting rows of `df` by `region` (rows) and `tariff` (columns).",
        setup="",
        solution='answer = pd.crosstab(df["region"], df["tariff"])',
        hint="pd.crosstab(index_series, column_series)",
    ),
    dict(
        prompt="Assign the running total of `kwh` **within each region**, in the original row order (a Series aligned with `df`).",
        setup="",
        solution='answer = df.groupby("region")["kwh"].cumsum()',
        hint="groupby(key)[col].cumsum() keeps the original index.",
    ),
    dict(
        prompt="Assign the **first** row of each region as a DataFrame indexed by region (columns tariff, kwh).",
        setup="",
        solution='answer = df.groupby("region").first()',
        hint="groupby(key).first()",
        note="Note: first() skips NaN within a column, so it is not always the literal first row.",
    ),
    dict(
        prompt="`p` is a long frame of readings per meter, already sorted by meter then day. Assign the rolling mean over 2 days **within each meter** as a Series aligned with `p` (same index as `p`).",
        setup='p = pd.DataFrame({"meter": ["A", "A", "A", "B", "B", "B"], "day": [1, 2, 3, 1, 2, 3], "kwh": [1.0, 2.0, 3.0, 10.0, 20.0, 30.0]})',
        solution='answer = p.groupby("meter")["kwh"].rolling(2).mean().reset_index(level=0, drop=True)',
        hint="groupby(...).rolling(2).mean() returns a MultiIndex (meter, original index); drop the meter level with reset_index(level=0, drop=True).",
    ),
    dict(
        prompt="Real data: from `../data/meter_readings_daily.csv`, assign the 3 meters with the highest **mean** daily kWh as a Series (meter_id → mean kwh), largest first.",
        setup='readings = pd.read_csv("../data/meter_readings_daily.csv")',
        solution='answer = readings.groupby("meter_id")["kwh"].mean().nlargest(3)',
        hint="groupby('meter_id')['kwh'].mean().nlargest(3)",
    ),
]
