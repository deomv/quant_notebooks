"""Exercises for 02_pandas/09_halfhourly_settlement_and_panel_data.ipynb"""
import numpy as np
import pandas as pd

COMMON_SETUP = """
import numpy as np
import pandas as pd
"""

LONG = 'long = pd.DataFrame({\n    "meter": ["A", "A", "A", "B", "B", "B"],\n    "period": [1, 2, 3, 1, 2, 3],\n    "kwh": [0.2, 0.3, 0.1, 1.0, 1.2, 0.9],\n})'

QUESTIONS = [
    dict(
        prompt="Build the UTC timestamp for each row of `toy`: local midnight of `settlement_date` in Europe/London, converted to UTC, plus (period − 1) × 30 minutes. Assign the resulting datetime Series (tz UTC).",
        setup='toy = pd.DataFrame({"settlement_date": ["2023-07-01"] * 4, "settlement_period": [1, 2, 3, 4], "kwh": [0.2, 0.3, 0.1, 0.4]})',
        solution='midnight_local = pd.to_datetime(toy["settlement_date"]).dt.tz_localize("Europe/London")\nmidnight_utc = midnight_local.dt.tz_convert("UTC")\nanswer = midnight_utc + pd.to_timedelta((toy["settlement_period"] - 1) * 30, unit="min")',
        hint="to_datetime → .dt.tz_localize('Europe/London') → .dt.tz_convert('UTC') → + pd.to_timedelta((period-1)*30, unit='min'). In July London is UTC+1, so period 1 is 23:00 UTC the day before.",
    ),
    dict(
        prompt="For each `settlement_date` in `rows`, assign the highest `settlement_period` present (a Series indexed by date). One of the dates is a clock-change day.",
        setup='rows = pd.DataFrame({"settlement_date": ["2023-03-25"] * 3 + ["2023-03-26"] * 3, "settlement_period": [46, 47, 48, 44, 45, 46]})',
        solution='answer = rows.groupby("settlement_date")["settlement_period"].max()',
        hint='groupby("settlement_date")["settlement_period"].max()',
    ),
    dict(
        prompt="Add the previous period's kwh **per meter** to `long` as a Series (NaN for each meter's first period). A bare `shift(1)` would leak meter A's last value into meter B.",
        setup=LONG,
        solution='answer = long.groupby("meter")["kwh"].shift(1)',
        hint='groupby("meter")["kwh"].shift(1)',
    ),
    dict(
        prompt="Reshape `long` to wide: one row per `period`, one column per `meter`, values `kwh`.",
        setup=LONG,
        solution='answer = long.pivot(index="period", columns="meter", values="kwh")',
        hint='pivot(index="period", columns="meter", values="kwh")',
    ),
    dict(
        prompt="Reshape `wide` back to long with columns `period`, `meter`, `kwh`, sorted by meter then period, with a fresh 0..n-1 index.",
        setup='wide = pd.DataFrame({"A": [0.2, 0.3, 0.1], "B": [1.0, 1.2, 0.9]}, index=pd.Index([1, 2, 3], name="period"))',
        solution='answer = (wide.reset_index()\n            .melt(id_vars="period", var_name="meter", value_name="kwh")\n            .sort_values(["meter", "period"])\n            .reset_index(drop=True))',
        hint="reset_index() → melt(id_vars='period', var_name='meter', value_name='kwh') → sort_values → reset_index(drop=True).",
    ),
    dict(
        prompt="`hh` has half-hourly kWh per meter with a `time` column. Aggregate to **hourly energy per meter** (a Series with a (meter, time) MultiIndex). Energy is summed, not averaged.",
        setup='hh = pd.DataFrame({\n    "meter": ["A"] * 4 + ["B"] * 4,\n    "time": list(pd.date_range("2023-01-01 00:00", periods=4, freq="30min")) * 2,\n    "kwh": [0.1, 0.2, 0.3, 0.4, 1.0, 1.0, 2.0, 2.0],\n})',
        solution='answer = hh.groupby(["meter", pd.Grouper(key="time", freq="h")])["kwh"].sum()',
        hint='groupby(["meter", pd.Grouper(key="time", freq="h")])["kwh"].sum()',
    ),
    dict(
        prompt="Coincidence factor of the portfolio: (peak of the summed load across meters) / (sum of each meter's individual peak). Assign a float.",
        setup='wide = pd.DataFrame({"A": [1.0, 3.0, 2.0], "B": [2.0, 1.0, 3.0]})',
        solution="answer = float(wide.sum(axis=1).max() / wide.max().sum())",
        hint="wide.sum(axis=1).max() over wide.max().sum()",
    ),
    dict(
        prompt="Find the missing (meter, period) combinations in `gappy` given that every meter should have periods 1..4. Assign a sorted list of (meter, period) tuples.",
        setup='gappy = pd.DataFrame({"meter": ["A", "A", "A", "B", "B"], "period": [1, 2, 4, 1, 3], "kwh": [1, 1, 1, 2, 2]})',
        solution='grid = pd.MultiIndex.from_product([["A", "B"], [1, 2, 3, 4]], names=["meter", "period"])\npresent = pd.MultiIndex.from_frame(gappy[["meter", "period"]])\nanswer = sorted(grid.difference(present).tolist())',
        hint="Build the full grid with pd.MultiIndex.from_product, the present keys with from_frame, then .difference().",
    ),
    dict(
        prompt="Add each row's share of its meter's total kwh as a Series aligned with `long` (kwh / meter total).",
        setup=LONG,
        solution='answer = long["kwh"] / long.groupby("meter")["kwh"].transform("sum")',
        hint='groupby("meter")["kwh"].transform("sum") broadcasts the total back to every row.',
    ),
    dict(
        prompt="Real data: load `../data/meter_halfhourly_2023.csv.gz`. How many rows are exact duplicates of another row? Assign the int.",
        setup='panel = pd.read_csv("../data/meter_halfhourly_2023.csv.gz")',
        solution="answer = int(panel.duplicated().sum())",
        hint="duplicated().sum()",
    ),
    dict(
        prompt="Real data: after dropping exact duplicates, which meter has the **most missing periods**? Expected periods per date = the maximum `settlement_period` seen on that date across all meters. Assign the meter id as a string.",
        setup='panel = pd.read_csv("../data/meter_halfhourly_2023.csv.gz").drop_duplicates()',
        solution='expected_total = panel.groupby("settlement_date")["settlement_period"].max().sum()\nrows_per_meter = panel.groupby("meter_id").size()\nanswer = str((expected_total - rows_per_meter).idxmax())',
        hint="Sum of per-date max periods = rows a complete meter would have; subtract each meter's row count; idxmax.",
    ),
    dict(
        prompt="Real data: peak power in kW of meter M100000 (a half-hourly kWh reading × 2). Assign a float.",
        setup='panel = pd.read_csv("../data/meter_halfhourly_2023.csv.gz").drop_duplicates()',
        solution='answer = float(panel.loc[panel["meter_id"] == "M100000", "kwh"].max() * 2)',
        hint="Filter the meter, take max kwh, multiply by 2 (30-minute periods).",
    ),
]
