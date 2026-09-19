"""Exercises for 02_pandas/05_merge_join_concat.ipynb"""
import numpy as np
import pandas as pd

COMMON_SETUP = """
import numpy as np
import pandas as pd
"""

QUESTIONS = [
    dict(
        prompt="Merge `customers` and `readings` on `meter_id`, keeping only meters present in **both** tables.",
        setup='customers = pd.DataFrame({"meter_id": ["M1", "M2", "M3"], "region": ["N", "S", "N"]})\nreadings = pd.DataFrame({"meter_id": ["M1", "M3", "M4"], "kwh": [10, 30, 40]})',
        solution='answer = customers.merge(readings, on="meter_id", how="inner")',
        hint='pd.merge / .merge with on="meter_id" and how="inner".',
    ),
    dict(
        prompt="Merge the same two tables so that **every customer** is kept, with NaN kwh where there is no reading.",
        setup='customers = pd.DataFrame({"meter_id": ["M1", "M2", "M3"], "region": ["N", "S", "N"]})\nreadings = pd.DataFrame({"meter_id": ["M1", "M3", "M4"], "kwh": [10, 30, 40]})',
        solution='answer = customers.merge(readings, on="meter_id", how="left")',
        hint='how="left" keeps all rows of the left table.',
    ),
    dict(
        prompt="Find the meter ids that appear in `readings` but **not** in `customers`. Give them as a sorted list of strings.",
        setup='customers = pd.DataFrame({"meter_id": ["M1", "M2", "M3"], "region": ["N", "S", "N"]})\nreadings = pd.DataFrame({"meter_id": ["M1", "M3", "M4"], "kwh": [10, 30, 40]})',
        solution='m = customers.merge(readings, on="meter_id", how="outer", indicator=True)\nanswer = sorted(m.loc[m["_merge"] == "right_only", "meter_id"])',
        hint='how="outer" with indicator=True adds a `_merge` column with values left_only / right_only / both.',
    ),
    dict(
        prompt="`tariffs` has meter M1 twice. Left-merge `readings` with `tariffs` on `meter_id` and assign the **number of rows** of the result (an int). Notice it is more than the 3 rows you started with.",
        setup='readings = pd.DataFrame({"meter_id": ["M1", "M3", "M4"], "kwh": [10, 30, 40]})\ntariffs = pd.DataFrame({"meter_id": ["M1", "M1", "M3"], "tariff": ["A", "B", "A"]})',
        solution='merged = readings.merge(tariffs, on="meter_id", how="left")\nanswer = len(merged)',
        hint="len() of the merged frame. A duplicated key on the right side duplicates the matching left rows.",
    ),
    dict(
        prompt="Does `readings.merge(tariffs, on='meter_id', validate='one_to_one')` raise an error? Run it inside try/except and assign `True` if it raises, `False` otherwise.",
        setup='readings = pd.DataFrame({"meter_id": ["M1", "M3", "M4"], "kwh": [10, 30, 40]})\ntariffs = pd.DataFrame({"meter_id": ["M1", "M1", "M3"], "tariff": ["A", "B", "A"]})',
        solution='try:\n    readings.merge(tariffs, on="meter_id", validate="one_to_one")\n    answer = False\nexcept Exception:\n    answer = True',
        hint="validate= makes pandas check key cardinality and raise pd.errors.MergeError when it does not hold.",
    ),
    dict(
        prompt="`sites` calls the key `site_id` while `readings` calls it `meter_id`. Inner-merge them so the result has the columns meter_id, kwh, site_id, name.",
        setup='readings = pd.DataFrame({"meter_id": ["M1", "M3", "M4"], "kwh": [10, 30, 40]})\nsites = pd.DataFrame({"site_id": ["M1", "M4"], "name": ["Depot", "Shop"]})',
        solution='answer = readings.merge(sites, left_on="meter_id", right_on="site_id")',
        hint="left_on= and right_on= name the key column on each side.",
    ),
    dict(
        prompt="`load` and `price` are both indexed by hour. Attach `price` to `load` **by index**, keeping every row of `load`.",
        setup='load = pd.DataFrame({"mw": [10, 20, 30]}, index=[0, 1, 2])\nprice = pd.DataFrame({"eur": [50, 70]}, index=[1, 2])',
        solution='answer = load.join(price, how="left")',
        hint=".join aligns on the index; how='left' keeps all rows of load.",
    ),
    dict(
        prompt="Merge `est` and `act` on **both** `meter_id` and `date` (inner).",
        setup='est = pd.DataFrame({"meter_id": ["M1", "M1", "M2"], "date": ["d1", "d2", "d1"], "kwh_est": [9, 11, 20]})\nact = pd.DataFrame({"meter_id": ["M1", "M2", "M2"], "date": ["d1", "d1", "d2"], "kwh_act": [10, 19, 21]})',
        solution='answer = est.merge(act, on=["meter_id", "date"])',
        hint="on= accepts a list of column names.",
    ),
    dict(
        prompt="Both tables have a column `kwh`. Inner-merge on `meter_id` so the result has columns `kwh_est` and `kwh_act` (in that order after meter_id).",
        setup='est = pd.DataFrame({"meter_id": ["M1", "M2"], "kwh": [9, 20]})\nact = pd.DataFrame({"meter_id": ["M1", "M2"], "kwh": [10, 19]})',
        solution='answer = est.merge(act, on="meter_id", suffixes=("_est", "_act"))',
        hint='suffixes=("_est", "_act") renames the overlapping columns.',
    ),
    dict(
        prompt="Stack `jan` on top of `feb` into one frame with a fresh 0..n-1 index.",
        setup='jan = pd.DataFrame({"day": [1, 2], "kwh": [10, 12]})\nfeb = pd.DataFrame({"day": [1, 2], "kwh": [9, 8]})',
        solution="answer = pd.concat([jan, feb], ignore_index=True)",
        hint="pd.concat([...], ignore_index=True).",
    ),
    dict(
        prompt="For every row of `actuals`, attach the **latest forecast published at or before** that time, but only if it was published within the last 2 hours (otherwise NaN).",
        setup='actuals = pd.DataFrame({"time": pd.to_datetime(["2023-01-01 01:00", "2023-01-01 02:00", "2023-01-01 05:00"]), "load": [10, 20, 50]})\nforecasts = pd.DataFrame({"time": pd.to_datetime(["2023-01-01 00:30", "2023-01-01 01:30"]), "fc": [11, 21]})',
        solution='answer = pd.merge_asof(actuals, forecasts, on="time", direction="backward", tolerance=pd.Timedelta("2h"))',
        hint='pd.merge_asof(left, right, on="time", direction="backward", tolerance=pd.Timedelta("2h")). Both sides must be sorted by time.',
    ),
    dict(
        prompt="`primary` has gaps. Fill them from `backup`, keeping the primary value wherever it exists.",
        setup='primary = pd.Series([1.0, np.nan, 3.0, np.nan], index=["a", "b", "c", "d"])\nbackup = pd.Series([9.0, 2.0, 9.0, 4.0], index=["a", "b", "c", "d"])',
        solution="answer = primary.combine_first(backup)",
        hint="combine_first takes values from the second Series only where the first is NaN.",
    ),
    dict(
        prompt="Real data: load `../data/meters.csv` and `../data/meter_readings_daily.csv`. How many reading rows belong to a meter that is **not** in the meters table? Assign the count as an int.",
        setup='meters = pd.read_csv("../data/meters.csv")\nreadings = pd.read_csv("../data/meter_readings_daily.csv")',
        solution='m = readings.merge(meters[["meter_id"]], on="meter_id", how="left", indicator=True)\nanswer = int((m["_merge"] == "left_only").sum())',
        hint='Left-merge readings with meters[["meter_id"]] using indicator=True, then count "left_only". (Or use ~isin.)',
    ),
]
