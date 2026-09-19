"""Exercises for 05_time_series_research/05_anomaly_detection_and_data_quality.ipynb"""
import numpy as np
import pandas as pd

COMMON_SETUP = """
import numpy as np
import pandas as pd
"""

QUESTIONS = [
    dict(
        prompt="`kwh` contains a stuck stretch. Find the **length** of the longest run of identical consecutive values and the **position** (0-based) where that run starts.",
        setup="kwh = pd.Series([0.5, 0.4, 0.3, 0.3, 0.3, 0.3, 0.6, 0.5])",
        solution="""changed = kwh != kwh.shift()
run_id = changed.cumsum()
sizes = run_id.value_counts()
longest = sizes.idxmax()
run_len = int(sizes.max())
run_start = int(run_id[run_id == longest].index[0])""",
        hint="kwh != kwh.shift() is True where a new run starts; cumsum() of that gives a run id per row; value_counts() gives run lengths.",
        answer_var=["run_len", "run_start"],
    ),
    dict(
        prompt="For each day, divide the daily total by the **median of the previous 3 days** (the current day excluded). The first three days have no full window and stay NaN.",
        setup='daily = pd.Series([10.0, 11.0, 9.0, 10.0, 10500.0, 9800.0], index=["d1", "d2", "d3", "d4", "d5", "d6"])',
        solution="answer = daily / daily.shift(1).rolling(3).median()",
        hint="shift(1) first so the window ends on the previous day, then rolling(3).median().",
    ),
    dict(
        prompt="Compute the robust z-score of `v`: (v − median) / (1.4826 × MAD), where MAD is the median of |v − median|.",
        setup="v = pd.Series([1.0, 1.1, 0.9, 1.0, 50.0, 1.0, 1.1])",
        solution="""med = v.median()
mad = (v - med).abs().median()
answer = (v - med) / (1.4826 * mad)""",
        hint="Three lines: the median, the MAD as the median of absolute deviations, then the ratio.",
    ),
    dict(
        prompt="Using the same `v`: does the spike exceed 3 under the **plain** z-score ((v − mean) / std) and under the **robust** z-score? Assign two booleans.",
        setup="v = pd.Series([1.0, 1.1, 0.9, 1.0, 50.0, 1.0, 1.1])",
        solution="""z_plain = (v - v.mean()) / v.std()
med = v.median()
mad = (v - med).abs().median()
z_robust = (v - med) / (1.4826 * mad)
spike_plain = bool((z_plain.abs() > 3).any())
spike_robust = bool((z_robust.abs() > 3).any())""",
        hint="The spike drags the mean and std with it, so the plain z of the spike itself stays small. Compare (z.abs() > 3).any() for both.",
        answer_var=["spike_plain", "spike_robust"],
    ),
    dict(
        prompt="Build the CUSUM of `x`: cumulative sum of (log(x) − mean of log(x)). Assign the **position** (index label) where |CUSUM| is largest, i.e. the last point before the level shift.",
        setup="x = pd.Series([10.0, 10.0, 10.0, 10.0, 14.0, 14.0, 14.0, 14.0])",
        solution="""logs = np.log(x)
cusum = (logs - logs.mean()).cumsum()
answer = int(cusum.abs().idxmax())""",
        hint="np.log, subtract the mean, cumsum(), then .abs().idxmax().",
    ),
    dict(
        prompt="For each settlement date in `dates`, compute the number of half-hour periods in that local (Europe/London) day: 48 normally, 46 on the spring clock change, 50 on the autumn one. Return a Series indexed by the date strings.",
        setup='dates = ["2023-03-25", "2023-03-26", "2023-10-29", "2023-12-01"]',
        solution="""start = pd.to_datetime(dates).tz_localize("Europe/London")
end = (pd.to_datetime(dates) + pd.Timedelta(days=1)).tz_localize("Europe/London")
answer = pd.Series(((end - start) / pd.Timedelta(minutes=30)).astype(int), index=dates)""",
        hint="Localise each midnight and the next midnight to Europe/London; the difference in minutes divided by 30 is the number of periods.",
    ),
    dict(
        prompt="`toy` has the readings of one meter for days d1..d3 (4 periods expected per day). Return a Series indexed by ['d1', 'd2', 'd3'] with the number of **missing** periods per day. A day with no rows at all must show 4, not be absent.",
        setup='toy = pd.DataFrame({"date": ["d1", "d1", "d1", "d1", "d2", "d2", "d2"], "period": [1, 2, 3, 4, 1, 2, 4], "kwh": [0.1, 0.2, 0.1, 0.3, 0.2, 0.2, 0.1]})',
        solution="""counts = toy.groupby("date").size().reindex(["d1", "d2", "d3"], fill_value=0)
answer = 4 - counts""",
        hint="groupby('date').size() only knows the dates that exist; reindex to the full list with fill_value=0.",
    ),
    dict(
        prompt="Count the rows of `toy` that break a physical rule: negative kWh on a meter **without** solar, or kWh above 11.5 (the half-hourly maximum of a 100 A supply).",
        setup='toy = pd.DataFrame({"kwh": [0.4, -0.3, -0.2, 15.0, -0.1, 3.0], "has_solar": [False, True, True, False, False, False]})',
        solution="""bad = ((toy["kwh"] < 0) & ~toy["has_solar"]) | (toy["kwh"] > 11.5)
answer = int(bad.sum())""",
        hint="Two boolean conditions combined with |; the negative-without-solar one needs & ~toy['has_solar'].",
    ),
    dict(
        prompt="Compute the two-sample Kolmogorov–Smirnov statistic between `history` and `new_month` (use scipy.stats).",
        setup="history = pd.Series([1.0, 1.2, 0.9, 1.1, 1.0, 1.3])\nnew_month = pd.Series([1.0, 1200.0, 1100.0, 900.0, 1300.0, 1000.0])",
        solution="""from scipy import stats
answer = float(stats.ks_2samp(history, new_month).statistic)""",
        hint="scipy.stats.ks_2samp(a, b) returns an object with .statistic and .pvalue.",
    ),
    dict(
        prompt="Fit `IsolationForest(n_estimators=100, random_state=0)` on `points` and assign the **row position** of the most anomalous point (the lowest `score_samples`).",
        setup='points = pd.DataFrame({"total": [10, 11, 9, 10, 12, 10, 11, 60], "peak": [1, 1.2, 0.9, 1.1, 1.3, 1.0, 1.1, 9]})',
        solution="""from sklearn.ensemble import IsolationForest
iso = IsolationForest(n_estimators=100, random_state=0).fit(points)
answer = int(np.argmin(iso.score_samples(points)))""",
        hint="score_samples gives higher values for normal points; np.argmin finds the lowest.",
    ),
    dict(
        prompt="Combine the two detection tables into one issues table with columns ['meter_id', 'date', 'issue_type'], where issue_type is 'stuck_value' for rows from `stuck` and 'level_shift' for rows from `shifts`. Sort by date then meter_id and reset the index.",
        setup='''stuck = pd.DataFrame({"meter_id": ["m1", "m4"], "date": ["2023-06-05", "2023-02-01"], "run_length": [336, 40]})
shifts = pd.DataFrame({"meter_id": ["m2"], "date": ["2023-09-01"], "ratio": [1020.5]})''',
        solution="""cols = ["meter_id", "date", "issue_type"]
a = stuck.assign(issue_type="stuck_value")[cols]
b = shifts.assign(issue_type="level_shift")[cols]
answer = pd.concat([a, b]).sort_values(["date", "meter_id"]).reset_index(drop=True)""",
        hint="assign(issue_type=...) on each frame, select the three columns, pd.concat, sort_values, reset_index(drop=True).",
        check_fn=lambda got, exp: (
            isinstance(got, pd.DataFrame) and list(got.columns) == list(exp.columns)
            and got.reset_index(drop=True).equals(exp),
            "check the column names, the order of rows and reset_index(drop=True)",
        ),
    ),
    dict(
        prompt="Real data: `m7` holds meter M100007 from the half-hourly panel with a `month` column. Compute the ratio of the mean kWh in September to the mean kWh in August.",
        setup='''panel = pd.read_csv("../data/meter_halfhourly_2023.csv.gz")
m7 = panel[panel["meter_id"] == "M100007"].copy()
m7["month"] = m7["settlement_date"].str[:7]''',
        solution="""sep = m7.loc[m7["month"] == "2023-09", "kwh"].mean()
aug = m7.loc[m7["month"] == "2023-08", "kwh"].mean()
answer = float(sep / aug)""",
        hint="Boolean masks on the month column, .mean() on kwh, divide. A ratio near 1000 means a unit change (Wh instead of kWh).",
        tol=1e-3,
    ),
    dict(
        prompt="Real data: `m3` holds meter M100003, deduplicated and sorted by date and period. What is the length of its longest run of identical consecutive kWh readings?",
        setup='''panel = pd.read_csv("../data/meter_halfhourly_2023.csv.gz")
m3 = panel[panel["meter_id"] == "M100003"].drop_duplicates().sort_values(["settlement_date", "settlement_period"]).reset_index(drop=True)''',
        solution="""changed = m3["kwh"] != m3["kwh"].shift()
run_id = changed.cumsum()
answer = int(run_id.value_counts().max())""",
        hint="Same recipe as task 1, on the real column. A week of half-hours is 7 × 48.",
    ),
]
