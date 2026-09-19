"""Exercises for 05_time_series_research/06_renewables_and_price_market_eda.ipynb"""
import numpy as np
import pandas as pd

COMMON_SETUP = """
import numpy as np
import pandas as pd
"""

QUESTIONS = [
    dict(
        prompt="Turbine power curve: output fraction is 0 below cut-in (3 m/s), ((v − 3) / (12 − 3))³ between cut-in and rated (12 m/s), 1 from rated up to cut-out (25 m/s), and 0 at or above cut-out. Compute it for `speeds` as a NumPy array.",
        setup="speeds = np.array([2.0, 5.0, 8.0, 12.0, 26.0])",
        solution="""cubic = ((speeds - 3.0) / 9.0) ** 3
answer = np.where(speeds < 3.0, 0.0, np.where(speeds < 12.0, cubic, 1.0))
answer = np.where(speeds >= 25.0, 0.0, answer)""",
        hint="Nested np.where: below cut-in → 0, below rated → cubic, else 1; then a final np.where for cut-out.",
    ),
    dict(
        prompt="`mw` is hourly output of a 100 MW wind farm. Compute its capacity factor (average output divided by capacity).",
        setup='mw = pd.Series([0, 20, 80, 100], index=["h1", "h2", "h3", "h4"])',
        solution="answer = float(mw.mean() / 100)",
        hint="mean output / 100.",
    ),
    dict(
        prompt="Build the duration curve of `mw`: the values sorted from largest to smallest, as a NumPy array.",
        setup="mw = pd.Series([30, 0, 100, 60, 0, 10])",
        solution="answer = np.sort(mw.values)[::-1]",
        hint="np.sort sorts ascending; reverse with [::-1].",
    ),
    dict(
        prompt="Add a `net_load` Series = demand − wind − solar for the three periods in `toy` (keep the index).",
        setup='toy = pd.DataFrame({"demand": [30000, 32000, 35000], "wind": [2000, 2000, 8000], "solar": [0, 5000, 0]}, index=["night", "noon", "evening"])',
        solution='answer = toy["demand"] - toy["wind"] - toy["solar"]',
        hint="Column arithmetic keeps the index.",
    ),
    dict(
        prompt="Largest absolute hour-to-hour ramp in `mw` (as a number).",
        setup="mw = pd.Series([100, 300, 900, 400, 350])",
        solution="answer = float(mw.diff().abs().max())",
        hint="diff(), abs(), max().",
    ),
    dict(
        prompt="Count the spikes in `px` under two definitions: `n_fixed` = prices above mean + 2 × std (pandas std), `n_quantile` = prices above the 75th percentile.",
        setup="px = pd.Series([80, 90, 85, 400, 95, 88, 92, 250])",
        solution="""n_fixed = int((px > px.mean() + 2 * px.std()).sum())
n_quantile = int((px > px.quantile(0.75)).sum())""",
        hint="Each definition is one boolean comparison; sum() counts the Trues.",
        answer_var=["n_fixed", "n_quantile"],
    ),
    dict(
        prompt="Average price per hour of day from `toy`, as a Series indexed by hour.",
        setup='toy = pd.DataFrame({"hour": [1, 2, 1, 2, 3, 3], "price": [10, 20, 30, 40, 50, 70]})',
        solution='answer = toy.groupby("hour")["price"].mean()',
        hint="groupby('hour')['price'].mean().",
    ),
    dict(
        prompt="On the toy `day`, the peak block is hours 7–18 inclusive. Compute `peak` = mean price in the peak block and `offpeak` = mean price outside it.",
        setup='day = pd.DataFrame({"hour": [2, 4, 6, 10, 14, 18], "price": [50, 45, 60, 100, 90, 140]})',
        solution="""in_peak = day["hour"].between(7, 18)
peak = float(day.loc[in_peak, "price"].mean())
offpeak = float(day.loc[~in_peak, "price"].mean())""",
        hint="between(7, 18) is inclusive; use it as a mask and its negation.",
        answer_var=["peak", "offpeak"],
    ),
    dict(
        prompt="A 4-hour battery earns the average of the 4 highest prices minus the average of the 4 lowest. Compute that spread for `px`.",
        setup="px = pd.Series([80, 90, 85, 400, 95, 88, 92, 250])",
        solution="""s = np.sort(px.values)
answer = float(s[-4:].mean() - s[:4].mean())""",
        hint="Sort, take the last 4 and the first 4.",
    ),
    dict(
        prompt="Compute `time_weighted` = the plain mean of `price`, and `load_weighted` = total cost (price × volume) divided by total volume.",
        setup='toy = pd.DataFrame({"price": [50, 100, 300], "volume_mwh": [10, 10, 1]})',
        solution="""time_weighted = float(toy["price"].mean())
load_weighted = float((toy["price"] * toy["volume_mwh"]).sum() / toy["volume_mwh"].sum())""",
        hint="The weighted version is sum(price × volume) / sum(volume).",
        answer_var=["time_weighted", "load_weighted"],
    ),
    dict(
        prompt="Degree-day features from `t`: `hdd` = heating degrees below 15 (0 when warmer), `cdd` = cooling degrees above 22 (0 when colder). Both as Series.",
        setup='t = pd.Series([-5, 5, 15, 20, 26], name="temp_c")',
        solution="""hdd = (15 - t).clip(lower=0)
cdd = (t - 22).clip(lower=0)""",
        hint="(15 - t).clip(lower=0) and (t - 22).clip(lower=0).",
        answer_var=["hdd", "cdd"],
    ),
    dict(
        prompt="Real data: apply `power_curve` to the `wind_ms` column of `df` for a 10,000 MW fleet and compute the fleet's overall capacity factor (a number between 0 and 1).",
        setup='''df = pd.read_csv("../data/hourly_power_clean.csv", parse_dates=["time"])

def power_curve(v, cut_in=3.0, rated=12.0, cut_out=25.0):
    v = np.asarray(v, dtype=float)
    cubic = ((v - cut_in) / (rated - cut_in)) ** 3
    out = np.where(v < cut_in, 0.0, np.where(v < rated, cubic, 1.0))
    return np.where(v >= cut_out, 0.0, out)''',
        solution="""wind_mw = 10_000 * power_curve(df["wind_ms"])
answer = float(wind_mw.mean() / 10_000)""",
        hint="Capacity factor = mean MW / capacity; the 10,000 cancels, so it is just the mean output fraction.",
        tol=1e-4,
    ),
    dict(
        prompt="Real data: how many hours in 2023 had a negative day-ahead price?",
        setup='df = pd.read_csv("../data/hourly_power_clean.csv", parse_dates=["time"])',
        solution="""y23 = df[df["time"].dt.year == 2023]
answer = int((y23["price_eur_mwh"] < 0).sum())""",
        hint="Filter on time.dt.year, then count price < 0.",
    ),
]
