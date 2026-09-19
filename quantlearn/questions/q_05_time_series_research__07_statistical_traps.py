"""Exercises for 05_time_series_research/07_statistical_traps.ipynb"""
import numpy as np
import pandas as pd

COMMON_SETUP = """
import numpy as np
import pandas as pd
"""

QUESTIONS = [
    dict(
        prompt="Simpson's paradox. Compute `overall` = cost per kWh by region (total cost / total kWh, a Series indexed by region) and `reverses` = True if region A is cheaper than B **within both customer types** yet dearer than B **overall**.",
        setup='''toy = pd.DataFrame({
    "region":        ["A", "A", "B", "B"],
    "customer_type": ["residential", "sme", "residential", "sme"],
    "kwh":           [9000, 1000, 1000, 9000],
    "cost":          [2700, 200, 320, 1890],
})''',
        solution="""sums = toy.groupby("region")[["cost", "kwh"]].sum()
overall = sums["cost"] / sums["kwh"]
within = (toy["cost"] / toy["kwh"]).groupby([toy["region"], toy["customer_type"]]).first().unstack()
reverses = bool((within.loc["A"] < within.loc["B"]).all() and overall["A"] > overall["B"])""",
        hint="Overall: sum cost and kWh per region first, then divide. Within: cost/kwh per row, arranged region × type; compare A's row with B's row.",
        answer_var=["overall", "reverses"],
    ),
    dict(
        prompt="Regression to the mean. `top2_change` = mean change (day2 − day1) for the 2 meters with the highest day1; `all_change` = mean change for all meters.",
        setup='toy = pd.DataFrame({"meter": list("abcdef"), "day1": [10, 50, 20, 45, 15, 30], "day2": [12, 30, 22, 28, 14, 29]})',
        solution="""top2 = toy.nlargest(2, "day1")
top2_change = float((top2["day2"] - top2["day1"]).mean())
all_change = float((toy["day2"] - toy["day1"]).mean())""",
        hint="nlargest(2, 'day1') picks the top two; the change is day2 − day1.",
        answer_var=["top2_change", "all_change"],
    ),
    dict(
        prompt="Survivorship. `growth_all` = mean growth over all five customers; `growth_survivors` = mean growth over customers who did **not** churn.",
        setup='toy = pd.DataFrame({"growth": [0.02, 0.01, -0.05, 0.03, -0.04], "churned": [False, False, True, False, True]})',
        solution="""growth_all = float(toy["growth"].mean())
growth_survivors = float(toy.loc[~toy["churned"], "growth"].mean())""",
        hint="Mask with ~toy['churned'] for the survivors.",
        answer_var=["growth_all", "growth_survivors"],
    ),
    dict(
        prompt="Base rates. A spike detector has sensitivity 0.95 and specificity 0.95; spikes occur in 2% of hours. Compute the precision: P(spike | alarm).",
        setup="sensitivity = 0.95\nspecificity = 0.95\nprevalence = 0.02",
        solution="""true_alarms = sensitivity * prevalence
false_alarms = (1 - specificity) * (1 - prevalence)
answer = true_alarms / (true_alarms + false_alarms)""",
        hint="true alarms = sensitivity × prevalence; false alarms = (1 − specificity) × (1 − prevalence); precision = true / (true + false).",
    ),
    dict(
        prompt="Mean of ratios vs ratio of sums. `mean_of_prices` = plain mean of price; `volume_weighted` = total cost / total volume.",
        setup='toy = pd.DataFrame({"price": [50, 100, 300], "volume_mwh": [10, 10, 1]})',
        solution="""mean_of_prices = float(toy["price"].mean())
volume_weighted = float((toy["price"] * toy["volume_mwh"]).sum() / toy["volume_mwh"].sum())""",
        hint="Weighted: sum(price × volume) / sum(volume).",
        answer_var=["mean_of_prices", "volume_weighted"],
    ),
    dict(
        prompt="Averaging percentages. `unweighted` = mean of the three regional solar shares; `weighted` = total meters with solar / total meters.",
        setup='g = pd.DataFrame({"region": ["A", "B", "C"], "meters": [900, 50, 50], "solar_share": [0.05, 0.40, 0.45]})',
        solution="""unweighted = float(g["solar_share"].mean())
weighted = float((g["meters"] * g["solar_share"]).sum() / g["meters"].sum())""",
        hint="meters × share gives meters with solar per region.",
        answer_var=["unweighted", "weighted"],
    ),
    dict(
        prompt="Expected maximum of noise. With `rng = np.random.default_rng(0)`, draw `rng.normal(size=(10000, 5))` (10,000 trials of 5 strategies with zero true skill) and compute the average of the best-of-5 value across trials.",
        setup="rng = np.random.default_rng(0)",
        solution="""draws = rng.normal(size=(10000, 5))
answer = float(draws.max(axis=1).mean())""",
        hint="max(axis=1) is the best of the 5 in each trial; then mean over trials. The answer is about 1.16, not 0.",
        check_fn=lambda got, exp: (abs(float(got) - exp) < 0.03, f"expected about {exp:.3f}, got {float(got):.3f}"),
    ),
    dict(
        prompt="Best-of-N selection. Pick the strategy with the best P&L in the first 10 days (`best`, its name) and report that same strategy's P&L in the next 10 days (`next_pnl`).",
        setup='toy = pd.DataFrame({"strategy": ["s1", "s2", "s3", "s4", "s5"], "first10": [2, -4, 8, 0, -2], "next10": [1, 3, -6, 2, 0]}).set_index("strategy")',
        solution="""best = toy["first10"].idxmax()
next_pnl = int(toy.loc[best, "next10"])""",
        hint="idxmax() on first10 gives the label; .loc[label, 'next10'] the out-of-sample number.",
        answer_var=["best", "next_pnl"],
    ),
    dict(
        prompt="Multiple comparisons. From the 20 p-values in `pvals`, count how many are below 0.05 (`n_raw`) and how many are below the Bonferroni threshold 0.05 / 20 (`n_bonf`).",
        setup="pvals = np.array([0.30, 0.01, 0.55, 0.03, 0.72, 0.001, 0.41, 0.09, 0.66, 0.21, 0.38, 0.85, 0.12, 0.47, 0.93, 0.06, 0.29, 0.51, 0.77, 0.19])",
        solution="""n_raw = int((pvals < 0.05).sum())
n_bonf = int((pvals < 0.05 / 20).sum())""",
        hint="Two boolean comparisons and sum().",
        answer_var=["n_raw", "n_bonf"],
    ),
    dict(
        prompt="The improvement that is one period. `wins_b` = number of months in which model B has lower RMSE than A; `annual_b_better` = True if B's annual RMSE (sqrt of the mean of squared monthly RMSEs) is lower than A's.",
        setup="rmse_a = pd.Series([100.0] * 12)\nrmse_b = pd.Series([103.0] * 11 + [55.0])",
        solution="""wins_b = int((rmse_b < rmse_a).sum())
annual_a = np.sqrt((rmse_a ** 2).mean())
annual_b = np.sqrt((rmse_b ** 2).mean())
annual_b_better = bool(annual_b < annual_a)""",
        hint="Monthly wins are a boolean sum; the annual figure pools squared errors, so one very good month can dominate.",
        answer_var=["wins_b", "annual_b_better"],
    ),
    dict(
        prompt="Berkson's paradox. Customers complain if they have a high bill **or** poor service. Compute `corr_all` = correlation between high_bill and poor_service over everyone, and `corr_complainers` = the same correlation among complainers only.",
        setup='toy = pd.DataFrame({"high_bill": [1, 1, 1, 1, 0, 0, 0, 0], "poor_service": [1, 1, 0, 0, 1, 1, 0, 0]})',
        solution="""corr_all = float(toy["high_bill"].corr(toy["poor_service"]))
complained = (toy["high_bill"] == 1) | (toy["poor_service"] == 1)
sub = toy[complained]
corr_complainers = float(sub["high_bill"].corr(sub["poor_service"]))""",
        hint="Build the complained mask with |, subset, then .corr() on the two columns.",
        answer_var=["corr_all", "corr_complainers"],
    ),
    dict(
        prompt="Real data, 2023 only: `time_weighted` = plain mean of the hourly price; `load_weighted` = sum(price × consumption) / sum(consumption).",
        setup='''df = pd.read_csv("../data/hourly_power_clean.csv", parse_dates=["time"])
h = df[df["time"].dt.year == 2023]''',
        solution="""time_weighted = float(h["price_eur_mwh"].mean())
load_weighted = float((h["price_eur_mwh"] * h["consumption_mwh"]).sum() / h["consumption_mwh"].sum())""",
        hint="Same formula as the toy, on real columns.",
        answer_var=["time_weighted", "load_weighted"],
        tol=1e-4,
    ),
]
