"""Exercises for 07_optimisation/05_imbalance_newsvendor_and_quantile_decisions.ipynb"""
import numpy as np
import pandas as pd

COMMON_SETUP = """
import numpy as np
import pandas as pd
from sklearn.linear_model import QuantileRegressor, LinearRegression
"""

_NV = """
scenarios = np.array([8.0, 9.0, 10.0, 11.0, 12.0])   # equally likely loads (MWh)
p_da = 100.0      # day-ahead price
p_buy = 130.0     # price paid for a shortfall
p_sell = 90.0     # price received for a surplus
c_u = p_buy - p_da    # 30: extra cost per MWh short
c_o = p_da - p_sell   # 10: extra cost per MWh long
"""

QUESTIONS = [
    dict(
        prompt="A retailer buys `q` MWh day-ahead at `p_da`; the shortfall is bought at `p_buy`, the surplus sold at `p_sell`. Compute the total cost when q=9 and the load turns out to be 11 (`cost_short`), and when q=12 and the load is 11 (`cost_long`).",
        setup=_NV,
        solution="cost_short = 9 * p_da + (11 - 9) * p_buy\ncost_long = 12 * p_da - (12 - 11) * p_sell",
        hint="Short: q·p_da + (L − q)·p_buy. Long: q·p_da − (q − L)·p_sell.",
        answer_var=["cost_short", "cost_long"],
    ),
    dict(
        prompt="The *excess* cost of a decision relative to perfect foresight is c_u·max(L − q, 0) + c_o·max(q − L, 0). Build a DataFrame `table` with one row per scenario (index = the scenario load) and one column per candidate q in [8, 9, 10, 11, 12] (column labels are the integers), holding that excess cost.",
        setup=_NV + "candidates = [8, 9, 10, 11, 12]",
        solution="table = pd.DataFrame(index=scenarios, columns=candidates, dtype=float)\nfor q in candidates:\n    table[q] = c_u * np.maximum(scenarios - q, 0) + c_o * np.maximum(q - scenarios, 0)",
        hint="Loop over the candidates; each column is a vector over the five scenarios. np.maximum(., 0) is the positive part.",
        answer_var="table",
    ),
    dict(
        prompt="From `table` (given), compute the expected excess cost per candidate q as a Series indexed by q, and the best q as an integer `q_best`.",
        setup=_NV + "candidates = [8, 9, 10, 11, 12]\ntable = pd.DataFrame(index=scenarios, columns=candidates, dtype=float)\nfor q in candidates:\n    table[q] = c_u * np.maximum(scenarios - q, 0) + c_o * np.maximum(q - scenarios, 0)",
        solution="expected = table.mean()\nq_best = int(expected.idxmin())",
        hint="Scenarios are equally likely, so the expectation is the column mean. idxmin gives the column label.",
        answer_var=["expected", "q_best"],
    ),
    dict(
        prompt="Compute the critical fractile `tau_star = c_u / (c_u + c_o)` and the corresponding quantile of the scenarios, `q_star = np.quantile(scenarios, tau_star)`. It must agree with the brute-force best q.",
        setup=_NV,
        solution="tau_star = c_u / (c_u + c_o)\nq_star = np.quantile(scenarios, tau_star)",
        hint="Being short is the expensive side here (30 vs 10), so the fractile is above 0.5.",
        answer_var=["tau_star", "q_star"],
    ),
    dict(
        prompt="Someone inverts the fractile to c_o / (c_u + c_o). Compute the quantity that gives (`q_wrong`) and its expected excess cost over the five scenarios (`cost_wrong`).",
        setup=_NV,
        solution="tau_wrong = c_o / (c_u + c_o)\nq_wrong = np.quantile(scenarios, tau_wrong)\ncost_wrong = np.mean(c_u * np.maximum(scenarios - q_wrong, 0) + c_o * np.maximum(q_wrong - scenarios, 0))",
        hint="Quantile at 0.25 of [8..12] is 9. Then average the excess-cost formula over the scenarios.",
        answer_var=["q_wrong", "cost_wrong"],
    ),
    dict(
        prompt="Pinball loss. For tau=0.9, `actual` and `forecast` given, compute the mean pinball loss: with e = actual − forecast, loss = max(tau·e, (tau − 1)·e).",
        setup="tau = 0.9\nactual = np.array([10.0, 10.0, 10.0, 10.0])\nforecast = np.array([8.0, 9.0, 11.0, 12.0])",
        solution="e = actual - forecast\nanswer = np.mean(np.maximum(tau * e, (tau - 1) * e))",
        hint="Under-forecasts (e > 0) cost tau·e, over-forecasts cost (1 − tau)·|e|.",
    ),
    dict(
        prompt="Fit a median regression (`QuantileRegressor(quantile=0.5, alpha=0, solver='highs')`) of `y8` on `x8` (reshape x8 to a column). Assign the fitted `slope` and `intercept`.",
        setup="x8 = np.array([1, 2, 3, 4, 5, 6, 7, 8], dtype=float)\ny8 = np.array([1, 3, 2, 5, 4, 8, 5, 9], dtype=float)",
        solution="qr = QuantileRegressor(quantile=0.5, alpha=0, solver='highs').fit(x8.reshape(-1, 1), y8)\nslope = qr.coef_[0]\nintercept = qr.intercept_",
        hint="sklearn wants X as 2-D: x8.reshape(-1, 1). coef_ is an array, intercept_ a float.",
        answer_var=["slope", "intercept"],
        tol=1e-4,
    ),
    dict(
        prompt="Calibration check. `q90` is a 90% quantile forecast and `actual` the outcomes. Compute the empirical coverage: the share of hours where actual <= q90. A calibrated q90 should give about 0.9.",
        setup="actual = np.array([10, 12, 9, 15, 11, 13, 8, 14, 10, 12], dtype=float)\nq90 = np.array([13, 13, 12, 14, 13, 13, 12, 15, 13, 13], dtype=float)",
        solution="answer = np.mean(actual <= q90)",
        hint="(actual <= q90) is a boolean array; its mean is the share of True.",
    ),
    dict(
        prompt="Write the realised total cost of a volume decision over several hours: sum over hours of q·p_da + max(L − q, 0)·p_buy − max(q − L, 0)·p_sell, for the arrays given. Assign the total.",
        setup="q = np.array([10.0, 12.0, 9.0, 11.0])\nL = np.array([11.0, 11.0, 11.0, 11.0])\np_da = np.array([100.0, 110.0, 90.0, 100.0])\np_buy = p_da + 30\np_sell = p_da - 10",
        solution="answer = np.sum(q * p_da + np.maximum(L - q, 0) * p_buy - np.maximum(q - L, 0) * p_sell)",
        hint="Three vector terms, then np.sum.",
    ),
    dict(
        prompt="Real data. `L` is a 1% share of 2023 hourly consumption and `q_naive` is the same hour one week earlier (both Series indexed by time, first week dropped). With p_buy = price + 30 and p_sell = price − 10, compute the total *excess* cost of the naive decision over perfect foresight: sum of c_u·max(L − q, 0) + c_o·max(q − L, 0) with c_u = 30, c_o = 10.",
        setup='df = pd.read_csv("../data/hourly_power_clean.csv", parse_dates=["time"]).set_index("time")\nd23 = df[df.index.year == 2023]\nL_all = d23["consumption_mwh"] * 0.01\nq_naive = L_all.shift(168).dropna()\nL = L_all.loc[q_naive.index]',
        solution="c_u, c_o = 30.0, 10.0\nanswer = float((c_u * np.maximum(L - q_naive, 0) + c_o * np.maximum(q_naive - L, 0)).sum())",
        hint="Same excess-cost formula as the toy, on Series aligned by index; .sum() at the end.",
        check_fn=lambda got, exp: (abs(float(got) - exp) <= 1e-6 * abs(exp), "" if abs(float(got) - exp) <= 1e-6 * abs(exp) else f"expected about {exp:,.0f}"),
    ),
    dict(
        prompt="Still on the real data: the naive forecast error is e = L − q_naive. A quantile-based margin at the critical fractile 0.75 is `np.quantile(e, 0.75)`. Compute it (`margin`) and the excess cost when the decision is q_naive + margin (`cost_margin`). It should be lower than task 10.",
        setup='df = pd.read_csv("../data/hourly_power_clean.csv", parse_dates=["time"]).set_index("time")\nd23 = df[df.index.year == 2023]\nL_all = d23["consumption_mwh"] * 0.01\nq_naive = L_all.shift(168).dropna()\nL = L_all.loc[q_naive.index]\nc_u, c_o = 30.0, 10.0',
        solution="e = L - q_naive\nmargin = float(np.quantile(e, 0.75))\nq_adj = q_naive + margin\ncost_margin = float((c_u * np.maximum(L - q_adj, 0) + c_o * np.maximum(q_adj - L, 0)).sum())",
        hint="Add the 0.75-quantile of the past errors to the forecast, then reuse the excess-cost formula.",
        answer_var=["margin", "cost_margin"],
        check_fn=lambda got, exp: (abs(float(got) - exp) <= 1e-6 * max(abs(exp), 1), "" if abs(float(got) - exp) <= 1e-6 * max(abs(exp), 1) else f"expected about {exp:,.2f}"),
        note="In practice the margin must be estimated on past data only (e.g. 2022), never on the period you evaluate.",
    ),
]
