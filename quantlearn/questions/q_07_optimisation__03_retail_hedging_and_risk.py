"""Exercises for 07_optimisation/03_retail_hedging_and_risk.ipynb"""
import numpy as np
import pandas as pd

COMMON_SETUP = """
import numpy as np
import pandas as pd
from scipy.optimize import linprog, minimize
"""

TOY = '''
toy = pd.DataFrame({"cost":   [75.0, 160.0, 110.0, 60.0],     # unhedged cost per month, kEUR
                    "payoff": [-10.0, 20.0, 5.0, -15.0]},     # forward payoff per MW, kEUR
                   index=["Jan", "Feb", "Mar", "Apr"])
'''

CVAR = '''
a = np.array([100.0, 120.0, 90.0, 200.0])      # unhedged cost per scenario
B = np.array([-10.0, 10.0, -20.0, 60.0])       # forward payoff per MW per scenario
w = np.array([0.25, 0.25, 0.25, 0.25])         # scenario probabilities
ALPHA = 0.75
'''

QUESTIONS = [
    dict(
        prompt="Compute the sample covariance (ddof=1) between `cost` and `payoff` in `toy` as a number.",
        setup=TOY,
        solution='answer = float(toy["cost"].cov(toy["payoff"]))',
        hint="Series.cov uses n−1. By hand: sum of (cost − mean)(payoff − mean) divided by 3.",
        tol=1e-6,
    ),
    dict(
        prompt="Compute the minimum-variance hedge ratio Q* = cov(cost, payoff) / var(payoff) in MW, both with ddof=1.",
        setup=TOY,
        solution='answer = float(toy["cost"].cov(toy["payoff"]) / toy["payoff"].var())',
        hint="Series.cov and Series.var both default to ddof=1, so they are consistent.",
        tol=1e-6,
    ),
    dict(
        prompt="Build the hedged cost Series: cost − Q*·payoff, keeping the month index.",
        setup=TOY + '''
Q_star = toy["cost"].cov(toy["payoff"]) / toy["payoff"].var()
''',
        solution='answer = toy["cost"] - Q_star * toy["payoff"]',
        hint="Vectorised subtraction on the columns; the index comes along.",
        tol=1e-6,
    ),
    dict(
        prompt="Compute the hedge effectiveness 1 − var(hedged) / var(cost) as a number.",
        setup=TOY + '''
Q_star = toy["cost"].cov(toy["payoff"]) / toy["payoff"].var()
hedged = toy["cost"] - Q_star * toy["payoff"]
''',
        solution='answer = float(1 - hedged.var() / toy["cost"].var())',
        hint="Both variances with the same ddof; the ratio is what matters.",
        tol=1e-6,
    ),
    dict(
        prompt="Show that OLS gives the same hedge: regress `cost` on `payoff` with `np.polyfit` (degree 1) and give the slope.",
        setup=TOY,
        solution='answer = float(np.polyfit(toy["payoff"], toy["cost"], 1)[0])',
        hint="np.polyfit(x, y, 1) returns [slope, intercept]; x is the payoff.",
        tol=1e-6,
    ),
    dict(
        prompt="Reproduce the ddof pitfall: compute the **wrong** ratio `np.cov(cost, payoff)[0, 1] / np.var(payoff)` (np.var defaults to ddof=0) and give it as a number, so you can see how far it is from Q*.",
        setup=TOY,
        solution='answer = float(np.cov(toy["cost"], toy["payoff"])[0, 1] / np.var(toy["payoff"]))',
        hint="np.cov uses n−1 by default, np.var uses n. Mixing them inflates the ratio by n/(n−1).",
        tol=1e-6,
        note="Q* is 2.8; the mismatched version gives 3.733 = 2.8 × 4/3.",
    ),
    dict(
        prompt="Volumetric risk on two hours: price [50, 100] EUR/MWh, load [2, 3] MWh. The true cost is Σ price·load. A flat hedge assumes mean(price)·mean(load)·2. Give the difference (true − assumed) as a number: that is the price–load covariance term.",
        setup='''price = np.array([50.0, 100.0])
load = np.array([2.0, 3.0])''',
        solution='answer = float((price * load).sum() - price.mean() * load.mean() * len(price))',
        hint="(50·2 + 100·3) − (75 · 2.5 · 2).",
        tol=1e-6,
    ),
    dict(
        prompt="For the 4 equally likely unhedged costs in `a`, compute CVaR at 75% by hand: the mean of the worst 25% of scenarios (here: the single worst one). Give a number.",
        setup=CVAR,
        solution='''k = int(round((1 - ALPHA) * len(a)))
answer = float(np.sort(a)[-k:].mean())''',
        hint="Sort a, take the top (1 − α)·n = 1 value, average.",
        tol=1e-6,
    ),
    dict(
        prompt="Solve the Rockafellar–Uryasev LP for the hedge that minimises E[hedged cost] + CVaR_75. Variables x = [q, t, z1, z2, z3, z4], hedged cost per scenario = a − q·B, constraints z_k ≥ a_k − q·B_k − t and z ≥ 0, q ≥ 0, t free. Give `q` and `t` (assign both).",
        setup=CVAR,
        solution='''n = len(a)
# objective: E[a - qB] + t + 1/(1-alpha) * sum(w z)  -> coefficients on [q, t, z...]
c = np.concatenate([[-(w * B).sum(), 1.0], w / (1 - ALPHA)])
# z_k >= a_k - q B_k - t  ->  -B_k q - t - z_k <= -a_k
A_ub = np.zeros((n, 2 + n))
A_ub[:, 0] = -B
A_ub[:, 1] = -1.0
A_ub[:, 2:] = -np.eye(n)
b_ub = -a
bounds = [(0, None), (None, None)] + [(0, None)] * n
res = linprog(c, A_ub=A_ub, b_ub=b_ub, bounds=bounds, method="highs")
q = float(res.x[0])
t = float(res.x[1])''',
        hint="Objective coefficients: −E[B] on q, 1 on t, w_k/(1−α) on z_k. Each scenario is one ≤ row. t must be unbounded below.",
        answer_var=["q", "t"],
        tol=1e-3,
        note="t equals the empirical VaR75 of the hedged costs; with 4 scenarios that is the worst hedged cost.",
    ),
    dict(
        prompt="Minimise f(x, y) = (x − 1)² + (y − 2)² subject to x + y ≤ 2 and x, y ≥ 0 with `scipy.optimize.minimize(method='SLSQP')` from x0 = [0, 0]. Give the optimal point `res.x` as a NumPy array.",
        setup='',
        solution='''def fun(v):
    return (v[0] - 1) ** 2 + (v[1] - 2) ** 2
cons = [{"type": "ineq", "fun": lambda v: 2 - v[0] - v[1]}]
res = minimize(fun, np.array([0.0, 0.0]), method="SLSQP", bounds=[(0, None), (0, None)], constraints=cons)
answer = res.x''',
        hint="Inequality constraints are given as functions that must be ≥ 0: 2 − x − y. The unconstrained optimum (1, 2) violates it, so the answer sits on the line x + y = 2.",
        tol=1e-3,
    ),
    dict(
        prompt="Check convergence: for the same SLSQP problem, give `res.success` as a boolean.",
        setup='',
        solution='''def fun(v):
    return (v[0] - 1) ** 2 + (v[1] - 2) ** 2
cons = [{"type": "ineq", "fun": lambda v: 2 - v[0] - v[1]}]
res = minimize(fun, np.array([0.0, 0.0]), method="SLSQP", bounds=[(0, None), (0, None)], constraints=cons)
answer = bool(res.success)''',
        hint="res.success; always print it together with res.message.",
    ),
    dict(
        prompt="Real data: load `../data/hourly_power_clean.csv` and compute the Pearson correlation between `price_eur_mwh` and `consumption_mwh` over 2022 only. Give a number. (Positive correlation is why a flat hedge under-covers the expensive hours.)",
        setup='df = pd.read_csv("../data/hourly_power_clean.csv", parse_dates=["time"]).set_index("time")',
        solution='answer = float(df.loc["2022", "price_eur_mwh"].corr(df.loc["2022", "consumption_mwh"]))',
        hint="Partial-string indexing df.loc['2022'] then Series.corr.",
        tol=1e-6,
    ),
]
