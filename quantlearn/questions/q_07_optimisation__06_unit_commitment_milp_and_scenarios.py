"""Exercises for 07_optimisation/06_unit_commitment_milp_and_scenarios.ipynb"""
import numpy as np
import pandas as pd

COMMON_SETUP = """
import numpy as np
import pandas as pd
from scipy.optimize import milp, linprog, LinearConstraint, Bounds, differential_evolution, minimize
"""

_TINY = """
# minimise 2x + 3y  subject to  x + y >= 4.5,  0 <= x <= 3,  y >= 0
c = np.array([2.0, 3.0])
A = np.array([[1.0, 1.0]])
constraints = LinearConstraint(A, [4.5], [np.inf])
bounds = Bounds([0, 0], [3, np.inf])
"""

_UC = """
# Two units, two hours. Variable order:
#   p_A0, p_A1, p_B0, p_B1   production (MW)      continuous
#   u_A0, u_A1, u_B0, u_B1   on/off               binary
#   s_A0, s_A1, s_B0, s_B1   start-up indicator   binary
names = ["p_A0", "p_A1", "p_B0", "p_B1", "u_A0", "u_A1", "u_B0", "u_B1", "s_A0", "s_A1", "s_B0", "s_B1"]
col = {n: i for i, n in enumerate(names)}
demand = np.array([6.0, 11.0])
mc    = {"A": 10.0, "B": 30.0}     # marginal cost EUR/MWh
oncost = {"A": 5.0, "B": 2.0}      # cost per hour while on
startup = {"A": 20.0, "B": 10.0}   # cost of switching on (both units start OFF)
pmax = {"A": 8.0, "B": 6.0}
pmin = {"A": 2.0, "B": 1.0}

c = np.zeros(12)
for g in ["A", "B"]:
    for h in range(2):
        c[col[f"p_{g}{h}"]] = mc[g]
        c[col[f"u_{g}{h}"]] = oncost[g]
        c[col[f"s_{g}{h}"]] = startup[g]

rows, lb, ub = [], [], []
for h in range(2):                                   # demand: p_A + p_B = demand
    r = np.zeros(12); r[col[f"p_A{h}"]] = 1; r[col[f"p_B{h}"]] = 1
    rows.append(r); lb.append(demand[h]); ub.append(demand[h])
for g in ["A", "B"]:
    for h in range(2):                               # p <= pmax * u   ->  p - pmax*u <= 0
        r = np.zeros(12); r[col[f"p_{g}{h}"]] = 1; r[col[f"u_{g}{h}"]] = -pmax[g]
        rows.append(r); lb.append(-np.inf); ub.append(0.0)
        r = np.zeros(12); r[col[f"p_{g}{h}"]] = 1; r[col[f"u_{g}{h}"]] = -pmin[g]   # p >= pmin * u
        rows.append(r); lb.append(0.0); ub.append(np.inf)
        r = np.zeros(12); r[col[f"s_{g}{h}"]] = 1; r[col[f"u_{g}{h}"]] = -1          # s >= u_h - u_{h-1}
        if h > 0:
            r[col[f"u_{g}{h-1}"]] = 1
        rows.append(r); lb.append(0.0); ub.append(np.inf)
A = np.array(rows); lb = np.array(lb); ub = np.array(ub)
integrality = np.array([0, 0, 0, 0, 1, 1, 1, 1, 1, 1, 1, 1])
hi = np.array([pmax["A"], pmax["A"], pmax["B"], pmax["B"]] + [1.0] * 8)
"""

_KN = """
proj = pd.DataFrame({"capex": [40.0, 50.0, 30.0, 60.0], "margin": [10.0, 13.0, 7.0, 15.0]},
                    index=["A", "B", "C", "D"])
budget = 100.0
"""

QUESTIONS = [
    dict(
        prompt="Solve the tiny problem as a plain LP (no integrality) with `milp`: minimise 2x + 3y subject to x + y ≥ 4.5, 0 ≤ x ≤ 3, y ≥ 0. Assign the optimal `x_lp` (array of the two variables) and `cost_lp`.",
        setup=_TINY,
        solution="res = milp(c, constraints=constraints, integrality=np.array([0, 0]), bounds=bounds)\nx_lp = res.x\ncost_lp = res.fun",
        hint="milp(c, constraints=..., integrality=np.array([0, 0]), bounds=...). res.x and res.fun.",
        answer_var=["x_lp", "cost_lp"],
        tol=1e-4,
    ),
    dict(
        prompt="Now require y to be an integer (integrality = [0, 1]). Assign the MILP solution `x_mip` and its cost `cost_mip`. Why is the cost higher?",
        setup=_TINY,
        solution="res = milp(c, constraints=constraints, integrality=np.array([0, 1]), bounds=bounds)\nx_mip = res.x\ncost_mip = res.fun",
        hint="Only the integrality vector changes. y = 1.5 is no longer allowed; the solver must pick y = 2 and adjust x.",
        answer_var=["x_mip", "cost_mip"],
        tol=1e-4,
        note="The LP gives (3, 1.5) at cost 10.5; the MILP gives (2.5, 2) at cost 11. Integer constraints can only make the optimum worse (or equal).",
    ),
    dict(
        prompt="Make the tiny problem infeasible: keep x ≤ 3 but also cap y ≤ 1 (so x + y ≤ 4 < 4.5). Solve with milp and assign the returned `status` code (2 means infeasible).",
        setup=_TINY,
        solution="res = milp(c, constraints=constraints, integrality=np.array([0, 1]), bounds=Bounds([0, 0], [3, 1]))\nstatus = res.status",
        hint="Change the upper bound of y in Bounds. Always read res.status before using res.x (which is None here).",
        answer_var="status",
    ),
    dict(
        prompt="Unit commitment, 2 units × 2 hours: `c`, `A`, `lb`, `ub`, `integrality` and the variable upper bounds `hi` are built for you (read the comments). Solve with milp (lower bounds 0). Assign the optimal `cost` and the four on/off decisions `u` as an integer array in the order u_A0, u_A1, u_B0, u_B1.",
        setup=_UC,
        solution="res = milp(c, constraints=LinearConstraint(A, lb, ub), integrality=integrality, bounds=Bounds(0, hi))\ncost = res.fun\nu = np.round(res.x[4:8]).astype(int)",
        hint="LinearConstraint(A, lb, ub) and Bounds(0, hi). The u variables are positions 4..7 of res.x.",
        answer_var=["cost", "u"],
        tol=1e-4,
    ),
    dict(
        prompt="From the same MILP solution, assign the production plan `p` as a 2×2 DataFrame with index ['A', 'B'] and columns [0, 1] (unit × hour). Check by eye that each column sums to the demand.",
        setup=_UC,
        solution="res = milp(c, constraints=LinearConstraint(A, lb, ub), integrality=integrality, bounds=Bounds(0, hi))\np = pd.DataFrame(res.x[:4].reshape(2, 2), index=['A', 'B'], columns=[0, 1])",
        hint="res.x[:4] holds p_A0, p_A1, p_B0, p_B1; reshape(2, 2) gives rows A, B and columns hours 0, 1.",
        answer_var="p",
        tol=1e-4,
    ),
    dict(
        prompt="Solve the LP relaxation of the unit-commitment instance (integrality all zeros). Assign `cost_lp` and the relative gap `gap = cost_mip / cost_lp − 1`.",
        setup=_UC,
        solution="res = milp(c, constraints=LinearConstraint(A, lb, ub), integrality=integrality, bounds=Bounds(0, hi))\nres_lp = milp(c, constraints=LinearConstraint(A, lb, ub), integrality=np.zeros(12), bounds=Bounds(0, hi))\ncost_lp = res_lp.fun\ngap = res.fun / res_lp.fun - 1",
        hint="np.zeros(12) as integrality. The LP is always cheaper or equal because fractional u is allowed.",
        answer_var=["cost_lp", "gap"],
        tol=1e-4,
    ),
    dict(
        prompt="Rounding the relaxed solution does not work. In the LP relaxation u_B1 is fractional. Force it to 0 (round down) by setting both its lower and upper bound to 0, re-solve as an LP, and assign the resulting `status` (expect 2 = infeasible: unit A alone cannot cover 11 MW).",
        setup=_UC,
        solution="lo_r = np.zeros(12); hi_r = hi.copy()\nlo_r[col['u_B1']] = 0.0; hi_r[col['u_B1']] = 0.0\nres_r = milp(c, constraints=LinearConstraint(A, lb, ub), integrality=np.zeros(12), bounds=Bounds(lo_r, hi_r))\nstatus = res_r.status",
        hint="Bounds(lo, hi) with lo and hi arrays; set position col['u_B1'] to 0 in both.",
        answer_var="status",
    ),
    dict(
        prompt="Knapsack. Choose the subset of projects in `proj` that maximises total margin with total capex ≤ `budget`, using milp with all variables binary (minimise −margin). Assign the chosen project names as a sorted list `chosen` and the total margin `best_margin`.",
        setup=_KN,
        solution="c_kn = -proj['margin'].values\nA_kn = proj['capex'].values.reshape(1, -1)\nkn = milp(c_kn, constraints=LinearConstraint(A_kn, -np.inf, budget), integrality=np.ones(4), bounds=Bounds(0, 1))\nchosen = sorted(proj.index[kn.x > 0.5].tolist())\nbest_margin = float(-kn.fun)",
        hint="milp minimises, so pass the negative margins. One constraint row: capex · x ≤ budget. Read the chosen items with kn.x > 0.5.",
        answer_var=["chosen", "best_margin"],
        tol=1e-6,
    ),
    dict(
        prompt="Greedy by margin-per-capex: sort projects by margin/capex descending and add each one if it still fits in the budget. Assign the greedy choice as a sorted list `greedy` and its total margin `greedy_margin`. Compare with the exact answer.",
        setup=_KN,
        solution="ratio = (proj['margin'] / proj['capex']).sort_values(ascending=False)\ngreedy, spent = [], 0.0\nfor name in ratio.index:\n    if spent + proj.loc[name, 'capex'] <= budget:\n        greedy.append(name); spent += proj.loc[name, 'capex']\ngreedy = sorted(greedy)\ngreedy_margin = float(proj.loc[greedy, 'margin'].sum())",
        hint="Compute the ratio Series, sort it, loop and keep a running spend.",
        answer_var=["greedy", "greedy_margin"],
        note="Greedy picks B then A (margin 23) and leaves 10 of budget unused; the exact answer A + D gives 25.",
    ),
    dict(
        prompt="Two-stage hedge as an LP. Decide the forward volume F now (price K), then in each of 2 equally likely scenarios s buy the shortfall at P_s + c_u or sell the surplus at P_s − c_o. Variables: [F, short_1, long_1, short_2, long_2]. Minimise K·F + Σ_s ½[(P_s + c_u)·short_s − (P_s − c_o)·long_s] subject to F + short_s − long_s = L_s, 0 ≤ F ≤ 20, others ≥ 0. Solve with linprog (method='highs') and assign `F_star`.",
        setup="L = np.array([10.0, 14.0])\nP = np.array([100.0, 120.0])\nK = 105.0\nc_u, c_o = 25.0, 12.0",
        solution="c_2s = np.array([K, (P[0] + c_u) / 2, -(P[0] - c_o) / 2, (P[1] + c_u) / 2, -(P[1] - c_o) / 2])\nA_eq = np.array([[1, 1, -1, 0, 0],\n                 [1, 0, 0, 1, -1]])\nr = linprog(c_2s, A_eq=A_eq, b_eq=L, bounds=[(0, 20)] + [(0, None)] * 4, method='highs')\nF_star = r.x[0]",
        hint="Two equality rows, one per scenario. The long variables have negative cost (you earn when selling). Check r.status == 0.",
        answer_var="F_star",
        tol=1e-4,
        note="F* = 14: hedging the high scenario fully is worth it because a shortfall costs 145 while a surplus still earns 88 or 108 against a forward price of 105.",
    ),
    dict(
        prompt="`bumpy(x) = (x − 2)² + 3·sin(3x)` has several local minima. Run `minimize` (Nelder-Mead) from x0 = 0 and `differential_evolution` on bounds [(−2, 6)] with seed=0. Assign the two minimisers `x_local` and `x_global` (floats). Are they the same?",
        setup="def bumpy(x_):\n    x_ = np.asarray(x_).ravel()[0]\n    return (x_ - 2) ** 2 + 3 * np.sin(3 * x_)",
        solution="x_local = float(minimize(bumpy, x0=[0.0], method='Nelder-Mead').x[0])\nx_global = float(differential_evolution(bumpy, bounds=[(-2, 6)], seed=0).x[0])",
        hint="minimize(..., method='Nelder-Mead').x[0]; differential_evolution(f, bounds=[(-2, 6)], seed=0).x[0].",
        answer_var=["x_local", "x_global"],
        tol=1e-3,
    ),
    dict(
        prompt="Real data, merit order by hand. For 2023-01-24, the demand in GW is `demand_gw` (24 values). Three units: nuclear 10 GW at 12 EUR/MWh, ccgt 20 GW at 80, peaker 10 GW at 160. Dispatch each hour greedily (cheapest first up to its capacity) and assign the total energy cost for the day in EUR (GW × 1000 = MW, hourly steps so MW·h = MWh).",
        setup='df = pd.read_csv("../data/hourly_power_clean.csv", parse_dates=["time"]).set_index("time")\ndemand_gw = df.loc["2023-01-24", "consumption_mwh"].values / 1000\ncap = np.array([10.0, 20.0, 10.0])\nmc = np.array([12.0, 80.0, 160.0])',
        solution="total = 0.0\nfor d in demand_gw:\n    remaining = d\n    for k in range(3):\n        take = min(cap[k], remaining)\n        total += take * 1000 * mc[k]\n        remaining -= take\nanswer = total",
        hint="For each hour: take min(capacity, remaining demand) from each unit in cost order; cost = GW × 1000 × EUR/MWh.",
        check_fn=lambda got, exp: (abs(float(got) - exp) <= 1e-6 * abs(exp), "" if abs(float(got) - exp) <= 1e-6 * abs(exp) else f"expected about {exp:,.0f}"),
    ),
]
