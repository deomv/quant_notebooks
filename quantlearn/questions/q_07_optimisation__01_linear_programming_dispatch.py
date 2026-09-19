"""Exercises for 07_optimisation/01_linear_programming_dispatch.ipynb"""
import numpy as np
import pandas as pd

COMMON_SETUP = """
import numpy as np
import pandas as pd
from scipy.optimize import linprog
"""

TOY = '''
gens = pd.DataFrame({"cost": [12, 50], "cap": [10, 20]}, index=["nuclear", "gas"])   # EUR/MWh, MW
demand = np.array([8, 25])                                                             # MW in hour 0, hour 1
names = ["nuclear_h0", "nuclear_h1", "gas_h0", "gas_h1"]                               # variable order of x
'''

QUESTIONS = [
    dict(
        prompt="Two generators, two hours. Build the cost vector `c` for `linprog` in the stated variable order `names` (nuclear_h0, nuclear_h1, gas_h0, gas_h1) as a NumPy array.",
        setup=TOY,
        solution='answer = np.array([12, 12, 50, 50], dtype=float)',
        hint="Each generator's cost repeats once per hour, in the order of `names`.",
    ),
    dict(
        prompt="Build `A_eq` (2 rows × 4 columns) so that row h says 'nuclear_h + gas_h = demand_h', using the same variable order.",
        setup=TOY,
        solution='answer = np.array([[1, 0, 1, 0],\n                   [0, 1, 0, 1]], dtype=float)',
        hint="Row 0 has a 1 in the columns of nuclear_h0 and gas_h0; row 1 in nuclear_h1 and gas_h1.",
    ),
    dict(
        prompt="Build the `bounds` list for `linprog`: each variable between 0 and its generator's capacity, in the order of `names`. Give it as a list of (low, high) tuples.",
        setup=TOY,
        solution='answer = [(0, 10), (0, 10), (0, 20), (0, 20)]',
        hint="One (0, cap) tuple per variable; nuclear twice, then gas twice.",
    ),
    dict(
        prompt="Solve the dispatch LP with `linprog(method='highs')` and give the optimal `x` as a NumPy array (order of `names`).",
        setup=TOY + '''
c = np.array([12, 12, 50, 50], dtype=float)
A_eq = np.array([[1, 0, 1, 0], [0, 1, 0, 1]], dtype=float)
bounds = [(0, 10), (0, 10), (0, 20), (0, 20)]
''',
        solution='res = linprog(c, A_eq=A_eq, b_eq=demand, bounds=bounds, method="highs")\nanswer = res.x',
        hint="linprog(c, A_eq=A_eq, b_eq=demand, bounds=bounds, method='highs') then res.x. Hour 0 needs 8 MW: all nuclear. Hour 1 needs 25: nuclear 10 + gas 15.",
        tol=1e-6,
    ),
    dict(
        prompt="Give the optimal total cost (the objective value) of the same LP as a number.",
        setup=TOY + '''
c = np.array([12, 12, 50, 50], dtype=float)
A_eq = np.array([[1, 0, 1, 0], [0, 1, 0, 1]], dtype=float)
bounds = [(0, 10), (0, 10), (0, 20), (0, 20)]
''',
        solution='res = linprog(c, A_eq=A_eq, b_eq=demand, bounds=bounds, method="highs")\nanswer = float(res.fun)',
        hint="res.fun. By hand: 8·12 + 10·12 + 15·50.",
        tol=1e-6,
    ),
    dict(
        prompt="Give the dual values (shadow prices) of the two demand constraints as a NumPy array of length 2. They are the marginal cost of one extra MW in each hour.",
        setup=TOY + '''
c = np.array([12, 12, 50, 50], dtype=float)
A_eq = np.array([[1, 0, 1, 0], [0, 1, 0, 1]], dtype=float)
bounds = [(0, 10), (0, 10), (0, 20), (0, 20)]
''',
        solution='res = linprog(c, A_eq=A_eq, b_eq=demand, bounds=bounds, method="highs")\nanswer = np.asarray(res.eqlin.marginals)',
        hint="res.eqlin.marginals. Hour 0 is served by nuclear with spare capacity (12); hour 1's marginal unit is gas (50).",
        tol=1e-6,
    ),
    dict(
        prompt="Raise hour-1 demand to 35 MW (total capacity is only 30). Solve and give `res.status` as an integer.",
        setup=TOY + '''
c = np.array([12, 12, 50, 50], dtype=float)
A_eq = np.array([[1, 0, 1, 0], [0, 1, 0, 1]], dtype=float)
bounds = [(0, 10), (0, 10), (0, 20), (0, 20)]
demand_big = np.array([8, 35])
''',
        solution='res = linprog(c, A_eq=A_eq, b_eq=demand_big, bounds=bounds, method="highs")\nanswer = int(res.status)',
        hint="Status 0 = optimal, 2 = infeasible. Always check it before reading res.x.",
    ),
    dict(
        prompt="Make the infeasible instance feasible by adding one 'unserved energy' slack variable per hour at a penalty of 1000 EUR/MWh (variable order: the 4 generator variables, then unserved_h0, unserved_h1). Solve and give the unserved MWh in hour 1 as a number.",
        setup=TOY + '''
demand_big = np.array([8, 35])
''',
        solution='''c = np.array([12, 12, 50, 50, 1000, 1000], dtype=float)
A_eq = np.array([[1, 0, 1, 0, 1, 0],
                 [0, 1, 0, 1, 0, 1]], dtype=float)
bounds = [(0, 10), (0, 10), (0, 20), (0, 20), (0, None), (0, None)]
res = linprog(c, A_eq=A_eq, b_eq=demand_big, bounds=bounds, method="highs")
answer = float(res.x[5])''',
        hint="Append two columns of 1s to A_eq (one per hour), cost 1000, bounds (0, None). Hour 1: 10 + 20 = 30 served, 5 short.",
        tol=1e-6,
    ),
    dict(
        prompt="Greedy merit order for a single hour with demand 25 MW: sort generators by cost and fill capacity in order. Give the dispatch as a NumPy array in the order [nuclear, gas].",
        setup=TOY,
        solution='''order = gens.sort_values("cost")
remaining = 25.0
dispatch = pd.Series(0.0, index=gens.index)
for g in order.index:
    take = min(order.loc[g, "cap"], remaining)
    dispatch[g] = take
    remaining -= take
answer = dispatch[["nuclear", "gas"]].to_numpy()''',
        hint="Loop over generators sorted by cost; each takes min(cap, remaining demand).",
        tol=1e-6,
    ),
    dict(
        prompt="`linprog` minimises. To **maximise** profit Σ (price_h − cost_g)·x, what cost vector `c` do you pass? Prices are 40 EUR/MWh in hour 0 and 60 in hour 1. Give `c` as a NumPy array in the order of `names`.",
        setup=TOY + '''
price = np.array([40, 60])
''',
        solution='''profit = np.array([price[0] - 12, price[1] - 12, price[0] - 50, price[1] - 50], dtype=float)
answer = -profit''',
        hint="Negate the profit per MWh: c = -(price_h - cost_g) for each variable.",
        tol=1e-6,
    ),
    dict(
        prompt="After solving the original LP, which generator sets the price (is marginal) in hour 1? Give its name as a string.",
        setup=TOY,
        solution='answer = "gas"',
        hint="The marginal unit is the one that is dispatched but not at full capacity; its cost equals the dual.",
    ),
    dict(
        prompt="Real data: load `../data/hourly_power_clean.csv`, take consumption at 2023-01-24 18:00 UTC scaled to 10% (MW), and dispatch it with a greedy merit order on the fleet `gens5` (ignore wind availability, wind capacity is fully available). Give the total cost in EUR as a number.",
        setup='''df = pd.read_csv("../data/hourly_power_clean.csv", parse_dates=["time"]).set_index("time")
gens5 = pd.DataFrame({"cost": [12, 0, 65, 85, 160],
                      "cap":  [900, 1500, 600, 1400, 500]},
                     index=["nuclear", "wind", "imports", "ccgt", "ocgt"])
demand_h = float(df.loc["2023-01-24 18:00", "consumption_mwh"]) * 0.10''',
        solution='''order = gens5.sort_values("cost")
remaining = demand_h
cost = 0.0
for g in order.index:
    take = min(order.loc[g, "cap"], remaining)
    cost += take * order.loc[g, "cost"]
    remaining -= take
answer = float(cost)''',
        hint="Same greedy loop as before; accumulate take × cost. Wind (cost 0) and nuclear go first.",
        tol=1e-3,
    ),
]
