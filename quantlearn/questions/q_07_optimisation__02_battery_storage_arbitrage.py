"""Exercises for 07_optimisation/02_battery_storage_arbitrage.ipynb"""
import numpy as np
import pandas as pd

COMMON_SETUP = """
import numpy as np
import pandas as pd
from scipy.optimize import linprog, milp, LinearConstraint, Bounds
"""

TOY = '''
prices = np.array([10.0, 50.0, 30.0])      # EUR/MWh in hours 0, 1, 2
P_MAX = 1.0                                 # MW  (max charge or discharge per hour)
E_MAX = 1.0                                 # MWh (capacity)
# variable order of x: [ch0, ch1, ch2, dis0, dis1, dis2, soc0, soc1, soc2]; battery starts empty
'''

BUILD = '''
def battery_lp(prices, P_MAX, E_MAX, eta=1.0, deg=0.0, soc_start=0.0, cyclic=False):
    """Return c, A_eq, b_eq, bounds for the 3-block battery LP (charge, discharge, soc)."""
    H = len(prices)
    c = np.concatenate([prices + deg, -prices + deg, np.zeros(H)])
    A_eq = np.zeros((H + (1 if cyclic else 0), 3 * H))
    b_eq = np.zeros(H + (1 if cyclic else 0))
    for h in range(H):
        A_eq[h, h] = -eta            # charge adds eta * ch
        A_eq[h, H + h] = 1.0         # discharge removes dis
        A_eq[h, 2 * H + h] = 1.0     # soc_h
        if h > 0:
            A_eq[h, 2 * H + h - 1] = -1.0
        else:
            b_eq[h] = soc_start
    if cyclic:
        A_eq[H, 2 * H + H - 1] = 1.0
        b_eq[H] = soc_start
    bounds = [(0, P_MAX)] * (2 * H) + [(0, E_MAX)] * H
    return c, A_eq, b_eq, bounds
'''

QUESTIONS = [
    dict(
        prompt="Build the `linprog` cost vector `c` that **maximises** revenue Σ price·(discharge − charge) for the 3-hour toy, in the stated variable order. Give a NumPy array of length 9.",
        setup=TOY,
        solution='answer = np.concatenate([prices, -prices, np.zeros(3)])',
        hint="linprog minimises, so minimise −revenue: +price on charge, −price on discharge, 0 on soc.",
        tol=1e-9,
    ),
    dict(
        prompt="Build the energy-balance matrix `A_eq` (3 × 9): row h says soc_h − soc_{h−1} − ch_h + dis_h = 0, with soc_{−1} = 0 (start empty). Give a NumPy array.",
        setup=TOY,
        solution='''answer = np.array([
    [-1,  0,  0,  1,  0,  0,  1,  0,  0],
    [ 0, -1,  0,  0,  1,  0, -1,  1,  0],
    [ 0,  0, -1,  0,  0,  1,  0, -1,  1],
], dtype=float)''',
        hint="Coefficients: −1 on ch_h, +1 on dis_h, +1 on soc_h, −1 on soc_{h−1} (row 0 has no previous soc).",
        tol=1e-9,
    ),
    dict(
        prompt="Give the `bounds` list: charge and discharge between 0 and P_MAX, soc between 0 and E_MAX, in variable order. A list of 9 (low, high) tuples.",
        setup=TOY,
        solution='answer = [(0, 1.0)] * 6 + [(0, 1.0)] * 3',
        hint="Six (0, P_MAX) then three (0, E_MAX).",
    ),
    dict(
        prompt="Solve the toy LP (b_eq is all zeros) and give the optimal **revenue** in EUR as a number (remember the sign of `res.fun`).",
        setup=TOY + '''
c = np.concatenate([prices, -prices, np.zeros(3)])
A_eq = np.array([[-1, 0, 0, 1, 0, 0, 1, 0, 0],
                 [0, -1, 0, 0, 1, 0, -1, 1, 0],
                 [0, 0, -1, 0, 0, 1, 0, -1, 1]], dtype=float)
b_eq = np.zeros(3)
bounds = [(0, 1.0)] * 9
''',
        solution='res = linprog(c, A_eq=A_eq, b_eq=b_eq, bounds=bounds, method="highs")\nanswer = float(-res.fun)',
        hint="Revenue = −res.fun. Buy 1 MWh at 10, sell it at 50.",
        tol=1e-6,
    ),
    dict(
        prompt="From the same solution, give the state-of-charge trajectory `soc` (the last 3 entries of `res.x`) as a NumPy array.",
        setup=TOY + '''
c = np.concatenate([prices, -prices, np.zeros(3)])
A_eq = np.array([[-1, 0, 0, 1, 0, 0, 1, 0, 0],
                 [0, -1, 0, 0, 1, 0, -1, 1, 0],
                 [0, 0, -1, 0, 0, 1, 0, -1, 1]], dtype=float)
b_eq = np.zeros(3)
bounds = [(0, 1.0)] * 9
''',
        solution='res = linprog(c, A_eq=A_eq, b_eq=b_eq, bounds=bounds, method="highs")\nanswer = res.x[6:]',
        hint="Full after hour 0, empty after hour 1, empty after hour 2.",
        tol=1e-6,
    ),
    dict(
        prompt="Add a round-trip efficiency: only 80% of charged energy reaches the battery (soc_h = soc_{h−1} + 0.8·ch_h − dis_h). Rebuild `A_eq`, solve, and give the revenue as a number.",
        setup=TOY + '''
c = np.concatenate([prices, -prices, np.zeros(3)])
b_eq = np.zeros(3)
bounds = [(0, 1.0)] * 9
''',
        solution='''A_eq = np.array([[-0.8, 0, 0, 1, 0, 0, 1, 0, 0],
                 [0, -0.8, 0, 0, 1, 0, -1, 1, 0],
                 [0, 0, -0.8, 0, 0, 1, 0, -1, 1]], dtype=float)
res = linprog(c, A_eq=A_eq, b_eq=b_eq, bounds=bounds, method="highs")
answer = float(-res.fun)''',
        hint="Only the charge coefficients change: −1 → −0.8. Charge 1 at 10, only 0.8 MWh arrives, sell 0.8 at 50.",
        tol=1e-6,
        note="If you had applied the efficiency to discharge instead (dis/0.8), you would have created energy: revenue 52.5.",
    ),
    dict(
        prompt="Negative prices: use `prices_neg = [-20, -20, 50]` with efficiency 0.8 and no degradation cost. Solve and give a boolean: does the optimal solution charge **and** discharge in the same hour (both > 1e-6) for any hour?",
        setup=TOY + BUILD + '''
prices_neg = np.array([-20.0, -20.0, 50.0])
''',
        solution='''c, A_eq, b_eq, bounds = battery_lp(prices_neg, P_MAX, E_MAX, eta=0.8)
res = linprog(c, A_eq=A_eq, b_eq=b_eq, bounds=bounds, method="highs")
ch, dis = res.x[:3], res.x[3:6]
answer = bool(((ch > 1e-6) & (dis > 1e-6)).any())''',
        hint="Use the provided battery_lp helper with eta=0.8. Being paid to consume makes wasting energy (charge and discharge at once) profitable.",
    ),
    dict(
        prompt="Add a degradation cost of 3 EUR per MWh of throughput (charged + discharged) to the negative-price case (eta 0.8). Give the optimal revenue net of degradation as a number.",
        setup=TOY + BUILD + '''
prices_neg = np.array([-20.0, -20.0, 50.0])
''',
        solution='''c, A_eq, b_eq, bounds = battery_lp(prices_neg, P_MAX, E_MAX, eta=0.8, deg=3.0)
res = linprog(c, A_eq=A_eq, b_eq=b_eq, bounds=bounds, method="highs")
answer = float(-res.fun)''',
        hint="battery_lp(..., deg=3.0) adds +3 to every charge and discharge cost coefficient.",
        tol=1e-6,
    ),
    dict(
        prompt="Start the battery **full** (soc_start = 1) with the original prices [10, 50, 30], eta = 1, no end constraint. Give the revenue. Then note why it is higher than 40.",
        setup=TOY + BUILD,
        solution='''c, A_eq, b_eq, bounds = battery_lp(prices, P_MAX, E_MAX, soc_start=1.0)
res = linprog(c, A_eq=A_eq, b_eq=b_eq, bounds=bounds, method="highs")
answer = float(-res.fun)''',
        hint="battery_lp(..., soc_start=1.0). The free initial MWh is sold at 50 and never paid for.",
        tol=1e-6,
        note="The extra revenue is the free initial energy. A cyclic constraint (end soc = start soc) removes it.",
    ),
    dict(
        prompt="Same as before but force the battery to end where it started (`cyclic=True`, soc_start 1). Give the revenue.",
        setup=TOY + BUILD,
        solution='''c, A_eq, b_eq, bounds = battery_lp(prices, P_MAX, E_MAX, soc_start=1.0, cyclic=True)
res = linprog(c, A_eq=A_eq, b_eq=b_eq, bounds=bounds, method="highs")
answer = float(-res.fun)''',
        hint="battery_lp(..., soc_start=1.0, cyclic=True) adds a row soc_2 = 1. Now every MWh sold must be bought back.",
        tol=1e-6,
    ),
    dict(
        prompt="Heuristic schedule for `prices6` with P_MAX 1 and E_MAX 2: charge in the 2 cheapest hours, discharge in the 2 most expensive. Give the charge plan as a NumPy array of 0/1 values (length 6).",
        setup='''prices6 = np.array([30.0, 10.0, 60.0, 20.0, 80.0, 40.0])
P_MAX, E_MAX = 1.0, 2.0''',
        solution='''n = int(E_MAX / P_MAX)
order = np.argsort(prices6)
answer = np.zeros(6)
answer[order[:n]] = 1.0''',
        hint="np.argsort(prices6) gives positions from cheapest to dearest; the first n are charge hours.",
        tol=1e-9,
    ),
    dict(
        prompt="Real data: load `../data/hourly_power_clean.csv`, take the 24 prices of 2023-01-24, and solve the perfect-foresight LP for a 50 MW / 100 MWh battery with eta 0.85, no degradation, starting empty, no end constraint. Give the day's revenue in EUR.",
        setup=BUILD + '''
df = pd.read_csv("../data/hourly_power_clean.csv", parse_dates=["time"]).set_index("time")
p24 = df.loc["2023-01-24", "price_eur_mwh"].to_numpy()''',
        solution='''c, A_eq, b_eq, bounds = battery_lp(p24, 50.0, 100.0, eta=0.85)
res = linprog(c, A_eq=A_eq, b_eq=b_eq, bounds=bounds, method="highs")
answer = float(-res.fun)''',
        hint="battery_lp(p24, 50.0, 100.0, eta=0.85), then −res.fun.",
        tol=1e-2,
    ),
]
