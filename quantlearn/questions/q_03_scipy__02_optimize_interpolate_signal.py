"""Exercises for 03_scipy/02_optimize_interpolate_signal.ipynb"""
import numpy as np
import pandas as pd
from scipy import optimize, interpolate, signal

COMMON_SETUP = """
import numpy as np
import pandas as pd
from scipy import optimize, interpolate, signal
rng = np.random.default_rng(0)
"""

QUESTIONS = [
    dict(
        prompt="Minimise f(x) = (x − 3)² + 2 with `optimize.minimize` starting from x0 = 0. Assign the minimiser (a float) to `x_star` and the minimum value to `f_star`.",
        setup="def f(x):\n    return (x[0] - 3) ** 2 + 2",
        solution="res = optimize.minimize(f, x0=[0.0])\nx_star = float(res.x[0])\nf_star = float(res.fun)",
        hint="optimize.minimize(f, x0=[0.0]) returns an object with .x (an array) and .fun.",
        answer_var=["x_star", "f_star"],
        tol=1e-4,
    ),
    dict(
        prompt="Minimise the two-variable function g(x, y) = (x − 1)² + (y + 2)² from the start point (0, 0). Assign the minimiser as a NumPy array of length 2 to `answer`.",
        setup="def g(p):\n    return (p[0] - 1) ** 2 + (p[1] + 2) ** 2",
        solution="answer = optimize.minimize(g, x0=[0.0, 0.0]).x",
        hint="Same call as before with a 2-element x0; .x is the array you want.",
        tol=1e-4,
    ),
    dict(
        prompt="The points `(x, y)` lie exactly on a straight line. Fit `line(x, a, b) = a*x + b` with `optimize.curve_fit` and assign the fitted parameters array `popt` to `answer`.",
        setup="def line(x, a, b):\n    return a * x + b\n\nx = np.array([0.0, 1.0, 2.0, 3.0, 4.0])\ny = np.array([1.0, 3.0, 5.0, 7.0, 9.0])",
        solution="popt, pcov = optimize.curve_fit(line, x, y)\nanswer = popt",
        hint="curve_fit(model, x, y) returns (popt, pcov). You want popt, which should be [2, 1].",
        tol=1e-5,
    ),
    dict(
        prompt="Fit the exponential model `expo(x, a, k) = a * exp(k * x)` to `(x, y)` with `curve_fit`, using starting values p0 = [1, 0.1]. Assign the fitted `a` to `a_hat` and the fitted `k` to `k_hat`.",
        setup="def expo(x, a, k):\n    return a * np.exp(k * x)\n\nx = np.array([0.0, 1.0, 2.0, 3.0, 4.0])\ny = 2.0 * np.exp(0.5 * x)",
        solution="popt, pcov = optimize.curve_fit(expo, x, y, p0=[1.0, 0.1])\na_hat = popt[0]\nk_hat = popt[1]",
        hint="Pass p0=[1.0, 0.1]; the answer should be close to a=2, k=0.5.",
        answer_var=["a_hat", "k_hat"],
        tol=1e-4,
    ),
    dict(
        prompt="Find the root of h(x) = x³ − 2x − 5 that lies between 2 and 3 with `optimize.brentq`. Assign it to `answer`.",
        setup="def h(x):\n    return x**3 - 2 * x - 5",
        solution="answer = optimize.brentq(h, 2.0, 3.0)",
        hint="brentq(function, a, b) needs a bracket where h(a) and h(b) have opposite signs.",
        tol=1e-6,
    ),
    dict(
        prompt="Find the minimiser of k(x) = x⁴ − 3x² + x on the interval [0, 2] with `optimize.minimize_scalar` using `method='bounded'`. Assign the minimiser to `answer`.",
        setup="def k(x):\n    return x**4 - 3 * x**2 + x",
        solution="answer = optimize.minimize_scalar(k, bounds=(0, 2), method='bounded').x",
        hint="minimize_scalar(k, bounds=(0, 2), method='bounded').x",
        tol=1e-4,
    ),
    dict(
        prompt="Solve this LP with `optimize.linprog` (method 'highs'): minimise 3·x₁ + 2·x₂ subject to x₁ + x₂ ≥ 4, x₁ ≤ 3, x₂ ≤ 3, both non-negative. Remember linprog wants ≤ constraints. Assign the solution array to `x_opt` and the objective value to `obj`.",
        setup="",
        solution="c = [3, 2]\nA_ub = [[-1, -1]]\nb_ub = [-4]\nbounds = [(0, 3), (0, 3)]\nres = optimize.linprog(c, A_ub=A_ub, b_ub=b_ub, bounds=bounds, method='highs')\nx_opt = res.x\nobj = res.fun",
        hint="x1 + x2 >= 4 becomes -x1 - x2 <= -4. Upper limits go into bounds=[(0, 3), (0, 3)].",
        answer_var=["x_opt", "obj"],
        tol=1e-6,
    ),
    dict(
        prompt="The LP in the previous task has an equality version: minimise 3·x₁ + 2·x₂ subject to x₁ + x₂ = 4, 0 ≤ x₁ ≤ 3, 0 ≤ x₂ ≤ 3. Solve it and assign the **dual value (marginal) of the equality constraint** to `answer`.",
        setup="",
        solution="res = optimize.linprog([3, 2], A_eq=[[1, 1]], b_eq=[4], bounds=[(0, 3), (0, 3)], method='highs')\nanswer = float(res.eqlin.marginals[0])",
        hint="Use A_eq / b_eq; the duals are in res.eqlin.marginals. It is the cost of one extra unit of the constraint.",
        tol=1e-6,
    ),
    dict(
        prompt="Given the known points `xp`, `fp`, linearly interpolate the values at `x_new = [0.5, 1.5, 2.5]` with `np.interp`. Assign the array to `answer`.",
        setup="xp = np.array([0.0, 1.0, 2.0, 3.0])\nfp = np.array([0.0, 10.0, 30.0, 60.0])\nx_new = np.array([0.5, 1.5, 2.5])",
        solution="answer = np.interp(x_new, xp, fp)",
        hint="np.interp(x_new, xp, fp) — note the argument order: query points first.",
    ),
    dict(
        prompt="`y` has a gap (NaN) at position 2. Fill it by **linear interpolation** between its neighbours using `interpolate.interp1d` on the non-NaN points (or `np.interp`). Assign the fully filled array to `answer`.",
        setup="t = np.arange(6.0)\ny = np.array([1.0, 2.0, np.nan, 4.0, 5.0, 6.0])",
        solution="ok = ~np.isnan(y)\nf = interpolate.interp1d(t[ok], y[ok])\nanswer = f(t)",
        hint="Build a mask ok = ~np.isnan(y), fit interp1d(t[ok], y[ok]), evaluate at every t.",
    ),
    dict(
        prompt="Remove the linear trend from `z` with `signal.detrend` and assign the detrended array to `answer`.",
        setup="z = np.array([1.0, 3.0, 5.0, 7.0, 9.0, 11.0]) + np.array([0.5, -0.5, 0.5, -0.5, 0.5, -0.5])",
        solution="answer = signal.detrend(z)",
        hint="signal.detrend(z) with the default type='linear'.",
    ),
    dict(
        prompt="Smooth `z` with a centred 3-point moving average using pandas (`rolling(3, center=True).mean()`), keeping NaN at the two ends. Assign the resulting Series to `answer`.",
        setup="z = pd.Series([1.0, 4.0, 2.0, 8.0, 5.0, 9.0])",
        solution="answer = z.rolling(3, center=True).mean()",
        hint="z.rolling(3, center=True).mean()",
    ),
]
