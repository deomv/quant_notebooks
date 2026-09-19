"""Exercises for 01_numpy/02_indexing_broadcasting_vectorisation.ipynb"""
import numpy as np
import pandas as pd

COMMON_SETUP = """
import numpy as np
import pandas as pd
"""

QUESTIONS = [
    dict(
        prompt="From the 2-D array `m`, select the element in row 1, column 2.",
        setup="m = np.array([[10, 11, 12],\n              [20, 21, 22],\n              [30, 31, 32]])",
        solution="answer = m[1, 2]",
        hint="m[row, col] with 0-based positions.",
    ),
    dict(
        prompt="From `m`, select the whole second column (positions 12, 22, 32 → 11, 21, 31) as a 1-D array.",
        setup="m = np.array([[10, 11, 12],\n              [20, 21, 22],\n              [30, 31, 32]])",
        solution="answer = m[:, 1]",
        hint="`:` means all rows; then the column position.",
    ),
    dict(
        prompt="From `v`, pick the elements at positions 4, 0 and 2, in that order (fancy indexing).",
        setup="v = np.array([5, 6, 7, 8, 9])",
        solution="answer = v[[4, 0, 2]]",
        hint="Index with a list of positions.",
    ),
    dict(
        prompt="Keep only the values of `v` that are greater than 6 and even.",
        setup="v = np.array([5, 6, 7, 8, 9, 10])",
        solution="answer = v[(v > 6) & (v % 2 == 0)]",
        hint="Two conditions in parentheses combined with &.",
    ),
    dict(
        prompt="Replace every negative value of `p` with 0 and keep the others (use `np.where` or `np.clip`).",
        setup="p = np.array([50.0, -5.0, 80.0, -20.0])",
        solution="answer = np.where(p < 0, 0, p)",
        hint="np.where(condition, value_if_true, value_if_false).",
    ),
    dict(
        prompt="Label each temperature in `t`: 'cold' if below 10, 'hot' if above 25, otherwise 'mild'. The result is an array of strings.",
        setup="t = np.array([2.0, 15.0, 30.0, 9.9])",
        solution='answer = np.select([t < 10, t > 25], ["cold", "hot"], default="mild")',
        hint="np.select([cond1, cond2], [choice1, choice2], default=...).",
        check_fn=lambda got, exp: (
            list(map(str, np.asarray(got))) == list(map(str, exp)),
            "" if list(map(str, np.asarray(got))) == list(map(str, exp)) else f"expected {list(exp)}",
        ),
    ),
    dict(
        prompt="Subtract from every row of `m` the vector `row_offset` (broadcasting). The result has the same shape as `m`.",
        setup="m = np.array([[10, 11, 12],\n              [20, 21, 22]])\nrow_offset = np.array([10, 10, 10])",
        solution="answer = m - row_offset",
        hint="A (2, 3) array minus a (3,) array broadcasts across rows automatically.",
    ),
    dict(
        prompt="Subtract from each row of `m` its **own** first element, i.e. `col0 = [10, 20]`. Reshape `col0` so it broadcasts down the columns; the result is [[0, 1, 2], [0, 1, 2]].",
        setup="m = np.array([[10, 11, 12],\n              [20, 21, 22]])\ncol0 = np.array([10, 20])",
        solution="answer = m - col0[:, np.newaxis]",
        hint="A (2,) array does not broadcast against (2, 3); make it (2, 1) with [:, np.newaxis] or reshape(-1, 1).",
    ),
    dict(
        prompt="Compute the sum of each **column** of `m` (result has 3 numbers) and the sum of each **row** (result has 2 numbers).",
        setup="m = np.array([[10, 11, 12],\n              [20, 21, 22]])",
        solution="col_sums = m.sum(axis=0)\nrow_sums = m.sum(axis=1)",
        hint="axis=0 collapses the rows (one number per column); axis=1 collapses the columns.",
        answer_var=["col_sums", "row_sums"],
    ),
    dict(
        prompt="Without a Python loop, compute `y = 3 * x**2 + 1` for the array `x`.",
        setup="x = np.array([0, 1, 2, 3])",
        solution="answer = 3 * x**2 + 1",
        hint="Arithmetic on an array applies element-wise.",
    ),
    dict(
        prompt="`flows` are hourly MWh. Compute the running total (cumulative sum) and the hour-to-hour change (difference; 3 numbers for 4 inputs).",
        setup="flows = np.array([5, 7, 4, 10])",
        solution="running = np.cumsum(flows)\nchange = np.diff(flows)",
        hint="np.cumsum and np.diff.",
        answer_var=["running", "change"],
    ),
    dict(
        prompt="Bin the temperatures `t` with edges `edges = [0, 10, 20]`: assign to each temperature the number of edges that are ≤ it (so 5 → 1, 10 → 2 because 10 counts as reached, 25 → 3, -3 → 0). Use `np.searchsorted`.",
        setup="edges = np.array([0, 10, 20])\nt = np.array([5, 10, 25, -3])",
        solution='answer = np.searchsorted(edges, t, side="right")',
        hint="searchsorted(edges, values, side='right') counts how many edges are <= each value.",
    ),
    dict(
        prompt="Get the positions that would sort `v` ascending, and the position of its maximum.",
        setup="v = np.array([30, 10, 20, 40])",
        solution="order = np.argsort(v)\npos_max = int(np.argmax(v))",
        hint="np.argsort and np.argmax return positions, not values.",
        answer_var=["order", "pos_max"],
    ),
    dict(
        prompt="Find the distinct values of `codes` and how many times each occurs (two arrays, distinct values sorted ascending).",
        setup='codes = np.array(["b", "a", "b", "c", "a", "b"])',
        solution="values, counts = np.unique(codes, return_counts=True)",
        hint="np.unique(..., return_counts=True) returns two arrays.",
        answer_var=["values", "counts"],
        check_fn=None,
    ),
    dict(
        prompt="`a` and `b` are equal up to floating-point noise. Assign to `naive` the result of `a == b` (an array of booleans) and to `answer` a single boolean saying whether they are all close.",
        setup="a = np.array([0.1 + 0.2, 1.0])\nb = np.array([0.3, 1.0])",
        solution="naive = a == b\nanswer = bool(np.allclose(a, b))",
        hint="np.allclose for the single boolean; plain == compares exactly.",
        answer_var=["naive", "answer"],
    ),
    dict(
        prompt="On the real data: `cons` is the consumption array. Compute the mean consumption over the hours where `hour == 18` and over the hours where `hour == 3` (two numbers).",
        setup='df = pd.read_csv("../data/hourly_power_clean.csv", parse_dates=["time"])\ncons = df["consumption_mwh"].to_numpy()\nhour = df["time"].dt.hour.to_numpy()',
        solution="mean_18 = cons[hour == 18].mean()\nmean_3 = cons[hour == 3].mean()",
        hint="Boolean mask on `hour`, then .mean() of the selected consumption values.",
        answer_var=["mean_18", "mean_3"],
    ),
]
