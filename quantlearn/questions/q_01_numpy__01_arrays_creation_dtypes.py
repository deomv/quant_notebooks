"""Exercises for 01_numpy/01_arrays_creation_dtypes.ipynb"""
import numpy as np
import pandas as pd

COMMON_SETUP = """
import numpy as np
import pandas as pd
"""


def _copy_check(got, exp):
    if not isinstance(got, np.ndarray):
        return False, f"expected an ndarray, got {type(got).__name__}"
    if got.shape != exp.shape or not np.array_equal(got, exp):
        return False, "values or shape differ from expected"
    if got.base is not None and got.base.shape == (6,) and got.base.dtype == float:
        # a view of the original 6-element base array
        return False, "this is still a view of `a` (writing to it would change `a`); make a copy"
    return True, ""


QUESTIONS = [
    dict(
        prompt="Create a 1-D array with the values 2, 4, 6, 8 (in that order).",
        setup="",
        solution="answer = np.array([2, 4, 6, 8])",
        hint="np.array([...]) or np.arange(start, stop, step).",
    ),
    dict(
        prompt="Create a 2 × 3 array filled with zeros of dtype `float64`.",
        setup="",
        solution="answer = np.zeros((2, 3))",
        hint="np.zeros takes a shape tuple; the default dtype is already float64.",
        check_fn=lambda got, exp: (
            isinstance(got, np.ndarray) and got.shape == (2, 3) and got.dtype == np.float64 and np.all(got == 0),
            "" if isinstance(got, np.ndarray) and got.shape == (2, 3) and got.dtype == np.float64 else "check the shape (2, 3) and dtype float64",
        ),
    ),
    dict(
        prompt="Create 5 evenly spaced numbers from 0 to 1 inclusive (0, 0.25, 0.5, 0.75, 1).",
        setup="",
        solution="answer = np.linspace(0, 1, 5)",
        hint="linspace(start, stop, how_many) includes both ends; arange does not include the stop.",
    ),
    dict(
        prompt="`a` is a 1-D array of 6 floats. Assign its shape to `shape`, its number of dimensions to `ndim` and its number of elements to `size`.",
        setup="a = np.array([1.0, 2.0, 3.0, 4.0, 5.0, 6.0])",
        solution="shape = a.shape\nndim = a.ndim\nsize = a.size",
        hint="Three attributes: .shape (a tuple), .ndim, .size.",
        answer_var=["shape", "ndim", "size"],
    ),
    dict(
        prompt="Convert `prices` (strings) to a float array.",
        setup='prices = np.array(["70.5", "23.0", "115.2"])',
        solution="answer = prices.astype(float)",
        hint=".astype(float) parses numeric strings.",
        check_fn=lambda got, exp: (
            isinstance(got, np.ndarray) and got.dtype.kind == "f" and np.allclose(got, exp),
            "" if isinstance(got, np.ndarray) and got.dtype.kind == "f" else "the result must be a float array (check .dtype)",
        ),
    ),
    dict(
        prompt="`x` holds integers. Compute `x / 2` so the result keeps fractions (e.g. 1.5), and assign the result's dtype kind (`'f'` for float, `'i'` for int) to `kind` and the array to `answer`.",
        setup="x = np.array([1, 3, 5])",
        solution="answer = x / 2\nkind = answer.dtype.kind",
        hint="`/` always gives floats in NumPy; `//` would floor to integers. `.dtype.kind` is a one-letter code.",
        answer_var=["answer", "kind"],
    ),
    dict(
        prompt="Reshape the 6-element array `a` into 2 rows and 3 columns.",
        setup="a = np.array([1.0, 2.0, 3.0, 4.0, 5.0, 6.0])",
        solution="answer = a.reshape(2, 3)",
        hint="reshape(rows, cols); the row-major order fills the first row first.",
    ),
    dict(
        prompt="Reshape `a` into 3 rows and 2 columns and then transpose it. The result must be a 2 × 3 array whose first row is [1, 3, 5].",
        setup="a = np.array([1.0, 2.0, 3.0, 4.0, 5.0, 6.0])",
        solution="answer = a.reshape(3, 2).T",
        hint="reshape(3, 2) then .T (or np.transpose).",
    ),
    dict(
        prompt="`view = a[::2]` shares memory with `a`. Assign to `answer` a boolean saying whether `view` and `a` share memory, and to `shares_after_copy` the same test for `a[::2].copy()`.",
        setup="a = np.array([1.0, 2.0, 3.0, 4.0, 5.0, 6.0])\nview = a[::2]",
        solution="answer = np.shares_memory(view, a)\nshares_after_copy = np.shares_memory(a[::2].copy(), a)",
        hint="np.shares_memory(x, y) returns True/False. Slices are views; .copy() is not.",
        answer_var=["answer", "shares_after_copy"],
    ),
    dict(
        prompt="Take every second element of `a` as an **independent copy** (so that later changes to `a` do not affect it). The values must be [1, 3, 5].",
        setup="a = np.array([1.0, 2.0, 3.0, 4.0, 5.0, 6.0])",
        solution="answer = a[::2].copy()",
        hint="A slice is a view; add .copy() to break the link.",
        check_fn=_copy_check,
    ),
    dict(
        prompt="Turn the 1-D array `v` (shape (3,)) into a column vector of shape (3, 1).",
        setup="v = np.array([1, 2, 3])",
        solution="answer = v[:, np.newaxis]",
        hint="v[:, np.newaxis] or v.reshape(3, 1) or v.reshape(-1, 1).",
        check_fn=lambda got, exp: (
            isinstance(got, np.ndarray) and got.shape == (3, 1) and np.array_equal(got, exp),
            "" if isinstance(got, np.ndarray) and got.shape == (3, 1) else f"shape must be (3, 1), got {getattr(got, 'shape', None)}",
        ),
    ),
    dict(
        prompt="Stack `r1` and `r2` as two rows of one 2 × 3 array (`r1` on top).",
        setup="r1 = np.array([1, 2, 3])\nr2 = np.array([4, 5, 6])",
        solution="answer = np.vstack([r1, r2])",
        hint="np.vstack([...]) stacks vertically; np.column_stack would put them side by side.",
    ),
    dict(
        prompt="Put `r1` and `r2` side by side as two **columns** of one 3 × 2 array (`r1` is the first column).",
        setup="r1 = np.array([1, 2, 3])\nr2 = np.array([4, 5, 6])",
        solution="answer = np.column_stack([r1, r2])",
        hint="np.column_stack([...]), or np.stack([...], axis=1).",
    ),
    dict(
        prompt="On the real data: load `../data/hourly_power_clean.csv` and take the `consumption_mwh` column as a NumPy array. Assign the array to `answer` and its shape to `shape`.",
        setup='df = pd.read_csv("../data/hourly_power_clean.csv")',
        solution='answer = df["consumption_mwh"].to_numpy()\nshape = answer.shape',
        hint=".to_numpy() on the column; .shape of a 1-D array is (n,).",
        answer_var=["answer", "shape"],
    ),
]
