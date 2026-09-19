"""Exercises for 02_pandas/10_interactive_debugging_pdb.ipynb

Every task hands you a traceback (or a function to step through). Solve it the way you
would in a live session: `%debug` / `breakpoint()` and the pdb commands, or the
traceback-walking code from section 7. The checker only looks at the value you assign.
"""
import numpy as np
import pandas as pd

COMMON_SETUP = """
import sys
import traceback
import numpy as np
import pandas as pd

def _frames(tb):
    # helper: list of (function name, locals dict) from a traceback, innermost last
    out = []
    while tb is not None:
        out.append((tb.tb_frame.f_code.co_name, tb.tb_frame.f_locals))
        tb = tb.tb_next
    return out
"""

INTRO = """
Tip: after a failing cell you can type `%debug` in a new cell and use `w`, `p`, `u`,
`d`, `q`. The setup of each task captures the traceback in `tb` (and the exception in
`exc`) so the tasks also work without an interactive prompt: `_frames(tb)` returns
`[(function_name, locals), ...]`, innermost frame last.
"""

_PIPE = '''
def convert(value, unit):
    scale = {"kwh": 1, "mwh": 1000}[unit]
    return value * scale / (scale - 1000)      # fails for mwh

def load_row(rec):
    unit = rec["unit"]
    return convert(rec["value"], unit)

def load_all(records):
    done = []
    for rec in records:
        done.append(load_row(rec))
    return done

records = [{"value": 3, "unit": "kwh"}, {"value": 7, "unit": "kwh"}, {"value": 2, "unit": "mwh"}, {"value": 9, "unit": "kwh"}]
try:
    load_all(records)
except ZeroDivisionError as e:
    exc, tb = e, sys.exc_info()[2]
'''

QUESTIONS = [
    dict(
        prompt="`load_all(records)` raised. In which **function** was the exception raised? Assign its name as a string.",
        setup=_PIPE,
        solution='answer = traceback.extract_tb(tb)[-1].name',
        hint="`%debug` lands in the raising frame; the `>` line names the function. In code: traceback.extract_tb(tb)[-1].name",
    ),
    dict(
        prompt="How many frames does the traceback have (the cell itself counts as one)? Assign the integer.",
        setup=_PIPE,
        solution='answer = len(traceback.extract_tb(tb))',
        hint="`w` lists them; count the lines starting with a file path. In code: len(traceback.extract_tb(tb)).",
    ),
    dict(
        prompt="What was the value of `scale` in the frame that raised? Assign it.",
        setup=_PIPE,
        solution='answer = _frames(tb)[-1][1]["scale"]',
        hint="`%debug` then `p scale`. In code: the last element of _frames(tb) holds the innermost locals.",
    ),
    dict(
        prompt="Move one frame up (the caller). What was `unit` there? Assign the string.",
        setup=_PIPE,
        solution='answer = _frames(tb)[-2][1]["unit"]',
        hint="`u` once, then `p unit`. In code: _frames(tb)[-2].",
    ),
    dict(
        prompt="Two frames up, in `load_all`: how many records had already been processed successfully when it failed? Assign the integer (look at `done`).",
        setup=_PIPE,
        solution='answer = len(_frames(tb)[-3][1]["done"])',
        hint="`u`, `u`, `p done`, then count. In code: _frames(tb)[-3][1]['done'].",
    ),
    dict(
        prompt="Which record (as a dict) was being processed when it failed? Assign the dict.",
        setup=_PIPE,
        solution='answer = _frames(tb)[-3][1]["rec"]',
        hint="In the `load_all` frame the loop variable is `rec`.",
    ),
    dict(
        prompt="What is the exception's class name? Assign it as a string (e.g. 'KeyError').",
        setup=_PIPE,
        solution='answer = type(exc).__name__',
        hint="type(exc).__name__; after a real failure IPython keeps it in sys.last_type.",
    ),
    dict(
        prompt="`df.apply(per_row, axis=1)` failed. Which row **label** (index value) was being processed? Assign it.",
        setup='''
df = pd.DataFrame({"kwh": [1.0, 2.0, 3.0, 4.0], "days": [1, 2, 0, 4]}, index=["m1", "m2", "m3", "m4"])

def per_row(row):
    return row["kwh"] / row["days"] if row["days"] != 0 else 1 / 0

try:
    df.apply(per_row, axis=1)
except ZeroDivisionError as e:
    exc, tb = e, sys.exc_info()[2]
''',
        solution='answer = [loc for name, loc in _frames(tb) if name == "per_row"][0]["row"].name',
        hint="`%debug` lands in per_row; `p row.name` is the label. In code: find the frame named 'per_row' in _frames(tb).",
    ),
    dict(
        prompt="How many frames of that `apply` traceback come from pandas' own source files (file path contains 'pandas')? Assign the integer. This is what `w` scrolls past before reaching your code.",
        setup='''
df = pd.DataFrame({"kwh": [1.0, 2.0, 3.0, 4.0], "days": [1, 2, 0, 4]}, index=["m1", "m2", "m3", "m4"])

def per_row(row):
    return row["kwh"] / row["days"] if row["days"] != 0 else 1 / 0

try:
    df.apply(per_row, axis=1)
except ZeroDivisionError as e:
    exc, tb = e, sys.exc_info()[2]
''',
        solution='answer = sum("pandas" in rec.filename for rec in traceback.extract_tb(tb))',
        hint="traceback.extract_tb(tb) gives records with a .filename attribute.",
    ),
    dict(
        prompt="`df.groupby('site').apply(check)` failed. Which group key was being processed? Assign it as a string.",
        setup='''
df = pd.DataFrame({"site": ["A", "A", "B", "B", "C"], "kwh": [1.0, 2.0, np.nan, 4.0, 5.0]})

def check(g):
    if g["kwh"].isna().any():
        raise ValueError("NaN in group")
    return g["kwh"].sum()

try:
    df.groupby("site").apply(check)
except ValueError as e:
    exc, tb = e, sys.exc_info()[2]
''',
        solution='answer = [loc for name, loc in _frames(tb) if name == "check"][0]["g"].name',
        hint="In the `check` frame, `p g.name` gives the group key.",
    ),
    dict(
        prompt="Step through `accumulate([3, 1, 4, 1, 5])` (put `breakpoint()` at the top, or use a conditional breakpoint `b <line>, i == 3`). What is the value of `acc` at the **start** of the iteration where `i == 3`? Assign the integer.",
        setup='''
def accumulate(values):
    acc = 1
    for i, v in enumerate(values):
        acc = acc * 2 + v
    return acc
''',
        solution='''acc = 1
for i, v in enumerate([3, 1, 4, 1, 5]):
    if i == 3:
        answer = acc
        break
    acc = acc * 2 + v''',
        hint="Stop when i == 3 before the update line runs; `p acc`. (acc goes 1 -> 5 -> 11 -> 26 -> ...)",
    ),
    dict(
        prompt="`finalize(5)` returns a value computed from a local `k`. Using the debugger (`!k = 10` before the return, then `r`) or by reading the code, what would `finalize(5)` return if `k` were 10 instead? Assign the number.",
        setup='''
def finalize(a):
    k = a * 3
    result = (a + k) * 2
    return result
''',
        solution='answer = (5 + 10) * 2',
        hint="Stop after `k = a * 3`, run `!k = 10`, then `n` and `p result` (or `r` to see the return value).",
    ),
    dict(
        prompt="With `pd.options.mode.chained_assignment = 'raise'`, does `df[df['a'] > 1]['b'] = 0` raise an exception? Assign True or False (test it inside try/except, and put the option back to 'warn' afterwards).",
        setup='''
df = pd.DataFrame({"a": [1, 2, 3], "b": [10, 20, 30]})
''',
        solution='''pd.options.mode.chained_assignment = "raise"
try:
    df[df["a"] > 1]["b"] = 0
    answer = False
except Exception:
    answer = True
pd.options.mode.chained_assignment = "warn"''',
        hint="The silent no-op becomes a SettingWithCopyError when the option is 'raise'.",
    ),
]
