"""Exercises for 02_pandas/00_series_fundamentals.ipynb"""
import numpy as np
import pandas as pd

COMMON_SETUP = """
import numpy as np
import pandas as pd
"""

QUESTIONS = [
    dict(
        prompt="Create a Series with values 5, 10, 15 and index labels 'a', 'b', 'c', named 'load'.",
        setup="",
        solution='answer = pd.Series([5, 10, 15], index=["a", "b", "c"], name="load")',
        hint="pd.Series(values, index=..., name=...)",
        check_fn=lambda got, exp: (isinstance(got, pd.Series) and got.equals(exp) and got.name == "load",
                                   "" if isinstance(got, pd.Series) and got.name == "load" else "check the values, the index and the name"),
    ),
    dict(
        prompt="From `s`, select the values for labels 'c' and 'a', in that order.",
        setup='s = pd.Series([100, 200, 300, 400], index=["a", "b", "c", "d"])',
        solution='answer = s.loc[["c", "a"]]',
        hint="Use .loc with a list of labels; the order of the list is the order of the result.",
    ),
    dict(
        prompt="From `s`, take the last two values **by position**.",
        setup='s = pd.Series([100, 200, 300, 400], index=["a", "b", "c", "d"])',
        solution="answer = s.iloc[-2:]",
        hint="Positional selection is .iloc; negative positions count from the end.",
    ),
    dict(
        prompt="Keep only the values of `s` that are strictly between 150 and 350.",
        setup='s = pd.Series([100, 200, 300, 400], index=["a", "b", "c", "d"])',
        solution="answer = s[(s > 150) & (s < 350)]",
        hint="Two comparisons in parentheses combined with &.",
    ),
    dict(
        prompt="Reindex `s` to the labels ['d', 'e', 'a'] so that the label that does not exist becomes 0 instead of NaN.",
        setup='s = pd.Series([100, 200, 300, 400], index=["a", "b", "c", "d"])',
        solution='answer = s.reindex(["d", "e", "a"], fill_value=0)',
        hint="reindex has a fill_value argument.",
    ),
    dict(
        prompt="Add `a` and `b` so that labels present in only one of them are treated as 0 (no NaN in the result).",
        setup='a = pd.Series([1, 2, 3], index=["x", "y", "z"])\nb = pd.Series([10, 20, 30], index=["y", "z", "w"])',
        solution="answer = a.add(b, fill_value=0)",
        hint="Plain a + b gives NaN for x and w. The method form of + has a fill_value argument.",
    ),
    dict(
        prompt="`temp` uses -999 as a 'missing' marker. Replace every value below -100 with NaN (keep the other values).",
        setup="temp = pd.Series([5.0, -999.0, 7.0, -999.0, 9.0])",
        solution="answer = temp.mask(temp <= -100)",
        hint="mask(condition) sets values to NaN where the condition is True. where() does the opposite.",
    ),
    dict(
        prompt="Build a DataFrame with three columns from `y`: `y` itself, `lag1` (the previous value) and `roll2` (the mean of the two **previous** values, not including the current one).",
        setup="y = pd.Series([10, 20, 30, 40, 50], index=pd.date_range('2023-01-01', periods=5, freq='h'))",
        solution='answer = pd.DataFrame({"y": y, "lag1": y.shift(1), "roll2": y.shift(1).rolling(2).mean()})',
        hint="lag1 is y.shift(1). For roll2, shift first so the window ends at the previous row, then rolling(2).mean().",
    ),
    dict(
        prompt="`region` should be mapped to a zone with the dictionary `zones`. Regions that are **not** in the dictionary must keep their original name.",
        setup='region = pd.Series(["London", "Wales", "North", "Scotland"])\nzones = {"London": "South", "Wales": "West"}',
        solution="answer = region.replace(zones)",
        hint="map(dict) turns unmapped values into NaN; the other lookup method keeps them.",
    ),
    dict(
        prompt="`txt` holds prices as text, with one bad value. Convert it to numbers so the bad value becomes NaN, and assign the number of NaN to `n_bad` and the numeric Series to `answer`.",
        setup='txt = pd.Series(["70.16", "23.19", "missing", "115.79"])',
        solution='answer = pd.to_numeric(txt, errors="coerce")\nn_bad = int(answer.isna().sum())',
        hint="pd.to_numeric(..., errors='coerce'), then .isna().sum().",
        answer_var=["answer", "n_bad"],
    ),
    dict(
        prompt="`d` has a duplicated label 'a'. Keep only the **last** occurrence of each label so the index becomes unique.",
        setup='d = pd.Series([1, 2, 3, 4], index=["a", "a", "b", "c"])',
        solution='answer = d[~d.index.duplicated(keep="last")]',
        hint="index.duplicated(keep='last') marks the earlier duplicates; negate it with ~ and use it as a mask.",
    ),
    dict(
        prompt="On the real data: load `../data/hourly_power_clean.csv`, make a Series of `consumption_mwh` indexed by `time`, and assign the **timestamp** (index label) of the maximum consumption.",
        setup='df = pd.read_csv("../data/hourly_power_clean.csv", parse_dates=["time"])',
        solution='cons = df.set_index("time")["consumption_mwh"]\nanswer = cons.idxmax()',
        hint="set_index('time') then pick the column; idxmax() returns the label of the max, argmax() the position.",
        check_fn=lambda got, exp: (pd.Timestamp(got) == exp, "" if pd.Timestamp(got) == exp else f"expected {exp}"),
    ),
]
