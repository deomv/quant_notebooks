"""Exercises for 02_pandas/01_load_inspect.ipynb"""
import io
import numpy as np
import pandas as pd

COMMON_SETUP = """
import io
import numpy as np
import pandas as pd

CSV = '''time,price,region,flag
2023-01-01 00:00,70.16,GB,1
2023-01-01 01:00,missing,GB,1
2023-01-01 02:00,23.19,GB,0
2023-01-01 02:00,23.19,GB,0
2023-01-01 04:00,115.79,GB,1
'''
"""

QUESTIONS = [
    dict(
        prompt="Read the CSV text in `CSV` (wrap it with `io.StringIO`) so that the `time` column is parsed as a datetime. Assign the DataFrame.",
        setup="",
        solution='answer = pd.read_csv(io.StringIO(CSV), parse_dates=["time"])',
        hint="pd.read_csv(io.StringIO(CSV), parse_dates=[...])",
        check_fn=lambda got, exp: (
            isinstance(got, pd.DataFrame) and pd.api.types.is_datetime64_any_dtype(got["time"]) and got.equals(exp),
            "" if isinstance(got, pd.DataFrame) and pd.api.types.is_datetime64_any_dtype(got.get("time", pd.Series(dtype=object)))
            else "the time column is not a datetime64 dtype (did you pass parse_dates?)"),
    ),
    dict(
        prompt="Read `CSV` again, but keep only the columns `time` and `price` (in that order) and make `price` numeric by telling `read_csv` that the string `missing` means NaN.",
        setup="",
        solution='answer = pd.read_csv(io.StringIO(CSV), usecols=["time", "price"], na_values=["missing"])',
        hint="usecols=[...] and na_values=[...] are both read_csv arguments.",
        check_fn=lambda got, exp: (
            isinstance(got, pd.DataFrame) and list(got.columns) == ["time", "price"] and got["price"].dtype.kind == "f" and got["price"].isna().sum() == 1,
            "expected columns ['time', 'price'] with a float price column containing one NaN"),
    ),
    dict(
        prompt="`df` was read with no options. Assign the dtype of each column (the `dtypes` Series).",
        setup="df = pd.read_csv(io.StringIO(CSV))",
        solution="answer = df.dtypes",
        hint="df.dtypes is a Series indexed by column name.",
        check_fn=lambda got, exp: (isinstance(got, pd.Series) and list(got.index) == list(exp.index) and [str(x) for x in got] == [str(x) for x in exp],
                                   "expected the df.dtypes Series (time and price are object, region object, flag int64)"),
    ),
    dict(
        prompt="How many rows of `df` are exact duplicates of an earlier row? Assign the integer.",
        setup="df = pd.read_csv(io.StringIO(CSV))",
        solution="answer = int(df.duplicated().sum())",
        hint="df.duplicated() marks the second and later copies; sum the booleans.",
    ),
    dict(
        prompt="Assign the **share** (fraction between 0 and 1) of missing values per column of `df`, as a Series indexed by column name.",
        setup='df = pd.read_csv(io.StringIO(CSV), na_values=["missing"])',
        solution="answer = df.isna().mean()",
        hint="isna() gives booleans; the mean of booleans is the share of True.",
    ),
    dict(
        prompt="Assign the number of distinct values per column of `df` as a Series.",
        setup='df = pd.read_csv(io.StringIO(CSV), na_values=["missing"])',
        solution="answer = df.nunique()",
        hint="nunique() works on the whole frame and returns one number per column.",
    ),
    dict(
        prompt="Assign a **sorted list** of the names of the columns of `df` that are constant (hold a single distinct value).",
        setup='df = pd.read_csv(io.StringIO(CSV), na_values=["missing"])',
        solution="answer = sorted(df.columns[df.nunique() == 1].tolist())",
        hint="nunique() == 1 is a boolean Series over columns; use it to index df.columns.",
    ),
    dict(
        prompt="Parse `time`, sort it, and count how often each gap between consecutive timestamps occurs. Assign the `value_counts()` Series (index = Timedelta, values = counts).",
        setup="df = pd.read_csv(io.StringIO(CSV))",
        solution='t = pd.to_datetime(df["time"]).sort_values()\nanswer = t.diff().value_counts()',
        hint="pd.to_datetime, then .sort_values(), then .diff(), then .value_counts().",
    ),
    dict(
        prompt="Assign a **sorted list** of the column names of `df` whose dtype is object (text).",
        setup="df = pd.read_csv(io.StringIO(CSV))",
        solution='answer = sorted(df.select_dtypes(include="object").columns.tolist())',
        hint="select_dtypes(include='object').columns",
    ),
    dict(
        prompt="Real data: read `../data/hourly_power_raw.csv` with no options and assign its shape as a tuple `(rows, columns)`.",
        setup="",
        solution='raw = pd.read_csv("../data/hourly_power_raw.csv")\nanswer = raw.shape',
        hint="df.shape is already a tuple.",
    ),
    dict(
        prompt="Real data: in `raw`, how many rows have the literal text `missing` in `price_eur_mwh`? Assign the integer.",
        setup='raw = pd.read_csv("../data/hourly_power_raw.csv")',
        solution='answer = int((raw["price_eur_mwh"] == "missing").sum())',
        hint="Compare the column to the string and sum the booleans.",
    ),
    dict(
        prompt="Real data: parse `raw['time']` as UTC timestamps and assign the earliest to `t_min` and the latest to `t_max`.",
        setup='raw = pd.read_csv("../data/hourly_power_raw.csv")',
        solution='t = pd.to_datetime(raw["time"], utc=True)\nt_min = t.min()\nt_max = t.max()',
        hint="pd.to_datetime(..., utc=True), then .min() and .max().",
        answer_var=["t_min", "t_max"],
        check_fn=lambda got, exp: (pd.Timestamp(got) == exp, "" if pd.Timestamp(got) == exp else f"expected {exp}"),
    ),
]
