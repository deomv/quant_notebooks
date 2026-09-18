# quant_notebooks — Pandas / NumPy / sklearn cheat sheets for quant research interviews

Executed Jupyter notebooks, one topic per notebook, organised in the order you would
use them in a 45–60 minute "here is a dataset, investigate it" interview round.
Every notebook runs top-to-bottom on the synthetic energy datasets in `data/`, so
outputs are already visible and you can re-run any cell to experiment.

Only NumPy, Pandas, SciPy and scikit-learn are used (plus a few matplotlib plots).

## Layout

| Folder | Notebooks | Use it for |
|---|---|---|
| `INTERVIEW_QA.md` | conceptual questions and short spoken answers, each pointing at the notebook with the mechanics | read the night before |
| `01_numpy/` | arrays & dtypes · indexing/broadcasting/vectorisation · stats/linalg/random | raw array fluency, views vs copies, axis semantics, ddof, OLS by hand |
| `02_pandas/` | Series fundamentals · load & inspect · clean/missing/dtypes · select/filter/transform · groupby/pivot · merge/join/concat · time series · pitfalls & debugging · debugging toolkit · half-hourly settlement & panel data | the bulk of the interview: reading, cleaning, reshaping, joining, time handling, and the catalogue of things that go wrong |
| `03_scipy/` | stats · optimize/interpolate/signal · sampling distributions, standard errors & bootstrap · hypothesis tests, power & pitfalls | hypothesis tests, distributions, curve fitting, small constrained optimisation |
| `04_sklearn/` | regression workflow · preprocessing & pipelines · cross-validation for time series · diagnostics & interpretation · load-profile clustering & PCA | boring-but-correct modelling, honest evaluation, critiquing a result |
| `05_time_series_research/` | features/targets/leakage · evaluation & baselines · end-to-end template · classical time-series stats · anomaly detection & data quality · renewables & price market EDA · statistical traps | the Fuse-style flow: define the problem, build honest features, beat the naive baseline, explain the errors |
| `07_optimisation/` | LP dispatch · battery arbitrage · retail hedging & CVaR · curve fitting & calibration · imbalance newsvendor & quantile decisions · unit commitment MILP & scenarios | the optimisation problems an energy retailer-trader faces, with scipy.optimize (`linprog`, `milp`, `minimize`, `least_squares`) and sklearn |
| `06_mocks/` | 18 deliberately broken research notebooks + `solutions/` | the interview simulation: open one blind, 45 minutes, find the planted bugs, leakage and questionable choices, then check the answer key |
| `data/` | `make_data.py` + generated CSVs | shared synthetic hourly power data, customer meters, weather forecasts with origin timestamps |

Every cheat-sheet notebook is written to be read without running it: each concept
starts with a 3–6 element example, prints the input, applies one operation per cell,
prints the result, says what to notice, and only then shows the same thing on the real
data. Mocks are the exception: they are meant to look like a colleague's real research.

Each notebook opens with "What's in here" and uses two kinds of callout:

- **Pitfall:** a mistake that looks like working code.
- **Interview check:** the question an interviewer is likely to ask about that step.

## Suggested reading order

Once the cheat sheets feel familiar, spend most of your time in `06_mocks/` (see its README for the rules).

1. `02_pandas/00` (Series) → `01` → `02` → `06` → `07` (inspect, clean, time series, pitfalls) — this is 60% of the value.
2. `05_time_series_research/01` (leakage) and `03` (end-to-end template with talk track).
3. `04_sklearn/01`, `03`, `04` (workflow, time-series CV, diagnostics).
4. `07_optimisation/01`, `02`, `05` (dispatch LP, battery, imbalance decisions) — the Fuse-specific modelling.
5. `02_pandas/08` (debugging toolkit), `02_pandas/09` (half-hourly settlement & panels), `05_time_series_research/05` (data quality) — the "attention to detail" material.
6. `03_scipy/03`–`04` (standard errors, bootstrap, hypothesis tests, power) and `05_time_series_research/07` (statistical traps) — for the "is this result real?" questions.
7. `01_numpy/*`, `02_pandas/03`–`05`, `03_scipy/01`–`02`, `04_sklearn/02`, `04_sklearn/05`, `05_time_series_research/04`, `06`, `07_optimisation/03`, `04`, `06` as reference.

## Data

`data/make_data.py` generates everything deterministically (seed 42). Regenerate with:

```bash
python data/make_data.py
```

| File | What it is | Traps built in |
|---|---|---|
| `hourly_power_clean.csv` | 2022–2023 hourly UTC: consumption (MWh), temperature, wind, solar, day-ahead price | none — the tidy reference |
| `hourly_power_raw.csv` | same data "as received" | unsorted, 15 duplicate rows, a whole missing day (2022-03-27) plus scattered gaps, NaNs, `-999` temperature sentinels, price stored as text with `"missing"`, a constant `region` column |
| `meters.csv` | 300 customer meters: region, tariff, type, annual kWh estimate, signup date, solar flag | NaNs in tariff/estimate, inconsistent casing in region |
| `meter_readings_daily.csv` | daily kWh per meter, long format, 2023 | gaps; 200 rows for a meter that does not exist in `meters.csv` |
| `weather_forecasts.csv` | temperature forecasts issued at 00:00 and 12:00 UTC for the next 48 h | `origin_datetime` vs `forecast_datetime` — the point-in-time / leakage exercise; error grows with horizon; warm bias |
| `meter_halfhourly_2023.csv.gz` | UK-style half-hourly settlement data for 20 meters: local `settlement_date`, `settlement_period` 1–48, kWh | DST days with 46 and 50 periods; a meter stuck for a week; a meter in Wh for a month; a missing fortnight; scattered gaps; duplicates; solar export negatives |

## Setup

```bash
pip install -r requirements.txt
```

In PyCharm: open this folder as the project, pick the interpreter that has the
packages above, and open any `.ipynb`. Notebooks assume the working directory is
their own folder (paths are `../data/...`), which is PyCharm's and Jupyter's default.

Note: the topic folders are prefixed with numbers (`01_numpy`, not `numpy`) so
that they can never shadow the real `numpy` / `pandas` / `sklearn` packages when
the project root is on `sys.path`.
