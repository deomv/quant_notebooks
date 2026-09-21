# Interview Q&A — the conceptual questions behind the notebooks

Short answers you can say aloud, each with the notebook that shows the mechanics.
Grouped by the order they tend to come up in a "here is a notebook, investigate it" round.

---

## 1. Before touching the model: the data

**What is one row in this dataset?**
Say it explicitly: one meter-hour, one settlement period, one day per customer. Everything else (duplicates, sampling interval, which columns are keys) follows from that. → `02_pandas/01`

**How do you check the timestamps are what you think?**
Parsed dtype (`datetime64`, not object), timezone, sorted, unique, constant interval via `diff().value_counts()`, min/max, gaps against a full `date_range` grid. → `02_pandas/01`, `02_pandas/06`

**Why store UTC and derive local time?**
UTC is monotonic and unique; local time repeats an hour in autumn and skips one in spring. Features like "hour of day" should be local, the index should be UTC. UK settlement periods run 1–48 (46/50 on DST days). → `02_pandas/06`, `02_pandas/09`

**A column of numbers is object dtype. What happened, and what do you do?**
A sentinel string ("missing", "N/A", "-") is in there. `pd.to_numeric(errors="coerce")`, then count how many became NaN and look at them. Never `fillna(0)` blindly. → `02_pandas/02`

**How do you handle missing hours in the target?**
Reindex to the full grid so gaps are visible, then decide: drop for training, never forward-fill the target (you would be teaching the model a stale value), and make lag features NaN across the gap rather than pretending it is contiguous. → `02_pandas/02`, `05_time_series_research/03`

**Your merge went from 17,520 rows to 35,003. What happened?**
The right-hand key was not unique (e.g. four forecasts per target hour). Check key cardinality before merging, use `validate="one_to_one"` or `"many_to_one"`, and compare row counts before and after. → `02_pandas/05`

**What does `merge_asof` do and when is it the right join?**
Joins each left row to the latest right row at or before it (with `direction="backward"`). It is the right join whenever "what was known at time t" matters: forecasts, prices, reference data that updates. Both sides must be sorted. → `02_pandas/05`, `05_time_series_research/01`

**Why does `sum()` of an all-NaN column return 0?**
pandas skips NaN by default and an empty sum is 0. A missing day therefore looks like zero demand. Use `min_count=1` or check completeness before aggregating. → `02_pandas/07`, mock 05

**`size()` vs `count()` after a groupby?**
`size` counts rows including NaN; `count` counts non-null values per column. After a merge, `size` is "number of readings", not "number of meters". → `02_pandas/04`, mock 06

---

## 2. Time series and leakage

**Define leakage in one sentence.**
Any information in a feature that would not have been available at the moment the prediction is made.

**Why `shift(1)` before `rolling(24).mean()`?**
Without it the window ends at the current row and includes the current target value. With it the window ends at t−1. The R² difference can be small for long windows and enormous for short ones, which is why it is easy to miss. → `05_time_series_research/01`

**Why not shuffle the train/test split?**
Adjacent hours are nearly identical, so a shuffled split puts near-copies of test rows in training. It measures interpolation, not forecasting. Use a chronological split or `TimeSeriesSplit`, with a gap equal to the horizon. → `04_sklearn/03`

**What is the `gap` in TimeSeriesSplit for?**
If the target is h steps ahead, the last h training rows contain targets that overlap the first test rows. `gap=h` removes that overlap (an embargo). → `04_sklearn/03`

**You have observed temperature at the target hour as a feature. Is that allowed?**
Only for a nowcast. For a day-ahead forecast you need the temperature forecast that was available at decision time, joined on `origin_datetime <= decision_time` and `forecast_datetime == target_time`. → `05_time_series_research/01`, mock 03

**You fit StandardScaler on the whole dataset before splitting. Does it matter?**
It leaks test-set means and variances into training. Usually a small effect, but it is the kind of thing that signals whether you think about information flow. Put the scaler in a Pipeline. → `04_sklearn/02`

**What about target encoding or group means?**
Same issue, worse: a group mean computed on the full sample includes the test targets. Compute on train only, or out-of-fold. → `05_time_series_research/01`

**`X.dropna()` and `y.dropna()` separately, then `.values`: what goes wrong?**
The two lose different rows and positional alignment breaks; the model is trained on shifted pairs. Build one frame, `dropna` once. → `02_pandas/00`, `05_time_series_research/01`

**Why must you sort before `shift`, `rolling`, `merge_asof`?**
They operate on row order, not on time. On an unsorted frame the "lag" is a random other row. Also dedupe: with duplicate timestamps, `shift(24)` is off by one row after each duplicate. → `02_pandas/07`

**Long panel of many meters: what changes?**
Every `shift`/`rolling`/`diff` must be inside `groupby("meter_id")`, sorted by meter then time, or values leak across entities. Pooled R² is inflated by between-meter scale differences; report per-meter metrics too. → `02_pandas/09`, mock 15

---

## 3. Modelling and evaluation

**R² is 0.99. What do you check first?**
(1) Which feature has a coefficient near 1 and is it derived from the target? (2) Is the split chronological? (3) Does it beat the naive baseline by a plausible margin? (4) Train vs test gap. (5) Was the target actually shifted the right way? (6) Duplicates in test. → `04_sklearn/04`, `02_pandas/08`

**What baselines must a model beat?**
Persistence (last value), same hour yesterday (`shift(24)`), same hour last week (`shift(168)`), and a seasonal mean by hour × weekday from train. If the model does not beat `shift(168)` it has learned nothing useful. → `05_time_series_research/02`

**RMSE vs MAE vs MAPE?**
RMSE penalises large errors (matches squared-cost settlement); MAE is robust; MAPE explodes near zero and is undefined for negative prices, and optimising it biases forecasts low. → `05_time_series_research/02`

**Can out-of-sample R² be negative?**
Yes: R² compares to the test mean, and a model worse than "predict the mean" scores below zero. In-sample R² cannot be negative with an intercept. → `05_time_series_research/02`

**Model B beat model A by 3% RMSE. Is it better?**
Check per-month wins, whether one period drives it, the standard error of the difference with autocorrelation taken into account (Diebold–Mariano style), and whether B was chosen from many candidates on the same test set. → `05_time_series_research/02`, `03_scipy/04`

**Why does residual autocorrelation matter?**
It means there is structure the model misses (a lag or calendar term), the standard errors are wrong, and the model may be barely beating persistence. → `04_sklearn/04`, `05_time_series_research/04`

**Ridge vs OLS: when and why?**
When features are collinear (lag 1 and a shifted rolling mean), OLS coefficients are unstable across periods; Ridge trades a little bias for stability. Choose alpha with time-series CV, not shuffled KFold. → `04_sklearn/03`, `04_sklearn/04`

**Hour as a number or one-hot?**
Numeric hour forces a monotone effect; one-hot lets each hour have its own level. Sine/cosine encoding is a compromise. Show the RMSE difference. → `04_sklearn/01`

**What is a walk-forward evaluation?**
Refit periodically (e.g. monthly) using only past data, predict the next block, concatenate the out-of-sample predictions. It mimics production and shows regime deterioration. → `04_sklearn/03`, `05_time_series_research/02`

**How would you produce prediction intervals without a parametric model?**
Empirical quantiles of train residuals, checked for coverage on test, or quantile regression at the needed quantiles. → `05_time_series_research/02`, `07_optimisation/05`

---

## 4. Statistics

**Standard deviation vs standard error?**
Four distinct things: the population sd σ (true spread of individual observations), the sample sd s (your estimate of σ from n points), the variance of the sample mean σ²/n (how much x̄ would move across repeated samples), and its square root the standard error σ/√n, estimated by s/√n. The sample sd does not shrink with n; the standard error does. → `03_scipy/00`, `03_scipy/03`

**What does a 95% confidence interval mean?**
Under repetition of the procedure, 95% of such intervals contain the true value. It is a statement about the procedure, not a 95% probability for this particular interval. → `03_scipy/03`

**What is a p-value?**
The probability of data at least as extreme as observed if the null hypothesis were true. Not the probability the null is true. → `03_scipy/04`

**When does the bootstrap fail?**
Dependent data (autocorrelation: use block bootstrap), extremes (max, tail quantiles), very small n, and when the statistic is not smooth. → `03_scipy/03`

**Is r = 0.4 on 17,520 hours significant?**
Nominally yes with iid SE, but the effective sample size under lag-1 autocorrelation ρ is roughly n(1−ρ)/(1+ρ). With ρ ≈ 0.95 that is about 450 effective observations. Also ask whether both series trend (spurious correlation). → `03_scipy/03`, `05_time_series_research/04`

**You tested 20 features and one has p = 0.01. Finding?**
Expected number of false positives at 5% is one. Use Bonferroni/Benjamini–Hochberg, or better, an out-of-sample confirmation. → `03_scipy/04`

**Weekday vs weekend consumption: which test?**
Not a t-test on 17k hourly rows (dependence). Aggregate to daily or weekly means first, or use a block permutation test; then report the effect size in MWh with a CI. → `03_scipy/04`

**Simpson's paradox in one sentence.**
A relationship that holds within every group reverses in the aggregate because group sizes differ; always stratify by the obvious confounder (customer type, region, season). → `05_time_series_research/07`, mock 17

**Regression to the mean?**
Selecting the extreme cases on a noisy measure guarantees they look less extreme next time, with no intervention. Compare against a control group or check a placebo period. → `05_time_series_research/07`

---

## 5. Energy and optimisation

**What is a merit order and how does an LP recover it?**
Generators sorted by marginal cost; demand is met from cheapest up. The dispatch LP's dual on the demand constraint is the marginal cost of the last unit, i.e. the price. → `07_optimisation/01`

**Why does a battery LP charge and discharge in the same hour?**
With negative prices or zero degradation cost it is optimal to waste energy. Add a throughput cost or a binary mode variable (MILP). → `07_optimisation/02`

**Why is the hedge ratio for a retail load book above 1?**
Load and price are positively correlated (cold → high load → high price), so a flat baseload hedge under-covers the expensive hours. Peak blocks reduce the residual. → `07_optimisation/03`, mock 12

**What is the newsvendor critical fractile?**
Buy the quantile c_u/(c_u + c_o) of the demand distribution, where c_u is the cost of being short (imbalance buy premium) and c_o the cost of being long. Asymmetric costs mean the mean forecast is the wrong volume. → `07_optimisation/05`, mock 13

**How would you value a better weather forecast?**
Run the same decision policy with actual temperature (upper bound), the forecast, and no temperature; the cost differences in EUR are the value of information. → `07_optimisation/05`

**MW vs MWh?**
MW is power (a rate), MWh is energy. Hourly MW averages equal MWh; half-hourly kWh × 2 = kW. Summing power over periods without the time step is the classic unit error. → `07_optimisation/02`, mock 11, mock 14

**Why is capacity factor 61% for onshore wind impossible?**
Typical onshore is 25–35%. A number like that means the power curve is not capped at rated output or the units are wrong. → mock 16

---

## 6. How to talk during the round

- Say what you are checking before you check it: "Before I look at the metric I want to know what one row is and whether timestamps are unique."
- When a number looks good, say so and say why you do not believe it yet.
- Name the fix and its expected effect, then verify: "If the rolling mean leaks, removing it should lower R² noticeably; let me check."
- Distinguish bugs (wrong output) from choices (defensible but questionable) from results you would not present.
- Do not cry wolf: `random_state` on Ridge, `drop_first=True` with an intercept, a scaler inside a Pipeline are fine. → mock 18
- End with "what I would investigate next" rather than a verdict. → `05_time_series_research/03`
