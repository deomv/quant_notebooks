# Mock 08 — NumPy statistics notes: solution

Story: a "quick statistical pass" over two years of hourly data using mostly NumPy,
concluding that demand does not respond to price (positive correlation, price "leads"),
that the weekend effect is smaller than ops claim, and that a two-variable linear model
explains 96% of daily demand. All three conclusions are wrong.

The single most important planted problem is the **view mutation** in the "Daily profile"
section: `midnight = cons[::24]; midnight -= midnight.mean()` overwrites every midnight
value inside `cons` itself. Every statistic computed after that cell (annual GWh, weekend
effect, correlations, the regression, the bootstrap) runs on corrupted data. The clue is on
screen: hour 0 of the profile table is `-0.0`.

## Planted problems

| # | Where | Kind | Why it matters | How to detect | Fix | Interviewer follow-up |
|---|---|---|---|---|---|---|
| 1 | Sanity checks: `np.std(cons)` vs `df.consumption_mwh.std()`, "differ in the last digit, rounding" | bug in reasoning | NumPy defaults to `ddof=0`, pandas to `ddof=1`. The difference is systematic (ratio exactly sqrt(n/(n-1)) = 1.0000285), not rounding. Harmless at n = 17,520; a 5% error at n = 10. The "identical" column is also computed after rounding to one decimal, so it only flags one of two rows. | Compute the ratio; read the docs | `np.std(x, ddof=1)` | "Which one is the unbiased estimator and why does it matter for small groups?" |
| 2 | Temperature: `temp_clean[temp_clean < -5] = np.nan` then `np.percentile(temp_clean, 95)` | bug + research choice | (a) -6C in a British winter is real, and those 41 hours are exactly the hours that drive peak heating demand; deleting them biases everything downstream. (b) `np.mean`/`np.percentile` propagate NaN, so the table prints `NaN` and the author moves on. | The output literally says `NaN`; `(temp < -5).sum()` | `np.nanpercentile`, and do not mask plausible values | "Is -6C an error? What would a sensor error look like?" |
| 3 | Daily profile: `midnight = cons[::24]; midnight -= midnight.mean()` | bug (view mutation) | Basic slicing returns a view; in-place subtraction writes through to `cons`. Every midnight value becomes a deviation (~ -2,500 .. +2,500 instead of ~25,000). All later results are contaminated. | Profile table shows hour 0 = `-0.0`; `cons[0]` after the cell; `np.shares_memory(midnight, cons)` | `.copy()` before modifying; or never mutate in place | "Which NumPy operations return views and which return copies?" |
| 4 | `peak_prob = profile / profile.sum()` labelled "probability of peak" | research choice | A normalised mean profile is a share of daily energy, not a probability that the daily peak falls in that hour. The real distribution is 96% at hour 18 (fixed notebook); the table shows 5%. | Ask what a probability of 0.052 at 18:00 would mean | Count the `argmax` hour per day | "What fraction of days peak at 18:00?" |
| 5 | Annual energy: `astype(int) // 1000` | bug | Floors every hour to whole GWh, losing up to 0.999 GWh per row, ~4,380 GWh a year (1.7%). Then the corrupted midnight values lose more. | Compare with `consumption.sum() / 1000` | Float division, then sum | "How much energy is in the rounding error per year?" |
| 6a | Weekend effect: `dow >= 6` | bug | Monday = 0, so `>= 6` is Sunday only; Saturday is counted as a weekday. Effect shrinks from -2,082 to -1,750 MWh/h, and the author concludes ops overstated it. | `pd.Series(dow).value_counts()`; `n = 2520 / 15000` printed in the output is a 1:6 ratio, not 2:5 | `dow >= 5` | "Which day is 0 in pandas?" |
| 6b | `stats.ttest_ind(weekend, weekday)` on hourly rows | research choice | Hours are not independent (lag-1 autocorrelation 0.935). The t-statistic is inflated ~5.5x; p-values are meaningless. | Compute `np.corrcoef(cons[1:], cons[:-1])` | Test on daily means, or block bootstrap | "How many independent observations do you really have?" |
| 7 | Price and demand: `stats.pearsonr(price, cons)` on levels, "price causes consumption" | suspicious result | Both series share the hour-of-day cycle, the heating season and the 2022 gas regime; the level correlation is common seasonality. p-value assumes iid. Causality is backwards anyway: day-ahead price is set from forecast demand. | Correlate hour-demeaned or differenced series (0.58 -> 0.39 / 0.37) | Control for shared structure; never read causality off a correlation | "What else moves both price and demand at 6pm in January?" |
| 8 | `np.corrcoef(price[1:], cons[:-1])` described as "corr(consumption_t, price_t-1)" | bug | `price[1:]` is price at t+1 and `cons[:-1]` is consumption at t. It measures consumption leading price, the opposite of the text. | Write down the index of the first element of each slice | `np.corrcoef(price[:-1], cons[1:])` for price leading | "Which series is shifted forward here?" |
| 9 | Regression: `X = daily[["temp_c", "price_eur_mwh"]]` with no intercept column | bug | Forces the fit through the origin; coefficients absorb the mean (temp coefficient +346 instead of -284). | Coefficient signs disagree with physics (demand falls with temperature) | Add `np.ones` column | "What sign should the temperature coefficient have in winter-dominated data?" |
| 10 | `ss_tot = (y ** 2).sum()` | bug | Uncentred total sum of squares. With a mean of 29,000, ss_tot is huge and R² comes out 0.955 whatever the model does (0.999 on the intact data). Centred R² is 0.76. | R² near 1 with a two-variable model on daily demand is too good; compute `1 - ss_res / ((y-y.mean())**2).sum()` | Centre y | "Can R² be near 1 while RMSE is 6,000 MWh/h?" |
| 11 | `beta = inv(X.T@X) @ X.T @ y[:, None]` gives shape (2,1); `pred = X @ beta` is (n,1); `resid = y - pred` | bug (broadcasting) | (n,) minus (n,1) broadcasts to an (n,n) matrix of all pairwise differences. Its mean is meaningless and its sd/RMSE (6,348) is roughly the spread of y itself, not the model error (1,086). | `resid.shape` is `(730, 730)`; RMSE is inconsistent with R² | Keep everything 1-D; assert shapes | "What is the shape of `resid`?" |
| 12 | `np.mean(beta.ravel() == beta_lstsq)` printed as "0% identical" | bug in reasoning | Two solvers agree to 1e-9 but never bit-for-bit; `==` on floats is the wrong test. | Print the absolute difference | `np.allclose` | "When is float equality appropriate?" |
| 13 | Bootstrap: `np.random.randint` without a seed; iid resampling of hours | research choice | Not reproducible, and iid resampling ignores autocorrelation so the CI is far too narrow (width 0.026). A weekly block bootstrap on hour-demeaned data gives [0.31, 0.47]. | Rerun: the interval moves; think about dependence | `default_rng(seed)`; block bootstrap | "Why is the CI so narrow with 17k points that are nearly copies of their neighbours?" |
| 14 | Regression uses `inv(X.T @ X)` | research choice (minor) | Numerically worse than `np.linalg.solve` or `lstsq`; fine here, but a habit interviewers notice. | — | `np.linalg.solve` / `lstsq` | — |

## Honest result

| Statistic | Broken notebook | Fixed notebook |
|---|---|---|
| Annual GWh 2022 / 2023 | 243,946 / 242,329 | 257,609 / 255,995 |
| Weekend effect (MWh/h) | -1,750 (Sunday only, corrupted data) | -2,082 (Sat+Sun), matches the ops claim |
| Temperature p95 | NaN | 20.5C |
| corr(price, consumption) | 0.400 (corrupted levels) | 0.576 levels; 0.389 hour-demeaned; 0.373 differenced |
| 95% CI for that correlation | [0.387, 0.413] iid | [0.313, 0.468] weekly-block, demeaned |
| Lead-lag | "price leads" (actually cons_t vs price_t+1) | corr(price_t-1, cons_t) 0.52 vs corr(price_t+1, cons_t) 0.54: no evidence price leads |
| Daily model coefficients | temp +346, price +240, no intercept | const 30,715, temp -284, price +14.4 |
| Daily model R² | 0.955 (uncentred) | 0.761 (centred) |
| Daily model RMSE | 6,348 (from an (n,n) matrix) | 1,086 |
| "Peak hour probability" | 0.052 at 18:00 | 96% of days peak at 18:00 |

## Scoring

Spotting the `-0.0` at hour 0 and tracing it back to the view mutation is the key find;
so is refusing the R² = 0.955 / RMSE = 6,348 pair as internally inconsistent. Those two
plus the NaN percentile, the Sunday-only weekend and the wrong-column lead-lag is a strong
pass. ddof, uncentred R², the iid bootstrap and the causality story are what separates a
good candidate from a very good one.
