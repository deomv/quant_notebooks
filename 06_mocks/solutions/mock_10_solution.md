# Mock 10 — model validation report: solution

Claims in the notebook: **OOS R² 0.96, RMSE ≈ 840 MWh, no bias, ≈ 2% relative error, stable across all twelve months, outliers were data errors, temperature is the dominant driver.**
Reality: **the "OOS" numbers are in-sample; the test set double-counts October; there is a statistically significant bias and a ±1 000 MWh hour-of-day bias; only five months were validated; the "outliers" are real evening/morning ramps; temperature is a minor driver; adding hour dummies cuts RMSE from 820 to 549.**

| # | Where | Kind | Why it matters | How to detect | Fix | Likely follow-up |
|---|---|---|---|---|---|---|
| 1 | "## Out-of-sample performance": `r2_score(train["consumption"], pred_train)` | bug (mislabel) | The headline R² 0.9609 / RMSE 841.6 are **train** numbers. The real test numbers (0.9571 / 819.6) are computed in the next cell and never used. | Read what goes into the metric call. Variables named `r2_test`, `rmse_test` exist but never reach the summary. | Report test metrics under the OOS heading. | "Which numbers in the summary are in-sample and which out-of-sample?" |
| 2 | `test = pd.concat([test_a, test_b])` with `loc["2023-08":"2023-10"]` and `loc["2023-10":"2023-12"]` | bug | October is in both slices: 4 416 rows for 3 672 hours, 744 duplicated timestamps. Every test metric double-weights October; the "outlier" count (44) and CI are wrong too. | `test.index.duplicated().sum()` → 744; `len(test)` is not 153 × 24. | `frame.loc["2023-08":]`; `assert test.index.is_unique`. | "How would you have caught this without knowing the calendar?" |
| 3 | `r2_score(pred_test, test["consumption"])` | bug | R² is not symmetric in its arguments: 0.9556 (swapped) vs 0.9571 (correct). Small here, can be large. | Signature is `(y_true, y_pred)`. | Swap the arguments. | "Why isn't R² symmetric? What about RMSE?" |
| 4 | "## Bias": `err.mean()` = 53 MWh → "indistinguishable from zero" | suspicious result | With n ≈ 3 700 and sd ≈ 820 the standard error is ≈ 13, so t ≈ 3.6 — significant. And the overall mean hides hour-of-day biases of −935 (19:00) to +1 250 MWh (16:00) because `hour` enters as a straight line. | Compute `err.std()/np.sqrt(len(err))`; `err.groupby(err.index.hour).mean()`. | Report s.e./t-stat and the by-hour table; add hour dummies (see 10). | "Is a 50 MWh bias economically material? For whom?" |
| 5 | `np.sort(actual)` vs `np.sort(pred)`, "corr 0.9997, tracks closely" | bug (misleading) | Sorting both columns independently destroys the pairing; you get a quantile–quantile comparison. Any two series with similar marginal distributions correlate ≈ 1. Paired correlation is 0.978. | Two `np.sort` calls; the table's index (0, 441, 882, …) is not time. | Compare paired values; use the time index. | "What does the sorted comparison actually tell you? Is it useless?" |
| 6 | "## Relative error": `mape_price` = 20.6% (prices near or below zero) | research choice | MAPE is undefined/explosive when the denominator crosses zero. The price side-model belongs in a different report. | The number itself; `ptest["price"].min()` < 0. | Report consumption MAPE (2.25%) only; use MAE or scaled errors for prices. | "Which error metric would you use for prices, and why?" |
| 7 | Summary: `"MAPE (%)": mape_price`, `"Price model MAPE (%)": mape_cons` | bug (mislabel) | Labels swapped, so the consumption model appears to have 20.6% error and the conclusion "about 2%" points at the price row. | The summary contradicts the printed cell above it. | Fix the dict; keep one metric per key. | "You have two MAPE rows and neither matches conclusion 3. Which is right?" |
| 8 | "## Seasonality": `pd.concat([pred_train, pred_test])` then groupby month | research choice | 7 of 12 months are in-sample, so "stable across all twelve months" is not an out-of-sample statement. | `all_pred` is built from `pred_train`. | Month table on test months only (Aug 774 → Dec 841); state that winter/spring are unvalidated. | "Which months would you worry about most for this model?" |
| 9 | "## Outliers": drop `err.abs() > quantile(0.99)` from **test**, re-report | research choice (serious) | Selecting test hours on the outcome; "metering/data errors" is asserted, never shown. The worst hours are 16:00–19:00 and 00:00 ramps with lag-1 jumps of ±2 500–3 200 MWh — real load behaviour the linear-hour model cannot follow. | Look at the worst rows: `hour`, `consumption - lag1`. No data-quality evidence is presented. | Keep all hours; investigate; if you must exclude, exclude on an ex-ante rule and report both numbers. | "Show me one of those 'data errors'." |
| 10 | "## Residual autocorrelation": 0.21 "expected, not a concern" | research choice | Autocorrelated residuals + hour-of-day bias = missing structure. Adding hour dummies: RMSE 820 → 549, autocorr 0.21 → −0.006. Also the naive persistence baseline (1 522) is never shown. | The by-hour bias table; a quick refit. | Hour-of-day one-hot; report naive lag1/lag24 baselines. | "What would a residual autocorrelation of 0.7 have meant? Of 0?" |
| 11 | "## Confidence interval": `1.96 * se2.std() / sqrt(n)` | research choice | Treats 3 672 autocorrelated (and, in the mock, duplicated) hours as iid. Also computed on the trimmed errors. | — | Daily block bootstrap: [797, 844] vs iid [800, 839] on the full test set — modest here, but the method is wrong. | "When would the iid CI be badly wrong?" |
| 12 | "## Feature importance": raw Ridge coefficients, "temperature is the dominant driver" | research choice | −24.8 MWh/°C vs 1.11 MWh/MWh are in different units. Per 1 sd: lag1 +4 724, lag2 −2 175, lag168 +878, lag24 +560, **temp −171**, hour +16. | Units of each coefficient. | Standardise features or multiply by feature sd. | "Why might Ridge coefficients on collinear lags (lag1 = +1.11, lag2 = −0.51) be hard to interpret even after scaling?" |
| 13 | Summary dict: `"RMSE (MWh)"` and `"RMSE (MWh) "` (trailing space) | bug | Two different RMSEs (in-sample 841.6, trimmed test 791.8) appear under visually identical labels. | The table shows two "RMSE (MWh)" rows. | One key per metric, explicit labels. | "Which of the two RMSE rows is the one in conclusion 1?" |

## Honest result (from `mock_10_fixed.ipynb`, test = 2023-08-01 → 2023-12-31, 3 672 h)

| Metric | Mock | Honest |
|---|---|---|
| R² | 0.9609 (train) | 0.9571 (test) |
| RMSE MWh | 841.6 (train) / 791.8 (trimmed test) | 819.6 |
| naive lag1 RMSE | not shown | 1 522 |
| RMSE with hour dummies | not shown | 548.7 |
| MAPE consumption | 2.26 (labelled "price model") | 2.25 |
| Mean error | 53.2, "≈ 0" | 48.9, s.e. 13.5, t = 3.6 |
| Worst hour-of-day bias | not shown | +1 250 MWh at 16:00, −935 at 19:00 |
| Months validated | "all twelve" | Aug–Dec only |
| Residual autocorr | 0.21 dismissed | 0.207 → −0.006 after hour dummies |
| Temperature effect per 1 sd | "dominant" | −171 MWh (lag1: +4 724) |

## Scoring

Finding 1, 2, 5 and 9 (in-sample-as-OOS, duplicated October, sorted comparison, post-hoc trimming) is the pass mark. 4 and 10 (bias by hour → missing hour structure → RMSE 820 → 549) is what turns a review into research. 3, 6, 7, 8, 11, 12, 13 are the attention-to-detail items the recruiter email is hinting at.
