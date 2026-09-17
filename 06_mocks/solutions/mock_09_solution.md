# Mock 09 — walk-forward evaluation and tuning: solution

Headline claimed by the notebook: **OOS RMSE 549 MWh, best alpha 0.1, 42% improvement from tuning, walk-forward confirms (536 MWh).**
Honest answer: **≈ 954 MWh on the same holdout, tuning changes nothing (−0.5%), the 42% came from two leaked features plus comparing different windows.**

| # | Where | Kind | Why it matters | How to detect | Fix | Likely follow-up |
|---|---|---|---|---|---|---|
| 1 | Features: `feat["lag1"] = y.shift(1)`, `feat["lag6"] = y.shift(6)` | bug (leakage) | The intro says the forecast for hour *t* is issued at *t−24h*. Consumption at *t−1h* and *t−6h* does not exist yet. Every downstream number is contaminated. | Read the problem statement, then list each feature's availability time. Coefficient table: `lag1` = 3460 (standardised) dwarfs everything else. Drop it and RMSE jumps 538 → 954. | Only lags ≥ 24h; rolling windows ending at *t−24h*. | "Which of these features would you actually have at 00:00 the day before? Prove the RMSE drop is not just a worse model." |
| 2 | `TimeSeriesSplit(n_splits=5)` with no `gap` | research choice | With a 24h issuance lag, the last 24 training targets are not yet observed when the first test forecast is issued. | Think about what the forecaster knew at fit time. | `TimeSeriesSplit(n_splits=5, gap=24)`. Effect here is small (1067 → 1079 CV RMSE) but the principle is what is examined. | "Does `gap` matter more for a tree model or for Ridge? What if the horizon were 48h?" |
| 3 | `X = scaler.fit_transform(X_raw)` before any split | bug (leakage) | Test-period means/stds leak into training. Mild for Ridge, disqualifying as a habit. | Scaler fit outside the CV loop / on the full frame. | `make_pipeline(StandardScaler(), Ridge())`, fit inside each fold. | "How big is the effect here, and when would it be large?" |
| 4 | `GridSearchCV(..., cv=KFold(shuffle=True))` then `cross_val_score(..., cv=tscv)` on the **same** full data; markdown says "unbiased" | research choice | Shuffled folds on an autocorrelated hourly series put neighbours in train and test. Then the chosen alpha is scored on data that was used to choose it. | Two different CV schemes in the same notebook; grid fit on `X`, not `X_tr`. | Tune on the training period only with `TimeSeriesSplit(gap=24)`; score once on the holdout. | "What's the difference between selection bias and leakage here?" |
| 5 | `improvement = (rmse_untuned - rmse_tuned) / rmse_untuned` | suspicious result | `rmse_untuned` = v1 features, single 2023-08→12 holdout. `rmse_tuned` = **all** features incl. lag1/lag6, mean of 5 expanding CV folds over 2022–2023. Different features **and** different windows; labelled "improvement from tuning". | The alpha table itself shows all alphas ≤ 10 give identical CV RMSE (549.0) — tuning cannot be worth 42%. `v1_features` excludes `lag1`, `lag6`. | Same features, same holdout: tuned 958 vs untuned 954 → −0.5%. | "If tuning doesn't matter, what did? Decompose the 42%." |
| 6 | Walk-forward: `X.loc[:str(m)]` as training set | bug (peek) | `loc[:"2023-03"]` **includes** March. Every month is trained on itself. | Print `train_X.index.max()` vs `test_X.index.min()` inside the loop. | Train on `index < m.start_time`. Honest pooled 2023 RMSE 989 (vs 536 shown). | "Why are the monthly RMSEs so stable? Is that plausible for a genuinely out-of-sample loop?" |
| 7 | `y_true = y_all.iloc[split:]`, `n = min(len(...))`, `rmse(y_true[:n], preds_all[:n])` | bug (misalignment) | Predictions start 2023-01-01; `y_all.iloc[split:]` starts 2023-08-09. Truncating to a common length compares January predictions with August actuals → RMSE 6910. | A number 13× the fold RMSEs is not "boundary effects". `len(preds_all)` = 8760 vs `len(y_true)` = 3466. | Keep predictions as a Series indexed by time; `y_all.loc[preds.index]`. | "You saw 6910 and 536 in adjacent lines. What would you check first?" |
| 8 | "the mean of monthly RMSEs is the representative number" | research choice | Averaging fold RMSEs weights months equally regardless of size and hides which months fail. Pool the squared errors. | — | Pooled RMSE over all OOS hours (989 vs mean-of-months 989 here; can differ a lot with uneven folds). | "When do these two numbers diverge?" |
| 9 | `pd.get_dummies(feat["hour"])` (24 columns) + Ridge intercept; "intercept (baseline load)" | research choice | 24 dummies + intercept are exactly collinear. Ridge resolves it by penalty, so the intercept and per-hour coefficients are not identified — "baseline load 29 282" means nothing. | Count dummy columns; `X.sum(axis=1)` over the dummies is constant 1. | `drop_first=True`; interpret the intercept as the reference hour's load. | "Why does OLS not complain? Would it with `LinearRegression`?" |
| 10 | Alpha table: "small alphas have the best train fit … keep the grid-search choice" | research choice | Choosing by train fit always picks the least regularised model. | Train RMSE is monotone in alpha by construction. | Pick by out-of-sample CV; honest CV prefers alpha 100 (marginal). | "What would you expect the table to look like if regularisation mattered?" |
| 11 | No naive baseline anywhere | research choice | For a 24h horizon, "same hour yesterday" (1629) and "same hour last week" (1424) are the bar. The honest model (954) clears it; the mock never shows it. | Search the notebook for `shift(24)` used as a prediction. | Add naive lag24/lag168 rows to the results. | "How much of the 954 is the model and how much is the calendar?" |
| — | `Ridge(random_state=0)` | **not a problem** | Only used by stochastic solvers (`sag`/`saga`); harmless here. Do not flag it as a bug — say it is irrelevant. | — | — | "Why is that there?" — "It does nothing for the default solver." |

## Honest result (from `mock_09_fixed.ipynb`)

| Model (holdout 2023-08-09 → 2023-12-31, 3 466 h) | RMSE MWh |
|---|---|
| naive same hour yesterday (lag24) | 1 629 |
| naive same hour last week (lag168) | 1 424 |
| Ridge, honest features, alpha = 1 | 954 |
| Ridge, honest features, tuned alpha = 100 | 958 |
| Ridge **with leaked lag1/lag6** (what the mock had) | 538 |
| Honest walk-forward, pooled over 2023 | 989 |

Mock claims vs reality: OOS RMSE 549 → 954; improvement from tuning 42% → −0.5%; walk-forward 536 → 989; aggregated walk-forward 6910 → 989.

Note for the interviewer conversation: honest CV RMSE (≈ 1 080–1 100) is worse than the holdout (954) because expanding `TimeSeriesSplit` folds train on very little data early in 2022. That is a property of the evaluation design, not evidence of a bug.

## Scoring

Finding 1, 5, 6 and 7 (the two leaks, the false comparison, the 6910 misalignment) is the pass mark; 3–4 and 11 show good habits; 2, 8, 9, 10 are bonus. Flagging `random_state` as a bug is a small negative.
