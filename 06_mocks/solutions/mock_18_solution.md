# Mock 18 — A mostly-correct notebook: solution

Story: a competent day-ahead consumption forecast with a 12:00 D-1 decision time, honest
point-in-time features, a chronological split, baselines and residual checks. It contains
**two** real problems. Everything else, including several things that pattern-match to
classic mistakes, is correct.

This mock scores the opposite skill from the others: not crying wolf. In a real review,
flagging correct code as a bug costs credibility, and an interviewer who planted nothing in a
cell will ask "why do you think that is wrong?" and expect a precise answer.

## Real problems

| # | Where | Kind | Why it matters | How to detect | Fix | Interviewer follow-up |
|---|---|---|---|---|---|---|
| 1 | Baselines: `naive = s.shift(24).reindex(test.index)` labelled "naive same hour yesterday" | information mismatch between baseline and model | The model is only allowed data known at 12:00 on D-1. "Same hour yesterday" for hours 12:00-23:00 of D uses D-1 values that are not observed at the decision time. The baseline is stronger than any forecast the desk could actually make, so the model's improvement is understated (44% vs 49% against an honest persistence). Direction aside, baseline and model must play by the same rules. | Write down, for hour 18 of day D, what `shift(24)` returns and whether it is known at 12:00 D-1 | D-1 for hours before 12:00, D-2 otherwise (or D-2 throughout) | "Does the mismatch make the model look better or worse here? Does that change whether it matters?" |
| 2 | Features: `diff_week = lag168 - lag336` alongside `lag168` and `lag336` | exact collinearity | Predictions are unaffected (Ridge handles it), but the coefficient table is uninterpretable: the weight is split arbitrarily (291 / 297 / -15 in the mock; 250 / 337 after dropping it). Any reading of "lag336 matters as much as lag168" is an artefact. | The feature is a linear combination by construction; `np.linalg.matrix_rank` on the feature matrix | Drop it; keep only linearly independent features if coefficients are to be read | "Would this matter if you never printed the coefficients?" |

A third, cosmetic nit: the final table is printed with `.round(1)`, so R² shows as 0.8 / 0.9
while the sentence below quotes 0.948. Not a bug, but the kind of thing the interviewer's
"attention to detail" line is about.

## Red herrings — correct code that looks wrong (11)

| Looks like | Why it is fine |
|---|---|
| `Ridge(alpha=1.0, random_state=0)` | `random_state` is only used by the stochastic solvers (sag/saga). Harmless with the default solver. |
| `s.shift(1).rolling(24).mean()` then `.reindex(frame["decision_time"])` | The window ending at 11:00 D-1 is complete by 12:00 D-1. The `shift(1)` is what makes it so. |
| `StandardScaler` inside `make_pipeline` fitted before the split is defined | The pipeline is fitted on `train` only; the scaler never sees test rows. |
| `pd.get_dummies(..., drop_first=True)` | Correct with a model that has an intercept. Keeping all 24 dummies would be the mistake. |
| `merge_asof(..., direction="backward", left_by="target_time", right_by="forecast_datetime")` on `decision_time` vs `origin_datetime` | This is the honest point-in-time join: the latest forecast issued at or before the decision time, for the target hour. The assert verifies it; horizons are 12-35 h. |
| `frame["temp_fc"].ffill()` | Forward fill only ever uses earlier rows. Here it is a no-op except for the first archive day, which the warm-up `dropna` removes anyway. |
| One `dropna(subset=features + ["y"])` on the combined frame | Keeps X and y aligned; the mistake would be separate `dropna` calls. |
| No gap between train (to 31 Jul 23:00) and test (from 1 Aug 00:00) | The target is at the row's own time and every feature is lagged. No training target overlaps a test feature. A gap is needed when the *target* is shifted into the future relative to the row, which is not the design here. |
| `lag48`, not `lag24` | At 12:00 on D-1 the latest complete day is D-2. lag24 would be the leak. |
| R² 0.948, RMSE ~900 MWh | Consistent with the naive baseline's 0.83 and with a ~29 GWh mean. A suspicious R² would be > 0.99 with a lag-1 feature. |
| Hour dummies from the UTC index | Consistent throughout the notebook. Local-time hours would be a refinement, not a correction. |

## Honest result

| Metric (Aug-Dec 2023) | Mock | Fixed |
|---|---|---|
| Ridge RMSE / R² | 899 / 0.948 | 899 / 0.948 (unchanged: neither issue affects predictions) |
| Baseline used | shift(24), RMSE 1,618 | honest persistence, RMSE 1,777 |
| Improvement claimed | 44% | 49% vs honest persistence; 36% vs same hour last week |
| Coefficients lag168 / lag336 / diff_week | 291 / 297 / -15 | 250 / 337 / (dropped) |

## Scoring

Full marks: raise exactly the two real issues, mention the rounding nit if you like, and
explicitly *clear* the red herrings you looked at ("I checked the scaler is inside the pipeline
so it is fitted on train only"). Each correct item flagged as a bug is a deduction, and a
bigger one if you cannot say precisely what would go wrong. The ideal answer to this notebook
is short: "two things, neither changes the forecast, here is why the rest is fine."
