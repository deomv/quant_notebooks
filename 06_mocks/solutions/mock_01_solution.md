# Mock 01 — Day-ahead consumption forecast: solution

Headline in the broken notebook: **R² = 0.949, RMSE 960 MWh, "explains 95% of next-day variance, ready for production".**
Honest result: **R² = 0.861, RMSE 1473 MWh, vs naive "same hour today" R² = 0.830, RMSE 1630 MWh.**

The notebook does not forecast tomorrow at all. A misalignment between `X` and `y` turned it into a
nowcast of the *current* hour, and several look-ahead features would have leaked even without that.

| # | Where | Kind | Why it matters | How to detect | Fix | Interviewer follow-up |
|---|---|---|---|---|---|---|
| 1 | Load — `pd.to_datetime(df["time"])`, no `utc=True` | research choice | Index is tz-naive but the data are UTC; later called "local hour". GB local time is UTC+1 for 7 months, so the evening peak sits at 17h UTC in winter and 18h UTC in summer. Calendar features are blurred. | `df.index.tz` is None; compare `groupby(hour)` peak in Jan vs Jul. | `utc=True`, then `index.tz_convert("Europe/London")` for hour/dow. | "What time zone are the timestamps in? How would DST show up in your features?" |
| 2 | Load — `sort_index()` but no de-duplication | bug | 15 duplicated timestamps. Every `shift(k)` after a duplicate is off by one row. | `df.index.duplicated().sum()` → 15; `df.index.is_unique` False. | `drop_duplicates(subset="time")` before `set_index`; assert `is_unique`. | "Are the duplicates identical rows or conflicting readings? Which one do you keep?" |
| 3 | Load — no reindex to a full hourly grid | bug (subtle) | 78 hours are missing (a whole day on 2022-03-27 plus scattered gaps). `shift(24)` shifts 24 *rows*, which is not 24 hours across a gap. | `df.index.to_series().diff().value_counts()` shows 2h and 25h gaps. | `reindex(pd.date_range(..., freq="h"))`, then decide how to handle the NaN rows. | "What is one observation here, and is the sampling interval constant?" |
| 4 | `describe()` shows `temp_c` min = -999, never handled | bug | -999 is a sentinel. `ffill` in the next cell propagates it into real hours; the scaler stretches around it; the coefficient on temp collapses to 0.0. | The `describe()` output is right there: mean 6.3, std 61, min -999. | `df.loc[df.temp_c <= -100, "temp_c"] = np.nan` **before** filling. | "Why is the temperature coefficient exactly zero in a heating-driven system?" |
| 5 | Clean — `pd.to_numeric(...).fillna(0)` on price | research choice | 178 hours get price 0. Zero is a valid price in this market (negative prices exist), so the model sees invented observations. | `(df.price_eur_mwh == 0).sum()` vs the number of genuine zeros in the clean file (none). | Leave NaN, `ffill(limit=3)` for short gaps, drop the rest. | "What is the difference between a missing price and a zero price for this model?" |
| 6 | Clean — `temp_c.ffill()` without a limit | research choice | Fills across the 25h gap and across sentinels. On a still-unsorted frame it would fill from a random row. | Count NaNs before/after; check the longest run filled. | `interpolate(limit=3)` or leave NaN. | "How long a gap are you comfortable filling, and would you fill the target the same way?" |
| 7 | Features — `lag168 = c.shift(-168)` | bug (sign typo) | Negative shift is the future: consumption one week *ahead*. Pure leakage. | Look at the `tail()` output: `lag168` is NaN at the end of the sample, which only happens for negative shifts. | `shift(168)`. | "How do you know that feature is not available at prediction time?" |
| 8 | Features — `rolling(48, center=True)` | bug (leak) | A centred window uses 23 future hours. Any `center=True` feature is look-ahead. | Same tail test: NaN at the end. `df.smooth48.corr(df.target)` is suspiciously high. | Drop it, or `shift(1).rolling(48)`. | "Would this still be a leak if the horizon were 48 hours?" |
| 9 | Features — `rolling(24).mean()` without `shift(1)` | research choice | Includes the current hour. Harmless for a 24h target, fatal for a 1h target; the habit matters. | Compare `rolling(24).mean()` and `shift(1).rolling(24).mean()` on one row. | `c.shift(1).rolling(24).mean()`. | "Is that a leak here? When would it become one?" |
| 10 | Target — `X.dropna()`, `y.dropna()` separately, then `iloc[:n]` | bug (the big one) | X loses its first 24 rows (rolling NaN) and its last 168 (negative shift); y loses its last 24. After truncation `y.iloc[i]` is `c` at the **same hour** as `X.iloc[i]` (both are `shift(-24)` of row *i* vs row *i+24*). The model predicts the current hour from `lag1` and `mean24`, which is why `lag1` dominates. | Print `X.index[0]` and `y.index[0]`: 2022-01-02 vs 2022-01-01. `(X.index == y.index).all()` is False. The `print(len(X), len(y))` cell already shows different lengths. | Build one frame `df[features + ["target"]].dropna()`. | "If lengths matched, would that have proved alignment?" |
| 11 | Split — `StandardScaler().fit_transform(X)` before the split | research choice | Test-set mean and variance leak into training. Small effect on OLS, but it is the pattern the interviewer is looking for. | The scaler is fitted before `train_test_split` appears. | Fit on train, transform test; or use a `Pipeline`. | "Does it matter for a linear model? For a tree?" |
| 12 | Split — `train_test_split` with default `shuffle=True` | research choice | Hourly rows are autocorrelated; neighbours of each test hour are in the training set. Test error is optimistic. | `shuffle` not set; test indices are random. | Chronological split (last 20%), or `TimeSeriesSplit`. | "Why is shuffling wrong here but fine for the meters classification problem?" |
| 13 | Results — no baseline | suspicious result | 0.949 is judged in a vacuum. Naive "same hour today" already scores 0.83 for this target. | Ask "what would a trivial forecast score?" and compute it. | Report the naive R²/RMSE alongside. | "Your model beats naive by 10%. Is that a lot for day-ahead load?" |
| 14 | Results — `lag1` is the top coefficient for a 24h horizon | suspicious result | For a next-day target the same-hour lags should dominate. `lag1` dominating means the target is effectively the current hour. | Read the coefficient table with the horizon in mind. | Fixing #10 moves the weight to `lag24`/`lag168`. | "What coefficient pattern would you expect for this horizon?" |
| 15 | Calendar — hour × dow profile computed on the duplicated frame | bug (minor) | Duplicated rows are double-weighted in the group means; small here, but it is the same missing de-dup as #2. | Same as #2. | De-dup first. | — |

## Honest result

| | broken | fixed |
|---|---|---|
| Held-out R² | 0.949 | 0.861 |
| Held-out RMSE (MWh) | 960 | 1473 |
| Naive "same hour today" R² / RMSE | not computed | 0.830 / 1630 |
| Top coefficient | `lag1` (2120) | `lag168` (1793), `lag24` (1561), `temp_c` (-1413) |
| Usable rows | 17,265 (misaligned) | 15,833 (aligned, gaps dropped) |

Residuals in the fixed model are still autocorrelated and larger in the evening peak (hours 16–18 mean residual +500 to +870 MWh). The next honest step is a weather *forecast* feature, not more lags.

## Scoring

Found 10+ of 15, including #10 and at least one of #7/#8: strong.  
Found #10 or spotted "no baseline / lag1 dominant" and explained *why* R² is too good: that alone is the point of the exercise.  
Found only #4 and #12: keep practising the "what is one observation, is it aligned, what was known at t" questions.
