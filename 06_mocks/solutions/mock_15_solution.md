# Mock 15 — Panel forecast leakage: solution

Notebook: `mock_15_panel_forecast_leakage.ipynb`. Fixed version: `mock_15_fixed.ipynb`.
12 planted problems. Kinds: **bug** = wrong output, **choice** = runs but methodologically wrong,
**suspicious** = a result you should refuse to accept.

The headline does *not* collapse when you fix the notebook (0.79 → 0.80 pooled kWh R²). That is the
point of this mock: the number was never the right number. The half-hourly noise caps any model near
R² 0.5 within a meter, the pooled kWh figure is inflated by meter size, and the model does not beat a
naive baseline on the three meters that carry most of the energy.

| # | Where | Kind | Why it matters | How to detect | Fix | Likely follow-up |
|---|---|---|---|---|---|---|
| 1 | "Features" — `df["kwh"].shift(48)` on the long frame, no `groupby("meter_id")` | bug | The frame is sorted by meter then time, so the first 48 rows of each meter take the *previous meter's* last day as their lag; the same for `lag336` (336 rows) and the target (last 48 rows of each meter are the next meter's first day). Small here (20 boundaries) but it is a different meter's data leaking in, and with a different sort order it would be catastrophic. | `df.groupby("meter_id").apply(lambda g: g.utc.iloc[0])` vs where `lag48` first becomes non-NaN; compare `df.kwh.shift(48)` with `df.groupby("meter_id").kwh.shift(48)`: they differ on 48×19 rows. | `groupby("meter_id")[col].shift(k)`, or pivot to wide (one column per meter) and shift rows. | "How would you have noticed if the frame had been sorted by time first?" |
| 2 | `rolling(48).mean().shift(-48)` "aligned to the forecast day" | bug | Shifting the window *forward* makes `roll48` the mean of periods t+1..t+48 — the target day itself, target included. It is the largest coefficient (0.60). | Print the columns side by side: `roll48` at row t equals `kwh[t+1:t+49].mean()`. Ask "what is the last timestamp inside this window?" | `rolling(48).mean()` ending at t (optionally `shift(1)` if t itself is not yet known). | "Why doesn't the R² collapse when you fix it?" (the day's *level* is mostly known from lags anyway; the leak was real but not the source of the number) |
| 3 | `target = df["kwh"].shift(-48)` without groupby | bug | Same as #1 for the target: the last day of each meter is labelled with another meter's first day. | As #1. | Per-meter shift. | — |
| 4 | Faults left in: M100007's September is in Wh (×1000), M100003 stuck for a week | suspicious | Residential RMSE 7.09 kWh for meters averaging 0.2 kWh per half-hour is absurd; it is entirely M100007's September. The z-scoring for that meter is computed with a std of ~70, so its other 11 months are compressed to nothing. | `df.groupby("meter_id").kwh.describe()`; RMSE by meter and month. | Detect the unit change (monthly median ratio), fix or exclude; drop stuck days. | "Which single row would you look at first to explain RMSE 7?" |
| 5 | `train_test_split(test_size=0.2, random_state=0)` | choice | Shuffled rows: every test half-hour has neighbours from the same day in training. With autocorrelated data this is in-sample evaluation in disguise. | Look at `X_test.index` — interleaved with training indices. | Chronological split (fixed: Jan–Sep / Oct–Dec) or grouped by meter-day. | "What is the unit of independence here — a row, a day, a meter?" |
| 6 | `stats = df.groupby("meter_id").kwh.agg(["mean","std"])` on the full year | choice | Per-meter scaling uses test-period data. Mild leak here, but it is the pattern that becomes serious with a trending series. | The stats cell runs before any split. | Fit stats on train only. | "When does scaler leakage actually change results?" |
| 7 | `get_dummies(drop_first=True)` then "M100000 has no dummy ... its effect is zero" | bug (interpretation) | The dropped level is the *reference*, not zero: every other dummy is relative to M100000. The coefficients are also tiny (±0.03) because the target is z-scored per meter, so meter dummies cannot "capture level differences" — the z-scoring already removed them. | Look at the dummy coefficients: all ≈ 0. | Keep all dummies with Ridge, or drop the dummies (they are redundant after per-meter z-scoring). | "What would the dummies mean if the target were in kWh?" |
| 8 | "Convert back to kWh for reporting" — pooled `r2_score` in kWh | suspicious | R² in kWh across 20 meters of different sizes is inflated by *between-meter* variance: the model gets credit for knowing an SME is bigger than a household. 0.79 pooled vs 0.54 z-scored vs 0.46 median per meter. | Compute R² per meter and on the z-scored target; they disagree with the pooled number. | Per-meter metrics, scale-free (z or MAE/mean) for the portfolio. | "Which R² would you put in front of the trading desk?" |
| 9 | No naive baseline | choice | "Same period yesterday" and "same period last week" are never computed. Fixed version: the model beats both on 17/20 meters — but *not* on the two largest SMEs (M100000, M100015), which carry most of the kWh. | Absence. | Per-meter baseline table; weight the verdict by energy. | "Beats naive on 17 of 20 — is that a pass?" |
| 10 | Duplicates never removed | bug | 40 duplicated rows make `shift(48)` off by one row for those meters after the duplicate; `rolling` windows cover 47 real periods. | `df.duplicated().sum()` = 40. | Dedupe before any shift; assert one row per (meter, utc). | — |
| 11 | `utc` built from the local settlement date (as in mock 14) | bug | `hour` dummies and `is_weekend` are shifted by one hour Apr–Oct; the two DST days collide with their neighbours. | Periods per day of 46/50; `utc` has no tz. | Localise to Europe/London first. | "Would this bug change the R²?" (barely — which is why it survives) |
| 12 | "SME meters are forecast almost perfectly; residential meters are noisier" — read from RMSE in kWh | suspicious | RMSE is scale-dependent: SME 1.0 kWh vs residential 7.1 kWh says nothing about relative accuracy (and the 7.1 is #4). By R² the SMEs are *worse* (0.56 vs 0.79) and by MAE-vs-naive they are the meters the model fails on. | Compare the R² and RMSE columns of the same table: they rank the two groups oppositely. | Scale-free metrics per segment. | "Which metric would the SME account manager care about?" |

Also worth saying aloud: `X.shape` is (349,055, 48) — the author kept 349k rows of a 349,439-row file after creating 336-row lags, so almost no NaN were dropped, which only happens because lags ran across meters (#1); and the printed feature preview (rows 400–405) shows `roll48` ≈ 1.8 for a meter whose `kwh` at those rows is 0.4–4.7, a value that is clearly not the trailing mean of the previous day.

## Honest result

| Headline | Broken notebook | Fixed notebook |
|---|---|---|
| Pooled R² (kWh) | 0.791 (shuffled split, faults in, future window) | 0.796 (chronological, clean) — same number, still the wrong number |
| Pooled R² (z-scored) | 0.536 | 0.490 |
| Median per-meter R² | not computed | 0.465 (naive same-period-yesterday: 0.105) |
| Mean per-meter MAE | not computed | 0.151 kWh model vs 0.165 naive yesterday vs 0.162 naive week |
| Model beats naive | not computed | 17 of 20 meters; not M100000, M100015 (the two largest SMEs) |
| RMSE residential / SME (kWh) | 7.09 / 1.00 | — (replaced by per-meter, scale-free table) |
| Largest coefficient | `roll48_z` 0.60 (future window) | `roll48_z` 0.30, lags ≈ 0.22 each |

## Scoring

12 problems. 8+ with correct reasoning = strong. #1/#3 (no groupby) and #2 (the forward-shifted window) are the must-finds; #8 and #9 (pooled R², no baseline) are what separates "found the bugs" from "understood the result". Full marks only if you can explain why the R² barely moved after fixing everything and what number you would report instead.
