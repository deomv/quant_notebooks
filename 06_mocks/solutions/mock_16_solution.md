# Mock 16 — Wind fleet generation and forecast error: solution

Story: quote a PPA on a 10 GW wind portfolio. The notebook builds generation from a power
curve, emulates a day-ahead wind forecast, and concludes that forecast error is small and
imbalance risk is negligible. Every headline number is wrong, and the biggest problems are
not code bugs but modelling decisions that the author never questioned: an undocumented
measurement height, a self-made "forecast" that is far better than any real one, and a
capacity factor that no wind fleet has ever achieved.

The single number that should have stopped the analysis is **capacity factor 0.80**. UK
offshore wind runs at about 40%, onshore at about 27%. When your model says 80% the model
is wrong, not the wind.

## Planted problems

| # | Where | Kind | Why it matters | How to detect | Fix | Interviewer follow-up |
|---|---|---|---|---|---|---|
| 1 | Setup: `df["v_hub"] = df["wind_ms"] * (100/10) ** 0.14` "met-mast wind is at 10 m" | modelling choice (unverified assumption) | Nothing documents the measurement height. The 1.38x scaling raises mean wind from 7.3 to 10.1 m/s and, through the cubic curve, roughly doubles the energy (CF 0.31 -> 0.58 with a correct curve). This is the largest uncertainty in the whole quote. | Ask what height the column is; look at the data dictionary; compare CF under both assumptions | Treat the height as unknown; carry both cases; ask the data owner | "What is the first question you would send back to whoever supplied the wind data?" |
| 2 | Power curve: `curve_pts = {0:0, 3.5:0, 25:0, 12:1, 5:0.07, ...}` then `np.interp(v, list(keys), list(values))` | bug | `np.interp` requires increasing `xp`; with insertion order it returns garbage (0 up to 14 m/s, then 1). The printed check table shows this and the author blames "coarse tabulation". | The printed table is a step function; `np.all(np.diff(xp) > 0)` | `pd.Series(curve_pts).sort_index()`, assert monotonic | "What does `np.interp` do with unsorted x?" |
| 3 | `def power_curve(v, v_rated=9.0)` while `V_RATED = 12.0` is defined at the top and the manufacturer table has rated at 12 | bug | A rated speed of 9 m/s puts 60% of hours at full output. Capacity factor becomes 0.80; forecast error looks small because the curve is flat over most of the distribution. | The constant is defined and never used; CF 0.80 is impossible; median generation exactly 10,000 in `describe()` | Use `V_RATED` from the table | "How would you sanity-check a capacity factor?" |
| 4 | Markdown: "A capacity factor around 70% is consistent with a good offshore-weighted portfolio" | suspicious result accepted | It is not. The number should have triggered a search for the bug. | Domain knowledge; compare to public load-factor statistics | Reject and investigate | "What capacity factor would you expect for UK onshore? Offshore?" |
| 5 | Forecast: `v_fc = v_hub + normal(0, 0.5)` "emulate the vendor forecast" | modelling choice | Actual wind plus 0.5 m/s iid noise is a nowcast, not a day-ahead forecast. Real 12-35 h wind forecasts have errors of 1.5-3 m/s, growing with horizon, autocorrelated across hours, and biased. There is no origin time, so nothing can be checked for point-in-time correctness. | Ask "when was this forecast issued?"; compare the implied skill with persistence | Emulate an issued forecast with horizon-dependent, autocorrelated error (fixed notebook), or better, obtain real archived forecasts | "If you had a real forecast archive, which column would you check first?" |
| 6 | Merge: actual `ts = tz_convert("Europe/London").tz_localize(None)`, forecast `ts = tz_localize(None)` on UTC | bug | Local wall-clock joined to UTC wall-clock: every BST hour is paired with the forecast for the *next* hour. Row count is unchanged (17,520), so a count check does not catch it. | Monthly RMSE is 2-3x higher in Apr-Oct; check one known hour on both sides | Keep both sides tz-aware in UTC; `validate="one_to_one"` | "Why does the row count not change when the join is wrong?" |
| 7 | Markdown: "Summer months are noisier, which is expected from convective conditions" | suspicious result explained away | A plausible story attached to a symptom of the timezone bug. Winter RMSE 550-620 MW vs summer 1,300-1,540 MW is too clean a step at the DST dates to be weather. | The step happens at the clock change, not gradually | Fix the join, then re-check seasonality | "What would convince you it is weather rather than the join?" |
| 8 | Calibration: `LinearRegression().fit(m[["gen_fc_mw"]], m["gen_mw"])` on all rows, then RMSE on the same rows | modelling choice | In-sample calibration and evaluation. The RMSE reported is what the vendor's forecast *would* have done with a correction fitted on the answers. | No split anywhere in the notebook | Fit on 2022, evaluate on 2023 | "Would this correction have been available on 1 January 2023?" |
| 9 | `rmse_pct = rmse_mw / P_RATED` printed as "% of average output" | bug (label) | Divides by rated capacity, labels it as a share of average output. With a real ~30% capacity factor the two differ by a factor of three (16% vs 57%). | Read the code against the label; 11.5% of *average output* would be an outstanding forecast | Report both denominators, labelled | "Which denominator does a PPA counterparty care about?" |
| 10 | No persistence baseline anywhere | omission | Without "same hour yesterday" you cannot tell whether the forecast has skill. In the fixed notebook persistence is 29% of capacity vs 16% for the honest forecast, so the forecast does help, but you only know that once you compute it. | Look for the word "baseline" | Add persistence (D-1 for hours <= 12, D-2 otherwise, given a 12:00 origin) | "What is the simplest forecast you must beat?" |
| 11 | Daily energy: `sched = m[m["v_fc"] >= CUT_IN]` then `resample("D").mean() * 24` | bug | Filters out 432 low-wind hours, then scales the mean of the *remaining* hours to 24. Days with many calm hours are overstated. Hourly MW summed over the day is already MWh. | `describe()` of daily MWh; compare `.sum()` with `.mean()*24` on a day with gaps | `resample("D").sum()` with a count check | "What are the units of an hourly MW value summed over a day?" |
| 12 | Revenue: `(gen_mw * price.mean()).sum() / 2` | modelling choice | Values wind at the average price. Wind and price are negatively correlated (-0.30 here): the fleet earns less than average when it generates most. Capture price is 89% of the average in 2023. This is the cannibalisation effect and belongs in every wind PPA. | `np.corrcoef(gen, price)` | Sum `gen * hourly price` | "What is a capture price, and why is it below the baseload price for wind?" |
| 13 | Exposure: `rmse_mw * premium.mean() * 8760` | modelling choice | Treats forecast error and imbalance price as independent and symmetric. The fleet is a large part of the system: when it is short, the system tends to be short and the buy premium is high; when it is long, the sell price is low. Realised cost is 508 m EUR vs 318 m EUR if the premia were independent of the error. | Ask what the premium is *conditional on* the error sign | Settle the actual error hour by hour against premia that depend on system conditions | "Why is the imbalance cost of a wind fleet larger than error times average premium?" |
| 14 | Results text: "exposure is about 3% of revenue: imbalance risk is small; no separate risk premium is needed" | suspicious result | Every input to that ratio is wrong in the same direction (energy overstated 2.7x, error understated, price overstated, premium independent). The honest ratio is ~27%. | Trace each input | Rebuild from corrected inputs; quote a risk premium | "Which of the fourteen issues moves the PPA price most?" |

Non-issue to clear: the curve has no cut-out branch. Max wind is 16 m/s (22 m/s even in the shear case), so cut-out never triggers. Worth saying out loud with the number, not worth "fixing" in silence.

## Honest result

| Quantity | Broken notebook | Fixed notebook |
|---|---|---|
| Capacity factor | 0.801 | 0.310 fleet curve as measured (0.581 if the data really is 10 m wind) |
| Forecast error sd (m/s) | 0.5 iid | 2.2, autocorrelated, +0.3 bias |
| Forecast RMSE 2023 | 1,155 MW, "11.5% of average output" (in-sample, tz-shifted) | 1,642 MW = 16.4% of capacity = 57% of mean output (2022 calibration, 2023 test) |
| Persistence RMSE | not computed | 2,868 MW = 28.7% of capacity |
| Monthly RMSE pattern | winter 550-620, summer 1,300-1,540 | flat 1,400-1,850 all year |
| Annual energy 2023 | 69.2 TWh | 25.2 TWh |
| Revenue 2023 | 6,911 m EUR (average price, both years / 2) | 1,903 m EUR at hourly price; capture price 89% of average |
| Imbalance cost 2023 | 232 m EUR (RMSE x mean premium) | 508 m EUR realised; 318 m EUR if premia were independent of error |
| Cost / revenue | 3.4% | 26.7% |

## Scoring

The capacity factor is the tell. A candidate who says "80% is impossible, something upstream
is wrong" and traces it to the 9 m/s default argument, then asks what height the wind is
measured at, has done the important work. Add the unsorted `interp`, the timezone join (via
the summer RMSE step) and the missing persistence baseline for a strong pass. The capture
price and the error-premium covariance are what an energy-trading interviewer will push on
in the follow-up: those two are the difference between a data analyst and a trader.
