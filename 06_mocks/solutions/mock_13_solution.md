# Mock 13 — Day-ahead volume under imbalance prices (newsvendor): solution

Headline in the broken notebook: **"being long is the expensive side, buy less than the forecast";
q* = 0.378, safety margin −222 MWh, q*-rule cost 583 kEUR vs 1,249 kEUR for buying the forecast,
best margin −1% saving 7.2%, Monte Carlo CI of width zero ("very stable").**
Honest result: **q* = 0.622, buy about +0.9% *above* the median forecast; imbalance cost 1,195 kEUR
vs 1,203 kEUR for the plain forecast (−0.6%), while the original −2% rule costs 1,618 kEUR (+34%).
Imbalance is 0.5% of procurement cost, so the forecast matters far more than the rule.**

Three errors reinforce each other: the inverted fractile says "buy less", the swapped cost
function makes buying less look cheaper, and evaluating only the short hours makes the rule look
spectacular. Each on its own would have been visible; together they produce a confident, wrong
recommendation.

| # | Where | Kind | Why it matters | How to detect | Fix | Interviewer follow-up |
|---|---|---|---|---|---|---|
| 1 | Optimal quantile — `q_star = c_o / (c_u + c_o)` | bug (the big one) | Newsvendor optimum is `c_u / (c_u + c_o)` with `c_u` the cost of being *short* (premium ≈ 25) and `c_o` of being *long* (discount ≈ 15): 0.62. The notebook's 0.38 buys below the median although being short is the expensive side. | Sanity check with the extreme: if the premium were huge, would you buy more or less? The formula gives *less*. | Swap. | "Derive the critical fractile in one line from the marginal cost of one more MWh." |
| 2 | Load forecast — `Ridge().fit(data[...])` on all data, then `train`/`test` split *after* | modelling choice | 2023 residuals are in-sample. Small for Ridge here (sd 888 vs honest 894), but it is the pattern. | The split appears after `fit`. | Fit on 2022 only. | "How much would this matter with a tree model?" |
| 3 | Percentage-margin grid — minimised on `book` (2023) and reported on 2023 | modelling choice | The "optimal" margin is tuned on the evaluation set. | The grid loops over the test book. | Choose on 2022 (train residuals + the same price mechanics), apply once to 2023. Honest choice: +1.0%, test cost 1,203k = same as the forecast. | "What would you expect the OOS gain to be if the margin is chosen OOS?" |
| 4 | `adj = np.quantile(train.resid, q_star)` (national MWh) then `book.q = fc + adj / 100` | bug (units) | Residuals are in MWh of *national* load; the book is 1%. The ad-hoc `/100` happens to equal `SHARE` here, but the unconverted −222 MWh went into the summary as the "safety margin" for a 290 MWh book. | −222 MWh margin on a 290 MWh hourly load is 77% of the book. Read the units. | `adj * SHARE`. | "What is the unit of every number in your summary table?" |
| 5 | `imbalance_cost` — `short = max(q - load, 0)`, `long_ = max(load - q, 0)` | bug (sign) | Sides swapped: hours where we are actually short are charged the *discount*, long hours the *premium*. Buying less is now cheaper, which "confirms" #1. | Take one row: q < load means short; the function calls that `long_`. Or check the grid: cost falls as you buy less — opposite to the premium/discount asymmetry. | Swap the two lines. | "Which direction should the cost curve be skewed and why?" |
| 6 | "Cost on the hours where we were actually short" → summary "q* rule 583 kEUR" | bug | The summary takes the *short-hours-only* subtotal as the rule's total imbalance cost and compares it with the *all-hours* cost of the forecast rule. Half the cost is missing. | `book.cost_q.sum()` (1,167k) vs `short_hours.cost_q.sum()` (583k): the summary quotes the second. | Sum over all hours for both. | "Where did the long-side cost go?" |
| 7 | Results — a 53% "saving" from the q* rule and 7.2% from a −1% margin | suspicious result | With a residual sd of 3% of load and a 25/15 asymmetry, the newsvendor gain is theoretically small (<1%). A 50% reduction in imbalance cost from a quantile shift should not be believed. | Back-of-envelope: expected imbalance cost ≈ E|resid| × average of the two prices; the rule can only rebalance the tails. | Report the honest 0.6%. | "What is the most a perfect quantile rule could save here? Why?" |
| 8 | Quantile regression — `QuantileRegressor(quantile=0.9)` judged by RMSE | bug + modelling choice | Wrong quantile (decision quantile is 0.62) and wrong metric (RMSE rewards the mean; a 0.9-quantile forecast *should* have worse RMSE). "Discard QR" is unsupported. | RMSE 1,564 vs 890 is exactly what a 0.9 quantile must look like. | Fit at q*, evaluate with pinball loss at q* and realised imbalance cost (1,199k vs 1,203k: a wash). | "Which loss function corresponds to your decision?" |
| 9 | Load — `imb.index.tz_convert("Europe/London").tz_localize(None)` then `book.index.tz_localize(None)` and `join` | bug (tz) | The price frame is London wall-clock, the load frame UTC wall-clock. 10,416 hours (all BST) join to the wrong hour; the October 01:00 duplicate adds a row and the March gap loses one, so `len(book)` is still 8,760 and the check passes by coincidence. Effect on totals is small here only because the simulated premium has no diurnal shape. | `imb.index.duplicated().sum()` = 2 after localising; compare `book.da_price` with `df.price_eur_mwh` for a July hour. | Keep everything in UTC; convert to local only for display. | "The row count did not change. Is that proof the join is right?" |
| 10 | Monte Carlo — `np.random.seed(42)` **inside** the loop | bug | Every scenario draws the same residuals; 200 identical simulations, CI width 0, reported as "very stable". | The CI has identical bounds. `len(set(sims))` is 1. | One `default_rng` outside the loop. | "What does a zero-width CI tell you?" |
| 11 | `cost / da_price` monthly means | bug (numerics) | Day-ahead prices go to zero and negative (29 hours ≤ 0 in 2023): the ratio explodes and flips sign (March mean −1.47). Averaging it is meaningless. | A negative "cost per euro". | Normalise by something that cannot be zero (buy–sell spread) or exclude and say so. | "What is the ratio meant to show, and is there a quantity that answers it?" |
| 12 | Imbalance price simulation itself | modelling choice (to say aloud) | Premium and discount are drawn iid, independent of system state. Real imbalance prices are correlated with the system being short/long, with the day-ahead level and with our own forecast error (when we are short, the system usually is too). The decision rule's value depends on exactly that correlation. | Ask how the imbalance prices were made. | Note the assumption; a real study needs real imbalance prices. | "Which way does the correlation between your error and the system's error move the optimal quantile?" |
| 13 | Nowhere — imbalance vs procurement | context | Imbalance is 1.2m EUR against 227m EUR of day-ahead purchases (0.53%). The notebook optimises the small number and never mentions the big one. | Compute the day-ahead bill once. | Put the result in context; forecast RMSE is the lever. | "If you had a week, would you spend it on the quantile rule or on the forecast?" |

## Honest result

| | broken | fixed |
|---|---|---|
| Critical fractile | 0.378 (buy below median) | 0.622 (buy above median) |
| Safety margin | −222 MWh (national units on a 290 MWh book, then /100) | +264 MWh national = +2.6 MWh book (+0.9%) |
| Imbalance cost, buy forecast | 1,249 kEUR (swapped sides) | 1,203 kEUR |
| Imbalance cost, quantile rule | 583 kEUR (short hours only) | 1,195 kEUR (−0.6%) |
| Quantile regression | RMSE 1,564 "worse", discarded | at q*: 1,199 kEUR, pinball 3.42 vs 3.42, a wash |
| Original −2% / −1% rule on honest costs | — | 1,618 kEUR (+34%) / 1,339 kEUR (+11%) |
| Monte Carlo 95% CI | [1,196, 1,196] kEUR | [1,166, 1,226] kEUR |
| Imbalance as share of procurement | not shown | 0.53% |

## Scoring

Found #1 and #5 (and saw that they reinforce each other), #6, #10: strong.  
Refused to believe a 53% saving and worked out the theoretical maximum: that is the behaviour being tested.  
Found only #10 and #11: good eye for outputs; now check the economics of the formula.
