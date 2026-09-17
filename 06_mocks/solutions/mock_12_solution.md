# Mock 12 — Retail baseload hedge ratio: solution

Headline in the broken notebook: **hedge ratio 1.22× load, effectiveness 93%, residual risk
"a few thousand euros", hedged 95% VaR +8.4m EUR ("even the worst case makes money"), move the
hedge from 1.0 to 1.2.**
Honest result: **ratio 1.00× fitted on 2022 monthly surprises, out-of-sample 2023 effectiveness
0.997, residual sd 111 kEUR/month; 1.22× *adds* risk out of sample (effectiveness 0.949,
residual 489 kEUR/month) because the extra 0.22× is an outright long position.**

The notebook's own intermediate output already shows the hedge doubling risk (monthly
`margin_hedged` sd 10.7m vs 4.7m unhedged) and nobody stops to ask why.

| # | Where | Kind | Why it matters | How to detect | Fix | Interviewer follow-up |
|---|---|---|---|---|---|---|
| 1 | Hedge ratio — `np.cov(df.cost, df.payoff)` on **hourly levels** | modelling choice (the big one) | Regressing hourly cost on hourly price measures that expensive hours carry more load (intra-day shape) plus the common 2022 trend. The slope 1.22 is not a price hedge ratio. Baseload settles monthly; the risk is the monthly *surprise* vs the forward. On monthly surprises the ratio is 1.00. | Ask what the 0.22 extra MW is doing at 3 a.m.; recompute on monthly surprises; the per-year table gives 1.26 in both years, so it is structural (shape), not risk. | Fit `surprise = cost - F*load` on `(avg_spot - F) * hours` per month. Shape is hedged with shaped products, not by over-buying baseload. | "Why would a book with flat load want more than 1× baseload? What position does 1.22× leave you overnight?" |
| 2 | Same cell — `np.cov(..., ddof=0)[0,1] / df.payoff.var()` | bug (minor) | Numerator ddof=0, denominator pandas ddof=1: the ratio is scaled by (n−1)/n. Negligible with 17k rows, fatal with 24 months. | Two different libraries in one ratio. | Use one estimator. | "Which ddof does pandas use? numpy?" |
| 3 | Same cell — `effectiveness = r2` on the fitting sample | modelling choice | In-sample R² of a levels regression. Says nothing about how the hedge would have performed on unseen months. | No split anywhere. | Fit on 2022, report `1 - var(hedged)/var(unhedged)` on 2023. | "How much of that 0.93 is the 2022 gas trend?" |
| 4 | Forward — `F = df.price.mean()` over the whole sample | bug (look-ahead) | You cannot buy a forward at the realised 2022–2023 average in January 2022. Every payoff and every hedged margin uses a price from the future. | The "forward curve" is one number computed from the data being hedged. | Strike at a price known before the month (here: previous month's average as a proxy; in practice the forward curve). | "Where would you get a real forward curve, and what basis risk does that introduce?" |
| 5 | Everywhere — realised `load` in cost and payoff | modelling choice | Volume risk is assumed away. Cost = Σ price × load has a price–load covariance term (cold = high load = high price) that a fixed-volume hedge cannot remove. | Ask what is uncertain at the time of hedging: price *and* volume. | State it; hedge to expected load and quantify the volumetric residual with a load forecast. | "Which risk is bigger for a retailer in a cold snap: price or volume?" |
| 6 | Monthly — `cost = avg_price * load` (mean × sum) | bug | Σ(p·L) ≠ mean(p)·Σ(L). The load-weighted price is 1.4–4.4% above the simple average in every month; 2-year cost understated by 2.4%. | `df.cost.resample("MS").sum()` vs the product. | Sum hourly cost. | "What is the sign of the error and why is it always the same sign?" |
| 7 | Monthly — `margin_hedged` sd is **double** the unhedged sd in `describe()` | suspicious result | The notebook's own output says the hedge adds risk. Nobody comments. (Cause: #4's payoff is a level, not a surprise, combined with #9's sign.) | Read cell output. | Stop and investigate before proceeding to VaR. | "What did you expect that number to be?" |
| 8 | VaR — bootstrap draws days from **both** years; `np.percentile(..., 95)` | bug × 2 | (a) Mixing the 2022 spike into every "month" of a 2023 book misstates the distribution. (b) 95% VaR is a *loss* quantile: `-percentile(pnl, 5)`. `percentile(pnl, 95)` is the best case; that is why "VaR" came out as +8.4m of profit. | A positive VaR reported as good news; hedged "VaR" larger than unhedged. | Bootstrap within the year being priced (or block by regime); `var95 = -np.percentile(pnl, 5)`; add CVaR. | "If VaR is a profit, what is it measuring?" |
| 9 | VaR and monthly — `margin_hedged = margin - h * payoff` | bug (sign) | Payoff `(spot − F)·V·hours` is positive when spot rises, exactly when cost rises, so it *adds* to margin. Subtracting doubles the exposure. | Same evidence as #7; check one high-price month by hand. | `margin + h * payoff`. | "Walk me through the sign: spot goes up 10 €, what happens to cost, to the forward, to the hedged book?" |
| 10 | Seasonality — `groupby(df.index.month)` with the text "24 months" | bug (minor) | Pools Jan-2022 with Jan-2023: 12 rows, each a two-year sum. Harmless here, but the text is wrong and the same habit breaks any year-on-year analysis. | The table has 12 rows. | `groupby([year, month])`. | — |
| 11 | Results — "residual risk ≈ 3,700 EUR, negligible" | suspicious result | That is the sd of an *hourly* cost residual. Risk lives at settlement: monthly residual sd is 111 kEUR at 1.0× and 489 kEUR at 1.22×, against an average monthly margin of ~5m. Units matter. | Ask "per what?". | Report monthly (settlement-period) residual in EUR. | "Is 111k/month negligible for a book of this size? Compared with what?" |
| 12 | Results — "1.22× removes 93% of variance" | suspicious result | Direction of the recommendation is wrong: moving from 1.0 to 1.22 increases OOS residual sd from 111k to 489k. | Test the recommendation out of sample before making it. | — | "How would you know if the recommendation made things worse?" |

## Honest result

| | broken | fixed |
|---|---|---|
| Hedge ratio | 1.22 (hourly levels, full sample) | 1.00 (monthly surprises, fitted 2022) |
| Effectiveness | 0.927 in-sample R² | 0.997 OOS 2023 at 1.0×; 0.949 at 1.22× |
| Residual risk | 3,715 EUR (hourly) | 111 kEUR/month at 1.0×; 489 kEUR at 1.22× |
| Monthly cost | mean × sum, −2.4% | Σ hourly, load-weighted price +2.7% on average |
| 95% VaR | +8.4m EUR (best-case quantile, wrong sign, mixed regimes) | loss quantile within 2023; hedged sd 125k vs 775k unhedged |
| Forward price | realised 2-year mean (look-ahead) | previous month's average (known) |

Caveat in the fixed notebook: the synthetic prices have iid hourly noise, so monthly-average
surprises are unusually clean and 0.997 is optimistic for real forward-vs-spot basis.

## Scoring

Found #1 (and explained what the extra 0.22× is), #4, #9 (via #7) and #8b: strong.  
Stopped at the `describe()` showing the hedge doubling risk and refused to continue to VaR: that is the behaviour being tested.  
Only found #2 and #10: you are reading code, not results. Read the tables.
