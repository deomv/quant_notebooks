# Mock 11 — Battery arbitrage LP: solution

Headline in the broken notebook: **£2.68m/yr net, capex £15m, payback 5.6 years, "build five more".**
Honest result: **perfect-foresight upper bound £1.99m/yr against £30m capex (15 years); a naive
schedule on the trailing 7-day price profile earns £0.80m/yr (38 years).**

The LP in the notebook *minimises* revenue and the `abs()` hides it, so the number is the value of
the worst possible schedule, which by symmetry happens to look like a plausible best. Several other
errors partially cancel. A plausible-looking total is not evidence that the pieces are right.

| # | Where | Kind | Why it matters | How to detect | Fix | Interviewer follow-up |
|---|---|---|---|---|---|---|
| 1 | Daily LP — `c = np.concatenate([-p, p])` | bug (the big one) | `linprog` minimises `c @ x`. With `c = [-p, +p]` it minimises `p*(dis - ch)`, i.e. revenue. The battery charges at the evening peak and discharges at the trough. | The sanity-check day shows `charge = 25` at hours 17–19 (the most expensive) and `discharge` at hours 2–4. The daily revenue `describe()` is entirely negative. | `c = [+p, -p]` (minimise minus revenue). | "Why is every day negative? What does the schedule look like at the peak?" |
| 2 | Same cell — `A_soc = hstack([L / ETA, -L])` | bug | Energy stored is `eta * charge`; dividing by `eta` makes 1 MWh from the grid store 1.11 MWh. The battery creates energy. | On the sanity day `charged` ≈ 247 vs `discharged` 225: with 90% efficiency and a self-financing day discharge should be *less* than 0.9× charge. Or compute `soc` by hand for one day. | `L * ETA`. | "Which side do you apply efficiency to and why does the LP care?" |
| 3 | Same cell — no terminal constraint | modelling choice | Each day starts at 50 MWh and may end at 0, so the optimiser sells 50 MWh of free energy every day (≈ 18 MWh net after the other bugs). Over 730 days that is real money that never existed. | Compute `SOC0 + (ETA*ch - dis).sum()` for a day: it is not 50. | `A_eq` row forcing `soc_24 == SOC0` (or value the terminal energy at a price). | "Where does tomorrow's first discharge come from?" |
| 4 | Parameters — `P_MAX = 50 * 0.5  # MW -> MWh per settlement period` | bug (units) | The data is hourly (`index.diff()` is 1h, 24 rows per day; the code even reshapes to 24). The battery is throttled to 25 MW. The markdown says half-hourly; the code says hourly; neither was checked. | `prices.index.to_series().diff().value_counts()`; `groupby(date).size()` is 24. | `P_MAX = 50.0`. Discuss MW (rate) vs MWh (energy per period); at 30-minute settlement the period volume would be 25 MWh but there would be 48 periods. | "If this were half-hourly, what else in the notebook would change?" |
| 5 | Load — `prices = prices[prices > 0]` and the `try/except: continue` | bug + modelling choice | Negative prices are hours where a battery is *paid* to charge: the best hours of the year. Dropping them leaves 43 hours missing on 38 days; those days no longer reshape to 24 and are silently skipped. 692 of 730 days scheduled and nobody asks about the other 38. | `len(daily)` is 692, not 730. Count `prices < 0` in the original series. `res.status` is never inspected anywhere. | Keep negative prices; never swallow exceptions in a loop, count them; check `res.status == 0`. | "Which 38 days did you drop and what did they have in common?" |
| 6 | Payback — `annual_gbp = daily.loc["2022", "net_gbp"].sum() * FX` | bug | `net_gbp` is already in GBP (converted per day). The second `* FX` deflates by 13%. | Grep for `FX`: it appears in two places. | Convert once. | — |
| 7 | Whole loop — schedules against realised prices | modelling choice | Day-ahead bids are placed before prices are known. Scheduling on the actual prices is perfect foresight, an upper bound. The text calls it "realistic". | Ask what the battery knew when it bid. | Report the perfect-foresight number as an upper bound and add a forecast-based schedule (here: trailing 7-day profile, which earns 40% of the bound). | "How would you estimate what fraction of perfect foresight a real forecast captures?" |
| 8 | Degradation — `cycles = discharged / (2 * E_MAX)` and the cost applied *after* optimisation | bug + modelling choice | An equivalent full cycle is `E_MAX` MWh discharged; dividing by `2*E_MAX` halves the degradation bill. Separately, an LP that does not see the degradation cost cycles ~1,000 times a year chasing €5 spreads; the cost should be in the objective (≈ €17/MWh discharged), which cuts cycling to ~650 and *raises* net revenue. | 551 "cycles" over two years vs `discharged.sum()/E_MAX` ≈ 1,100. | `discharged / E_MAX`; add `+DEG` to the discharge cost vector. | "What is your marginal cost of a cycle and where does it enter the optimisation?" |
| 9 | Payback — 2022 used as "the" year | modelling choice | 2022 is the gas-spike regime. With everything else fixed 2023 is actually higher here, but the point is that one year is not an annual figure; show both and say which you rely on. | Two years exist; only one is used. | Average or the conservative year, stated. | "Which year would a bank lend against?" |
| 10 | Payback — `capex_gbp = 300 * (50 * 1000)` | bug (units) | £300/kWh is a price per unit of *energy*; the battery has 100,000 kWh, not 50,000. Capex is £30m, not £15m. | Read the units in the markdown against the code. | `300 * E_MAX * 1000`. | "Is that price per kW or per kWh, and which one is your battery sized in?" |
| 11 | Results — "payback 5.6 years, build five more" | suspicious result | Even before checking anything, a 2-hour battery paying back in 5.6 years on day-ahead arbitrage alone should trigger "what fraction of perfect foresight is this and what does the schedule look like". | Ask for a per-hour schedule and a naive baseline. | — | "What would make you *not* build it?" |
| 12 | Nowhere — simultaneous charge/discharge never checked | hygiene | With the wrong efficiency and objective the LP can profit from charging and discharging in the same hour. It does not happen on these days, but a battery model should assert it. In the fixed model the check is 0 hours. | `((ch > 0) & (dis > 0)).sum()` per day. | Add the check (or a binary mode variable if it ever triggers). | "Why can an LP without binaries still produce a physically valid schedule here?" |
| 13 | Sanity check — `-0.00` in the schedule table | cosmetic | Tiny negative solver values. Harmless, but the author printed the table and did not look at it: charging at 174 €/MWh is in the same table. | Read the table. | Round with `np.abs`/clip; more importantly, read it. | — |

## Honest result

| | broken | fixed |
|---|---|---|
| Days scheduled | 692 (38 silently skipped) | 730 (0 failed) |
| Gross revenue 2022–23 (£m) | 6.66 (abs of a *minimised* revenue) | 5.96 perfect foresight, with degradation in the objective |
| Equivalent full cycles / yr | 276 | 657 |
| Annual net (£m/yr) | 2.68 (2022 only, FX applied twice) | 1.99 perfect-foresight bound; 0.80 naive 7-day profile |
| Capex (£m) | 15 | 30 |
| Payback (years) | 5.6 | 15.1 (bound) / 37.5 (naive) |
| Schedule on 2022-12-12 | charges 17–19h at €154–174 | discharges 17–18h, charges 2–3h and 22–23h |

## Scoring

Found #1 (by looking at the schedule, not just the sign), #2 or #3, #4/#10 (units) and #7 (foresight): strong.  
Found #1 and #7 and asked for a baseline: that is the core of the exercise.  
Only found #6 and #13: read the intermediate tables, not just the summary.
