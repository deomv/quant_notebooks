# Mock 02 — Regional consumption report: solution

Headline in the broken notebook: **"All 300 meters covered, 125,191 readings. London customers use 1.52x more than Scottish customers. Customers consume 1x to 149x their annual estimate — the estimates need rebuilding. Largest price increase to London."**

Honest result: **London/Scotland mean 1.41x, median 1.11x, residential-only median 1.11x. The gap is customer mix (12% SME meters in London vs 5% in Scotland). Customers use 1.03x their estimate; the estimates are fine.** 200 readings belong to a meter that is not in the master.

| # | Where | Kind | Why it matters | How to detect | Fix | Interviewer follow-up |
|---|---|---|---|---|---|---|
| 1 | Load — `meters["region"].unique()` shows `london`, `wales`, `north`, `midlands` | bug | Nine groups instead of five in every groupby; the lowercase groups have 1–3 meters and distort the summary; the final `summary.loc["London"]` excludes those meters. | The `unique()` output is printed in the notebook. `meters.region.value_counts()`. | `str.strip().str.title()` before grouping. | "You printed the unique values. What did you see?" |
| 2 | Clean — `tariff.fillna("Fixed")` "missing means default Fixed" | research choice | An assumption stated as fact. 13 meters (4%) get a tariff and a unit rate they may not have; the cost figures inherit the guess. | `meters.tariff.isna().sum()` = 13; there is no business rule in the data. | Keep NaN / label "Unknown"; report the count; ask the business. | "What evidence do you have that missing means Fixed?" |
| 3 | Clean — `annual_kwh_estimate.fillna(0)` then `/ (estimate + 1)` | bug | Six meters get a denominator of 1, so their ratio is ~3000. The regional `mean` of the ratio is then 35–149x instead of 1.03x, and the notebook concludes the estimates are broken. | `per_meter.describe()` shows ratio max in the thousands with median 1.0. | Leave NaN; the ratio is NaN for those 6 and excluded. | "Median 1.0, mean 76. Which one do you believe and why?" |
| 4 | Clean — `kwh > 150` set to NaN as "meter faults" | research choice | All 257 affected readings are SME meters (SME mean 70 kWh/day, residential max 41). The rule trims the biggest customers and biases SME totals down. | Join the flagged rows to `customer_type` and count. | Flag, do not delete; or set the threshold per customer type. | "Who did that threshold remove?" |
| 5 | Join — `merge(..., how="inner")`, then "all readings matched" | bug (silent) | 200 readings for meter `M999999` are dropped without a trace. Row count goes 107,503 → 107,303 and nobody checks. | `len(merged) != len(readings)`; use `indicator=True`. | `how="left", indicator=True, validate="many_to_one"`, assert counts, report the orphan. | "What happened to the 200 rows?" |
| 6 | Rates — `merged.merge(rates, on="tariff")` with two TOU rows | bug | Every TOU reading is duplicated (peak + offpeak rows). Row count jumps 107,303 → 125,191, TOU meters' `total_kwh` and `total_cost` double, and "readings" in the results is inflated. | The `print(merged.shape)` right after the merge shows the jump. `validate="many_to_one"` would raise. | One rate row per tariff (blended), or split kWh into periods first. | "Your row count went up after a lookup join. What does that tell you?" |
| 7 | Per-meter — `n_readings=("kwh", "size")` | bug (minor) | `size` counts rows including the NaNs created in #4 and the duplicates from #6. `count` counts non-missing values. | Compare `size` and `count` for one SME meter. | Use `count`, and say which you mean. | "Size or count, and does it matter here?" |
| 8 | Per-meter — raw `total_kwh` compared with an *annual* estimate | research choice | Meters have 350–365 days of readings (2% missing). The raw sum understates by up to 4% before you even start. | `n_days.min()` = 350. | Annualise: `total * 365 / n_days`. | "Is a missing day zero consumption or unknown consumption?" |
| 9 | Summary — `avg_kwh = mean(total_kwh)` per region | suspicious result | Regional means are driven by the SME share (an SME uses ~7x a household). London 12% SME vs Scotland 5% explains the whole gap; residential medians are 3414 vs 3065. | Split by `customer_type`; compare mean vs median. | Report median and split by customer type before comparing regions. | "Is that a regional effect or a mix effect? How would you test it?" |
| 10 | Pivot — `pivot_table(index="region", columns="tariff", values="kwh")` under the heading "Total kWh" | bug | Default `aggfunc="mean"`; the table shows mean daily kWh (10–25), not totals (tens of thousands). Nobody reading the heading would guess. | Values are far too small to be annual totals. | `aggfunc="sum"`, `margins=True`. | "Are those totals plausible for 96 meters over a year?" |
| 11 | Targets — merge with upper-case region names, then `summary.head()` printed instead of `plan` | bug (silent) | `plan` is an empty DataFrame. The cell prints a different variable, so the failure is invisible. | `len(plan)` → 0. Print what you built. | Normalise keys; `assert len(plan) == len(summary)`. | "Show me `plan`." |
| 12 | Results — "All 300 meters covered, 125,191 readings" | suspicious result | 300 is right only because the lowercase groups are counted separately; 125,191 is inflated by #6 (true: 107,303 matched + 200 orphaned). | Compare with `len(readings)` at load time. | Fix #5/#6 and report both numbers. | "Where did 18,000 extra readings come from?" |
| 13 | Results — "1x to 149x their estimate" accepted | suspicious result | A 149x ratio for a whole region is impossible for real customers; it should have stopped the analysis. | Median of the same column is 1.0. | Fix #3; report the median. | "What would you have expected the ratio to be?" |

## Honest result

| | broken | fixed |
|---|---|---|
| Regions in summary | 9 | 5 |
| Readings after joins | 125,191 | 107,303 (+200 orphaned, reported) |
| London / Scotland | 1.52x (mean) | 1.41x mean, 1.11x median, 1.11x residential median |
| Actual vs estimate | 35x–149x (mean of ratio) | 1.03x (median), range 0.99–1.07 |
| Tariff "Fixed" count | 156 | 143 + 13 Unknown |
| `plan` table | empty | 5 rows |

The correct commercial message is the opposite of the broken one: regions look different because of customer mix, and the annual estimates are good.

## Scoring

Found 9+ of 13 including #5, #6 and #9: strong.  
Found #6 (row count jump) and #3 (absurd ratios) and refused the headline: that is the behaviour being tested.  
Found only #1 and #10: work on "check the row count after every merge" and "mean vs median".
