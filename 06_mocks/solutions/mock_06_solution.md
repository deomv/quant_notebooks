# Mock 06 — Groupby / pivot report: solution

Notebook: `mock_06_groupby_report.ipynb`. Fixed version: `mock_06_fixed.ipynb`.
12 planted problems (numbered 1–12; the fixed notebook also adds a "Fix 0" merge check).
Kinds: **bug** = wrong output, **choice** = runs but methodologically wrong, **suspicious** = a result you should refuse to accept.

| # | Where | Kind | Why it matters | How to detect | Fix | Likely follow-up |
|---|---|---|---|---|---|---|
| 1 | "Meters per region" — `data.groupby("region").size()` on the *merged reading-level* frame | bug | Counts readings, not meters: "London has 34,325 meters" when the portfolio has 300. Reported as finding 1. | `meters.shape` is (300, 7); the numbers are ~350x too large. Ask "what is one row of `data`?" | `meters.groupby("region").size()` (London 99). | "What is the unit of observation in each table you touched?" |
| 2 | Cross-check — `meters.groupby("region")["tariff"].count()` | bug | `count()` excludes NaN; 13 meters with no tariff vanish, so the "cross-check" sums to 287 not 300 and the author moves on. | `.sum()` of that Series is 287; `meters.tariff.isna().sum()` = 13. | `size()`, or `count()` on the key column. | "When do `size()` and `count()` differ?" |
| 3 | Consumption by segment — `groupby(["region","tariff"])` | bug | `dropna=True` (default) silently drops rows whose tariff is NaN *and* the 200 orphan rows with NaN region. Segment table totals 1,593,416 kWh vs a grand total of 1,695,749 printed in the Results cell — a 6% hole the author never reconciles. | Sum the table and compare with `data.kwh.sum()`; they differ by 102,333 kWh. | `groupby(..., dropna=False)`, name the NaN bucket, reconcile totals (fixed table = grand total = 1,692,372 after removing orphans). | "Where did the missing 102k kWh go?" |
| 4 | Solar penetration — `groupby("region").has_solar.mean()` on `data` after casting to float | choice | Reading-weighted, not meter-weighted; only correct if every meter has the same number of readings. Here gaps are uniform so numbers barely move (Scotland 0.141 vs 0.140), but the lower-case regions show 0% and the orphan is excluded by the NaN key. The `astype(float)` was needed because the left merge turned the bool column into `object` — a clue the merge introduced NaNs. | Ask what the denominator is. `data.has_solar.dtype` is object before the cast. | Compute on `meters` after fixing casing. | "It gave the same answer — is it still wrong?" (yes: correctness by coincidence) |
| 5 | Top 10 meters — `readings.nlargest(10, "kwh")` | bug | Top 10 *daily readings*, not meters: M100054 appears three times, M100236 and M100240 twice. Presented as "top 10 meters". | Duplicate `meter_id` values in the table; every row is a single date. | Aggregate per meter first (`groupby("meter_id").kwh.agg(total, per_day, days)`), then rank; report days observed. | "Total or per-day? What if meters have different coverage?" |
| 6 | Monthly trend "(total kWh)" — `pivot_table(index=month, columns=tariff, values=kwh)` | bug | Default `aggfunc="mean"`: the table shows ~10–22 kWh (mean daily reading) under a heading that says total. | The magnitudes: a tariff's *total* monthly kWh cannot be 20. | `aggfunc="sum"`. | "What is the default aggfunc of `pivot_table`? Of `pivot`?" |
| 7 | "Consumption is trending down: -52% from January to July" | suspicious / choice | One year of data with a heating-driven seasonal cycle. Jan→Jul is seasonality, not a trend; December is back to -3%. | The series is U-shaped and returns to January levels; there is no second year to compare like-for-like. | Same-month-different-year comparison or seasonal adjustment; otherwise refuse to call it a trend. | "How would you separate trend from seasonality with one year of data?" (you mostly can't; say so) |
| 8 | "Month-on-month growth" — `trend.pct_change(axis=1)` | bug | Changes across *tariff columns* (Fixed→TOU→Variable), not across months: the Fixed column is all NaN and "growth" is +11–14% every month. | First column entirely NaN; values nearly constant across months. | `pct_change()` with the default `axis=0` (fixed: Feb -12.7%, Mar +1.1%, Apr -16.4% ...). | "Which axis is which in pandas, and how do you check quickly?" |
| 9 | "Average daily kWh per customer type" — `transform("mean").head()` | bug | `transform` returns one value per *row*; `.head()` shows five copies of the SME mean because the first rows all belong to meter M100000 (an SME). The author concludes "≈ 70 kWh per customer per day"; residential is 8.9. | Five identical numbers with a `RangeIndex`, no group labels. | `agg(["mean","median","count"])`. | "When would you actually want `transform`?" (adding a group statistic back to every row, e.g. within-group z-score) |
| 10 | Region labels — `london`, `wales`, `north`, `midlands` | bug | Six meters form four phantom regions; they leak into every by-region table (meters per region, segment totals, solar share 0%, tariff mix). | `meters.region.value_counts()` shows 9 labels for 5 regions. | `str.strip().str.title()` before grouping. | "What other kinds of key inconsistency would you look for?" (whitespace, encodings, ids with leading zeros) |
| 11 | Actual vs estimate — `data[data.date >= "2023-04-01"]` then `sum()` vs `annual_kwh_estimate` | choice / suspicious | Nine months of actuals (and the *low* season, April–December) compared with an annual figure: "customers use -31% vs estimate". Also ignores the ~2% of days missing per meter and the 6 meters with a NaN estimate (count 294). | Ratios cluster tightly at 0.69 ≈ 9/12 with a seasonal haircut — a number that uniform is a construction artefact, not a customer behaviour. | Full-year daily mean × 365 (coverage-adjusted); honest ratio 1.031 (+3.1%). | "If you only had April–December, how would you make the comparison fair?" |
| 12 | "Tariff mix within each region" — `crosstab(..., normalize=True)` | bug | `normalize=True` divides by the grand total, so rows do not sum to 1 and "TOU share within London" reads 5.6% instead of 17.2%. | Row sums are not 1 (`tariff_mix.sum(axis=1)`). | `normalize="index"`. | "What does `normalize='columns'` give and when would you want it?" |

Merge hygiene (not counted above, but the first thing to do): `readings.merge(meters, how="left")` without `validate=` or `indicator=` hides the orphan meter M999999 (200 rows, NaN in every attribute). `validate="many_to_one", indicator=True` shows `left_only: 200` immediately.

## Honest result

| Finding | Broken notebook | Fixed notebook |
|---|---|---|
| Largest region | London, 34,325 "meters" (32%) | London, 99 meters of 300 (33%) |
| Portfolio kWh vs segment table | 1,695,749 vs 1,593,416 (unreconciled) | 1,692,372 = 1,692,372 (200 orphan rows excluded, no-tariff bucket shown) |
| Avg daily kWh per customer type | "≈70" | residential 8.9, SME 69.7 |
| Solar penetration low / high | london 0.0% / Scotland 14.1% | Midlands 2.2% / Scotland 14.0% |
| TOU share within London | 5.6% | 17.2% |
| Top 10 meters | 10 daily readings, 3 meters repeated | 10 distinct meters, all SME, with days observed |
| Jan→Jul | "-52% trend" | -52% seasonal; no trend claim |
| Actual vs estimate | -31% | +3.1% (annualised, coverage-adjusted) |

## Scoring

12 problems. 9+ found = strong. #1, #5, #9, #12 are visible from the printed output alone; missing them means you read the code but not the results. #3 requires actually reconciling two printed totals. #7 and #11 are judgement, not code — an interviewer will push on those.
