# Mock 17 — Customer insights, no code bugs: solution

Story: eleven "insights" for a retail pricing review. Every line of pandas is correct. Every
conclusion is wrong, because each one rests on a statistical fallacy. This mock trains the
second half of the interview: after "the code runs and the numbers are right", the question
is "do the numbers mean what the sentence says?".

There is nothing to fix in the code. What the candidate must do is say, for each finding,
what comparison would actually support the claim, and then (fixed notebook) run it.

## Planted fallacies

| # | Where | Fallacy | Why it matters | How to detect | Fix | Interviewer follow-up |
|---|---|---|---|---|---|---|
| 1 | "Regional efficiency": `cust.groupby("region")["kwh_2023"].mean()` — Scotland 35% below Wales | Simpson's paradox (confounding by mix) | SMEs use ~8x more than households. Scotland's book is 5% SME, the others 11-16%. Within residential Scotland is 8% below Wales, within SME it is 4% *above*. | Ask "what is the mix?"; stratify by anything that differs by 8x | Compare within customer type, or standardise the mix | "What other variable could differ between regions and drive this?" |
| 2 | "High users self-correct": top 50 on one day fall 19% a week later | Regression to the mean | Selecting on a noisy measurement guarantees the selected group looks more ordinary on any other day. The bottom 50 "rose" 17.7%; selecting the top 50 on the *later* day makes them look like they went *up* 21.6% into it. Over 30-day averages the same meters are flat (+0.2%). | Run the same comparison on the bottom 50; reverse the selection day | Select on one period, evaluate on another with a control group, or use longer averaging windows | "How would you design a test that could show the outreach letter works?" |
| 3 | "Consumption trend": H2 vs H1 on meters with >= 360 days | Seasonality read as trend, plus a survivorship-style filter | H2 < H1 by 11% is the heating season; the monthly profile alone predicts -10.8%. Restricting to "complete" meters is harmless here (gaps are random) but the *reason* matters: if gaps came from churn or faults on high users, the filter biases the answer (simulation: excluding the 60 fastest-falling meters moves the number by +0.9%). | Ask what causes missing days; compare with a seasonal baseline | Seasonal comparison (same period year on year, or profile-adjusted); include all meters, model missingness | "Why might a meter have gaps, and does that reason correlate with what you are measuring?" |
| 4 | "London is different, every month": 60 region x month t-tests, 18 significant, "12 independent confirmations" | Multiple comparisons and pseudo-replication | The twelve monthly tests reuse the same 87 London meters, so they are one finding measured twelve times. Sixty tests at 5% would give three false positives by chance anyway. Tested once at meter level and corrected for five regions (Benjamini-Hochberg), London is p = 0.09: suggestive, not established. | Count the tests; ask whether the months share meters | One pre-specified test per hypothesis at the right unit; correct for the number of hypotheses | "If you ran 60 tests on pure noise, how many would be significant?" |
| 5 | "Identifying solar customers": accuracy 0.889 | Base-rate neglect | 88.7% of meters have no solar; predicting "no" for everyone scores 0.889. The model has AUC 0.51 and recall 0 for solar. | Compare with the majority-class rate; look at recall or a confusion matrix | Report AUC / precision-recall against the base rate | "What accuracy would a coin with the right bias get?" |
| 6 | "Where the solar customers are": London 13, "more than the next two combined" | Denominator neglect | London has the most solar customers because it has the most customers (99). The rate is 13-14% in London, North and Scotland alike; Midlands (2%) is the real outlier. | Divide by the number of meters | Report rates with denominators | "Most of anything is in London. What would tell you London is special?" |
| 7 | "TOU increases consumption": +5%, p = 1e-35 on daily rows | Self-selection (correlation read as causation) and pseudo-replication | Customers choose TOU; whoever chooses it differs from those who do not. And 16,450 daily rows come from 46 meters: at the meter level the difference is +4.9% with CI [-3.5%, +13.3%], p = 0.25. | Ask how customers ended up on the tariff; count the meters behind the rows | Randomised or matched comparison; test at meter level | "Why is a day-level t-test wrong when the treatment is at meter level?" |
| 8 | "Solar reduces consumption": correlation -0.39 across five regional averages | Ecological fallacy (and n = 5) | A relationship between group averages does not transfer to individuals. At meter level solar customers use +1.4% (p = 0.78): no difference. Five points can produce any correlation. | Note n = 5; run the same comparison at the individual level | Meter-level comparison | "What is the sample size of that correlation?" |
| 9 | "Q4 volume outlook": linear fit on Jan-Jun, extrapolated to Oct-Dec = 6% of Q1 | Extrapolating a straight line through a cycle | Jan-Jun is the falling half of the annual cycle; a line through it predicts negative consumption in December (clipped to zero). A cosine fitted to the same six months gives 481 MWh for Q4 vs 495 actual. | Plot it; ask what happens in winter | Seasonal model or last year's profile | "What is the physical reason consumption should turn around in July?" |
| 10 | "Structural decline": Jan -> Jul -52% | Cherry-picked window (peak-to-trough) | The same series is +100% Jul -> Dec and -3% Jan -> Dec. | Look at the whole year | Compare like with like: same month year on year, or a full cycle | "Which two months would you pick to show consumption is *rising* 100%?" |
| 11 | "Fixed vs variable": p = 1e-53 on 75k daily rows, "one of the strongest effects" | Significance is not effect size; wrong unit of analysis | A 5% difference is small, and 75k daily rows are 209 meters. At meter level: +5.4%, CI [-2.2%, +13.1%], p = 0.17. The p-value measures sample size, not importance. | Ask for the effect size and CI; ask what the independent unit is | Effect size with CI at meter level | "What does a p-value of 1e-53 tell you about how much money is at stake?" |

## Honest result

| Finding | Mock claim | Fixed notebook |
|---|---|---|
| 1 Regional efficiency | Scotland -35% vs Wales | residential: Scotland -7.8%, North -9.1%, London +2.3%; SME: Scotland +4% |
| 2 High users self-correct | -19% in a week | bottom 50: +17.7%; 30-day windows: +0.2%; pure regression to the mean |
| 3 H2 vs H1 | -10.8%, "consumption falling" | seasonal profile predicts -10.8%; no trend |
| 4 London every month | 12/12 months significant | one test: +8.5%, raw p 0.018, BH p 0.09 |
| 5 Solar classifier | 89% accuracy | always-no 88.9%; AUC 0.51; recall 0 |
| 6 Solar by region | London "by far the most" (13) | rates 14.0 / 13.6 / 13.1% for Scotland / North / London; Midlands 2.2% |
| 7 TOU | +5%, p = 1e-35 | meter level +4.9%, CI [-3.5, +13.3], p = 0.25; observational anyway |
| 8 Solar reduces use | corr -0.39 (n = 5 regions) | meter level +1.4%, p = 0.78 |
| 9 Q4 volume | 32 MWh (6% of Q1) | actual 495 MWh; seasonal fit 481 |
| 10 Structural decline | -52% Jan-Jul | -3.2% Jan-Dec |
| 11 Variable vs Fixed | p = 1e-53 | +5.4%, CI [-2.2, +13.1], p = 0.17 at meter level |

Findings that survive in some form: none as stated. The only real signal is a modest London
residential premium (~8%), which needs a pre-registered test or another year of data.

## Scoring

The point is not to find a bug, because there is none; a candidate who spends 30 minutes
looking for a wrong `groupby` has missed the exercise. Naming Simpson's paradox (1),
regression to the mean (2), the base rate (5), the denominator (6) and pseudo-replication /
wrong unit of analysis (7, 11) is a strong pass. Recognising that 4, 9 and 10 are all the
same seasonality-and-selection problem in different clothes, and proposing the stratified,
meter-level, seasonally-adjusted comparison for each, is what a senior interviewer is
listening for.
