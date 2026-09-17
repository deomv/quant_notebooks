# 06_mocks — deliberately broken research notebooks

Eighteen notebooks that look like a normal candidate's research (NumPy / Pandas / sklearn
only) and each contain 8–12 planted problems: some genuine bugs, some questionable
research choices, some results that should make you suspicious.

**Rules for using them**

1. Open `mock_XX_*.ipynb` blind. Do not open `solutions/` first.
2. Give yourself 45 minutes. Run it top to bottom, then investigate. You do not have to
   finish; the point is the quality of the investigation.
3. Write down, for each problem you find: where it is, why it matters, how you
   confirmed it, how you would fix it.
4. Only then read `solutions/mock_XX_solution.md` and score yourself.
5. Every mock ends with a "Results" cell. Ask yourself whether you believe it, and why.

| Mock | Area |
|---|---|
| 01 | Consumption forecast with lag features (leakage, splits, scaling) |
| 02 | Customer data cleaning and merges (row multiplication, keys, casing) |
| 03 | Weather forecasts and point-in-time correctness (origin vs target time) |
| 04 | Price prediction and a toy trading rule (targets, returns, P&L look-ahead) |
| 05 | Resampling, timezones and daily aggregation (DST, label/closed, NaN sums) |
| 06 | Groupby / pivot reporting (size vs count, dtypes, dropped groups) |
| 07 | Classification pipeline on meters (target encoding, imbalance, thresholds) |
| 08 | NumPy and statistics (ddof, broadcasting, spurious correlation, tests) |
| 09 | Walk-forward evaluation and hyperparameter search (gaps, off-by-one, test peeking) |
| 10 | Residual analysis and reporting (in-sample vs OOS, MAPE, post-hoc outlier removal) |
| 11 | Battery arbitrage LP (objective sign, efficiency side, units, perfect foresight) |
| 12 | Retail baseload hedge ratio (levels vs surprises, look-ahead forward, VaR tail and sign) |
| 13 | Day-ahead volume under imbalance prices (critical fractile, cost-function sides, tuning on test) |
| 14 | UK settlement periods and DST (local vs UTC, mean vs sum of kWh, units, coincident peak) |
| 15 | Multi-meter panel forecast (shift without groupby, forward-shifted window, pooled R², no baseline) |
| 16 | Wind fleet generation and forecast error (power curve, undocumented units, self-made forecast, tz join, capture price) |
| 17 | Customer insights with no code bugs (Simpson, regression to the mean, base rate, pseudo-replication) |
| 18 | Mostly-correct forecast notebook: two real issues, eleven red herrings (do not cry wolf) |
