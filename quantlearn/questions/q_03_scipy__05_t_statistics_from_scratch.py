"""Exercises for 03_scipy/05_t_statistics_from_scratch.ipynb"""
import numpy as np
import pandas as pd
from scipy import stats

COMMON_SETUP = """
import numpy as np
import pandas as pd
from scipy import stats
"""

INTRO = """
Every task can be done with the formulas from the cheat sheet; use SciPy only to check.
Two-sided p: `2 * stats.t.sf(abs(t), df)`; one-sided: `stats.t.sf(t, df)`; critical value: `stats.t.ppf(1 - alpha/2, df)`.
"""

_A = "a = np.array([1.2, 0.4, 2.1, -0.3, 1.5, 0.9, 1.8, 0.2])"
_AB = "a = np.array([1.2, 0.4, 2.1, -0.3, 1.5, 0.9, 1.8, 0.2])\nb = np.array([0.1, -0.5, 0.8, 0.3, -0.2, 0.6, 0.0, -0.4, 0.5, 0.2])"

QUESTIONS = [
    dict(
        prompt="One-sample t-test of $H_0: \\mu = 0$ on `a`: compute the t-statistic by hand ($\\bar a / (s/\\sqrt n)$, $s$ with ddof=1). Assign the number.",
        setup=_A,
        solution="answer = a.mean() / (a.std(ddof=1) / np.sqrt(len(a)))",
        hint="stats.ttest_1samp(a, 0).statistic must match.",
        tol=1e-9,
    ),
    dict(
        prompt="For that t-statistic compute the two-sided p-value `p_two` and the one-sided p-value for $H_1: \\mu > 0$, `p_one`, using the t distribution with $n-1$ df.",
        setup=_A,
        solution="t = a.mean() / (a.std(ddof=1) / np.sqrt(len(a)))\np_two = 2 * stats.t.sf(abs(t), df=len(a) - 1)\np_one = stats.t.sf(t, df=len(a) - 1)",
        hint="sf = survival function = 1 - cdf. Two-sided doubles the tail.",
        answer_var=["p_two", "p_one"],
        tol=1e-9,
    ),
    dict(
        prompt="What is the two-sided 5% critical value of the t distribution with 7 degrees of freedom, and with 1000? Assign `crit7` and `crit1000`.",
        setup="",
        solution="crit7 = stats.t.ppf(0.975, df=7)\ncrit1000 = stats.t.ppf(0.975, df=1000)",
        hint="ppf(0.975) for two-sided 5%. The second one should be close to 1.96.",
        answer_var=["crit7", "crit1000"],
        tol=1e-6,
    ),
    dict(
        prompt="Build 20,000 draws of $t_5$ from its parts: `Z = rng.normal(0, 1, 20000)` then `V = rng.chisquare(5, 20000)` (exactly these two calls, in this order), and compute $t = Z/\\sqrt{V/5}$. Assign the fraction of $|t| > 1.96$ to `frac` and the theoretical value $2\\,P(T_5 > 1.96)$ to `theory`.",
        setup="rng = np.random.default_rng(1)",
        solution="Z = rng.normal(0, 1, 20000)\nV = rng.chisquare(5, 20000)\nt = Z / np.sqrt(V / 5)\nfrac = (np.abs(t) > 1.96).mean()\ntheory = 2 * stats.t.sf(1.96, df=5)",
        hint="Both should be around 0.107, well above 0.05.",
        answer_var=["frac", "theory"],
        tol=1e-9,
    ),
    dict(
        prompt="Two-sample **pooled** t-test of equal means for `a` vs `b`: compute the pooled variance $s_p^2$, the SE $s_p\\sqrt{1/n_a + 1/n_b}$ and the t-statistic. Assign `sp2` and `t_pooled`.",
        setup=_AB,
        solution="na, nb = len(a), len(b)\nsp2 = ((na - 1) * a.var(ddof=1) + (nb - 1) * b.var(ddof=1)) / (na + nb - 2)\nt_pooled = (a.mean() - b.mean()) / np.sqrt(sp2 * (1 / na + 1 / nb))",
        hint="stats.ttest_ind(a, b).statistic checks t_pooled.",
        answer_var=["sp2", "t_pooled"],
        tol=1e-9,
    ),
    dict(
        prompt="Now **Welch**: SE $= \\sqrt{s_a^2/n_a + s_b^2/n_b}$ and the Welch–Satterthwaite degrees of freedom. Assign `t_welch` and `df_welch`.",
        setup=_AB,
        solution="va, vb = a.var(ddof=1) / len(a), b.var(ddof=1) / len(b)\nt_welch = (a.mean() - b.mean()) / np.sqrt(va + vb)\ndf_welch = (va + vb) ** 2 / (va ** 2 / (len(a) - 1) + vb ** 2 / (len(b) - 1))",
        hint="stats.ttest_ind(a, b, equal_var=False) returns both statistic and df.",
        answer_var=["t_welch", "df_welch"],
        tol=1e-6,
    ),
    dict(
        prompt="Two models' daily absolute errors `e1`, `e2` on the same 8 days. Compute the **paired** t-statistic on the differences `e1 - e2`, and the **unpaired** Welch t-statistic. Assign `t_paired` and `t_unpaired`.",
        setup="e1 = np.array([3.1, 1.2, 4.5, 2.2, 5.0, 1.8, 3.6, 2.9])\ne2 = np.array([2.8, 1.0, 4.1, 2.0, 4.6, 1.7, 3.2, 2.6])",
        solution="d = e1 - e2\nt_paired = d.mean() / (d.std(ddof=1) / np.sqrt(len(d)))\nt_unpaired = stats.ttest_ind(e1, e2, equal_var=False).statistic",
        hint="Paired = one-sample test on the differences. The paired t should be far larger.",
        answer_var=["t_paired", "t_unpaired"],
        tol=1e-6,
    ),
    dict(
        prompt="For `x`, `y` compute the correlation coefficient and its t-statistic $t = r\\sqrt{n-2}/\\sqrt{1-r^2}$. Assign `r` and `t_r`.",
        setup="x = np.array([0.5, 1.5, 2.0, 3.5, 4.0, 5.5, 6.0, 7.5])\ny = np.array([1.0, 1.2, 2.8, 2.9, 4.1, 4.0, 5.9, 6.2])",
        solution="r = np.corrcoef(x, y)[0, 1]\nt_r = r * np.sqrt(len(x) - 2) / np.sqrt(1 - r ** 2)",
        hint="The same t as stats.linregress(x, y).slope / .stderr.",
        answer_var=["r", "t_r"],
        tol=1e-9,
    ),
    dict(
        prompt="A strategy has an annualised Sharpe ratio of 1.2 measured over 4 years. Assign its t-statistic (`SR * sqrt(years)`) to `t_sr`, and the number of years needed for that Sharpe to reach t = 2 to `years_needed`.",
        setup="sr, years = 1.2, 4",
        solution="t_sr = sr * np.sqrt(years)\nyears_needed = (2 / sr) ** 2",
        hint="t = SR * sqrt(Y) -> Y = (t / SR)^2.",
        answer_var=["t_sr", "years_needed"],
        tol=1e-9,
    ),
    dict(
        prompt="Power by simulation: with `rng`, draw `x = rng.normal(0.2, 1.0, size=(5000, 50))` (exactly once), compute the one-sample t of each row, and assign the fraction that exceeds the one-sided 5% critical value with 49 df to `power`.",
        setup="rng = np.random.default_rng(2)",
        solution="x = rng.normal(0.2, 1.0, size=(5000, 50))\nt = x.mean(axis=1) / (x.std(axis=1, ddof=1) / np.sqrt(50))\npower = (t > stats.t.ppf(0.95, df=49)).mean()",
        hint="Row-wise mean and std (axis=1, ddof=1); the critical value is stats.t.ppf(0.95, 49).",
        tol=1e-9,
    ),
    dict(
        prompt="`y` is an AR(1) series with true mean 0. Assign its naive t-statistic to `t_naive` and its effective sample size $n(1-\\hat\\rho_1)/(1+\\hat\\rho_1)$ (with the standard lag-1 estimator) to `n_eff`.",
        setup="rng = np.random.default_rng(9)\neps = rng.normal(0, 1, 500)\ny = np.zeros(500)\nfor t in range(1, 500):\n    y[t] = 0.7 * y[t - 1] + eps[t]",
        solution="t_naive = y.mean() / (y.std(ddof=1) / np.sqrt(len(y)))\nyc = y - y.mean()\nrho1 = (yc[1:] * yc[:-1]).sum() / (yc ** 2).sum()\nn_eff = len(y) * (1 - rho1) / (1 + rho1)",
        hint="rho1 = sum(yc[1:]*yc[:-1]) / sum(yc**2) with yc centred once.",
        answer_var=["t_naive", "n_eff"],
        tol=1e-6,
    ),
    dict(
        prompt="Real data: daily mean consumption, weekdays vs weekends, 2023 only. Assign the Welch t-statistic to `t_welch` and the difference of means (weekday minus weekend, MWh/h) to `diff`.",
        setup='df = pd.read_csv("../data/hourly_power_clean.csv", parse_dates=["time"]).set_index("time")\nday = df.loc["2023"].resample("D").mean()\ndow = day.index.dayofweek',
        solution='wd = day.loc[dow < 5, "consumption_mwh"]\nwe = day.loc[dow >= 5, "consumption_mwh"]\nt_welch = stats.ttest_ind(wd, we, equal_var=False).statistic\ndiff = wd.mean() - we.mean()',
        hint="dayofweek 5 and 6 are the weekend. equal_var=False for Welch.",
        answer_var=["t_welch", "diff"],
        tol=1e-4,
    ),
    dict(
        prompt="Real data, paired: per-day mean absolute error of the 'same hour yesterday' forecast (`shift(24)`) vs 'same hour last week' (`shift(168)`) for hourly consumption in 2023. Assign the paired t-statistic (lag24 minus lag168) to `t_paired`.",
        setup='df = pd.read_csv("../data/hourly_power_clean.csv", parse_dates=["time"]).set_index("time")\ncons = df["consumption_mwh"]',
        solution='e24 = (cons - cons.shift(24)).abs().loc["2023"].resample("D").mean()\ne168 = (cons - cons.shift(168)).abs().loc["2023"].resample("D").mean()\nt_paired = stats.ttest_rel(e24, e168).statistic',
        hint="Build both daily MAE series on the full data, then restrict to 2023 and resample('D').mean(); stats.ttest_rel on the two aligned series.",
        tol=1e-4,
    ),
    dict(
        prompt="Real data, correlation with the confounder removed: for daily means in 2023, subtract each ISO week's mean from wind and from price, then assign the correlation of the demeaned series to `r_dm` and its t-statistic to `t_dm`.",
        setup='df = pd.read_csv("../data/hourly_power_clean.csv", parse_dates=["time"]).set_index("time")\nday = df.loc["2023"].resample("D").mean()\nkey = [day.index.isocalendar().year, day.index.isocalendar().week]',
        solution='w = day["wind_ms"] - day["wind_ms"].groupby(key).transform("mean")\np = day["price_eur_mwh"] - day["price_eur_mwh"].groupby(key).transform("mean")\nr_dm = np.corrcoef(w, p)[0, 1]\nt_dm = r_dm * np.sqrt(len(w) - 2) / np.sqrt(1 - r_dm ** 2)',
        hint="groupby(key).transform('mean') gives each day its week's mean; subtract, then the correlation formula.",
        answer_var=["r_dm", "t_dm"],
        tol=1e-4,
    ),
]
