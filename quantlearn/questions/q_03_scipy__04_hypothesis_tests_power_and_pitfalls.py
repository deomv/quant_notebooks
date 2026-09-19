"""Exercises for 03_scipy/04_hypothesis_tests_power_and_pitfalls.ipynb"""
import numpy as np
import pandas as pd
from scipy import stats

COMMON_SETUP = """
import numpy as np
import pandas as pd
from scipy import stats
rng = np.random.default_rng(0)
"""

QUESTIONS = [
    dict(
        prompt="A test gave t = 2.3 with 20 degrees of freedom. Compute the **two-sided** p-value with `stats.t.sf` (the survival function, i.e. the upper tail). Assign it to `answer`.",
        setup="t_stat = 2.3\ndf = 20",
        solution="answer = 2 * stats.t.sf(abs(t_stat), df=df)",
        hint="Two-sided = 2 × upper-tail probability: 2 * stats.t.sf(abs(t), df).",
    ),
    dict(
        prompt="Same t and df, but the **one-sided** alternative 'greater than'. Assign the p-value to `answer`.",
        setup="t_stat = 2.3\ndf = 20",
        solution="answer = stats.t.sf(t_stat, df=df)",
        hint="One-sided upper tail is just stats.t.sf(t, df) — half the two-sided value.",
    ),
    dict(
        prompt="Permutation test, one step: `values` holds 6 observations, the first 3 from group A and the last 3 from group B. The observed statistic is mean(B) − mean(A). Apply the permutation `perm` to `values` (i.e. `values[perm]`), split it 3/3 again, and assign the **permuted** statistic mean(B') − mean(A') to `answer`.",
        setup="values = np.array([10.0, 12.0, 11.0, 15.0, 14.0, 16.0])\nperm = np.array([3, 0, 5, 1, 4, 2])",
        solution="shuffled = values[perm]\nanswer = shuffled[3:].mean() - shuffled[:3].mean()",
        hint="shuffled = values[perm]; then shuffled[3:].mean() - shuffled[:3].mean().",
    ),
    dict(
        prompt="Full permutation test: with `rng` (seeded 5), repeat 2000 times: `perm = rng.permutation(6)`, compute the permuted statistic as in the previous task, and count how often its **absolute value** is at least the absolute observed statistic (mean(B) − mean(A) on the original order). Assign the p-value = count / 2000 to `answer`.",
        setup="values = np.array([10.0, 12.0, 11.0, 15.0, 14.0, 16.0])\nrng = np.random.default_rng(5)",
        solution="observed = values[3:].mean() - values[:3].mean()\ncount = 0\nfor _ in range(2000):\n    perm = rng.permutation(6)\n    s = values[perm]\n    stat = s[3:].mean() - s[:3].mean()\n    if abs(stat) >= abs(observed):\n        count += 1\nanswer = count / 2000",
        hint="A for loop calling rng.permutation(6) exactly 2000 times, in order; compare abs(stat) >= abs(observed).",
        tol=1e-9,
    ),
    dict(
        prompt="Type I error by simulation: with `rng` (seeded 11), draw `sims = rng.normal(0, 1, size=(4000, 2, 15))` — 4000 experiments, each with two groups of 15 from the **same** distribution. Run `stats.ttest_ind` on each pair (equal_var=True) and assign the **fraction of p-values below 0.05** to `answer`. It should be close to 0.05.",
        setup="rng = np.random.default_rng(11)",
        solution="sims = rng.normal(0, 1, size=(4000, 2, 15))\np = stats.ttest_ind(sims[:, 0, :], sims[:, 1, :], axis=1).pvalue\nanswer = (p < 0.05).mean()",
        hint="ttest_ind accepts 2-D arrays with axis=1, so one call does all 4000 tests: sims[:, 0, :] vs sims[:, 1, :].",
        tol=1e-9,
    ),
    dict(
        prompt="Power by the normal approximation: for a two-sample test with n = 30 per group, sd = 1, true difference d = 0.5 and alpha = 0.05 (two-sided), the power is approximately `1 − Φ(z_crit − d / sqrt(2/n))` with z_crit = 1.96 and Φ the standard normal cdf. Assign it to `answer`.",
        setup="n = 30\nd = 0.5\nsd = 1.0",
        solution="se = sd * np.sqrt(2 / n)\nanswer = 1 - stats.norm.cdf(1.96 - d / se)",
        hint="se of the difference is sd*sqrt(2/n); power ≈ 1 - stats.norm.cdf(1.96 - d/se).",
    ),
    dict(
        prompt="Paired vs unpaired: `err_a` and `err_b` are the daily errors of two forecast models on the **same** 6 days. Run the paired t-test (`stats.ttest_rel`) and the unpaired one (`stats.ttest_ind`, equal_var=False) and assign the two p-values to `p_paired` and `p_unpaired`.",
        setup="err_a = np.array([5.0, 7.0, 6.0, 9.0, 4.0, 8.0])\nerr_b = np.array([4.5, 6.4, 5.6, 8.3, 3.7, 7.5])",
        solution="p_paired = stats.ttest_rel(err_a, err_b).pvalue\np_unpaired = stats.ttest_ind(err_a, err_b, equal_var=False).pvalue",
        hint="ttest_rel for paired, ttest_ind(equal_var=False) for unpaired. Pairing removes the day-to-day variation the models share.",
        answer_var=["p_paired", "p_unpaired"],
    ),
    dict(
        prompt="Bonferroni: for the 6 p-values in `pvals` at family-wise alpha 0.05, compute the corrected threshold and the boolean array of which tests are significant. Assign the threshold to `threshold` and the boolean array to `significant`.",
        setup="pvals = np.array([0.002, 0.009, 0.011, 0.030, 0.049, 0.300])",
        solution="threshold = 0.05 / len(pvals)\nsignificant = pvals < threshold",
        hint="threshold = 0.05 / 6; significant = pvals < threshold.",
        answer_var=["threshold", "significant"],
    ),
    dict(
        prompt="Benjamini–Hochberg on the same `pvals`: adjust them with `stats.false_discovery_control` and assign the **number** of adjusted p-values below 0.05 to `answer` (an integer).",
        setup="pvals = np.array([0.002, 0.009, 0.011, 0.030, 0.049, 0.300])",
        solution="adj = stats.false_discovery_control(pvals)\nanswer = int((adj < 0.05).sum())",
        hint="stats.false_discovery_control(pvals) then count < 0.05. BH is less strict than Bonferroni.",
    ),
    dict(
        prompt="Effect size: compute Cohen's d for `b` versus `a`, defined as (mean(b) − mean(a)) / pooled_sd, where pooled_sd = sqrt((var_a + var_b) / 2) with variances using ddof=1. Assign it to `answer`.",
        setup="a = np.array([10.0, 12.0, 11.0, 13.0, 9.0])\nb = np.array([14.0, 15.0, 13.0, 16.0, 12.0])",
        solution="pooled_sd = np.sqrt((a.var(ddof=1) + b.var(ddof=1)) / 2)\nanswer = (b.mean() - a.mean()) / pooled_sd",
        hint="pooled_sd = sqrt((a.var(ddof=1) + b.var(ddof=1)) / 2); d = (b.mean() - a.mean()) / pooled_sd.",
    ),
    dict(
        prompt="Newey–West with one lag: for the residuals `e` (mean zero), the iid variance of the mean is γ0/n and the lag-1-adjusted variance is (γ0 + 2·w·γ1)/n with Bartlett weight w = 1/2, where γ0 = mean(e²) and γ1 = mean(e[1:]·e[:-1]). Assign `var_iid` and `var_nw`.",
        setup="e = np.array([1.0, 0.8, 0.5, -0.2, -0.6, -1.0, -0.5, 0.0])",
        solution="n = len(e)\ng0 = np.mean(e**2)\ng1 = np.mean(e[1:] * e[:-1])\nvar_iid = g0 / n\nvar_nw = (g0 + 2 * 0.5 * g1) / n",
        hint="g0 = mean(e**2); g1 = mean(e[1:]*e[:-1]); var_nw = (g0 + 2*0.5*g1)/n.",
        answer_var=["var_iid", "var_nw"],
    ),
    dict(
        prompt="Optional stopping: `pvals_over_time` holds the p-value of a running test after each batch of data (batch 1, 2, ..., 8). An analyst stops and declares success at the **first** batch where p < 0.05. Assign the 1-based batch number at which they stop to `answer` (an integer), or 0 if they never would.",
        setup="pvals_over_time = np.array([0.40, 0.21, 0.09, 0.04, 0.12, 0.20, 0.03, 0.06])",
        solution="below = np.where(pvals_over_time < 0.05)[0]\nanswer = int(below[0] + 1) if len(below) else 0",
        hint="np.where(pvals_over_time < 0.05)[0] gives the positions; the first one plus 1 is the batch number.",
    ),
]
