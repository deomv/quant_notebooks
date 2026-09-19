"""Exercises for 03_scipy/01_stats.ipynb"""
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
        prompt="For a normal distribution with mean 100 and standard deviation 15, compute the probability that a value is **below 130**.",
        setup="",
        solution="answer = stats.norm(loc=100, scale=15).cdf(130)",
        hint="Build the distribution object stats.norm(loc, scale); the cumulative probability is .cdf(x).",
    ),
    dict(
        prompt="For the same normal distribution (mean 100, sd 15), find the value below which **95%** of the distribution lies.",
        setup="",
        solution="answer = stats.norm(loc=100, scale=15).ppf(0.95)",
        hint=".ppf is the inverse of .cdf: give it a probability, get back the value.",
    ),
    dict(
        prompt="Compute the sample mean, the sample standard deviation (divide by n−1) and the skewness of `x`. Assign them to `mean`, `sd`, `skew`.",
        setup="x = np.array([2.0, 4.0, 4.0, 5.0, 9.0, 12.0])",
        solution="mean = x.mean()\nsd = x.std(ddof=1)\nskew = stats.skew(x)",
        hint="np.std uses ddof=0 by default; pass ddof=1. scipy.stats.skew gives the skewness.",
        answer_var=["mean", "sd", "skew"],
    ),
    dict(
        prompt="Compute the Welch t-statistic **by hand** for the difference `b.mean() - a.mean()`: standard error of each mean is sd/sqrt(n) with ddof=1, the SE of the difference is sqrt(se_a² + se_b²). Assign the t statistic to `answer`.",
        setup="a = np.array([10.0, 12.0, 11.0, 13.0, 9.0])\nb = np.array([14.0, 15.0, 13.0, 16.0, 12.0])",
        solution="se_a = a.std(ddof=1) / np.sqrt(len(a))\nse_b = b.std(ddof=1) / np.sqrt(len(b))\nanswer = (b.mean() - a.mean()) / np.sqrt(se_a**2 + se_b**2)",
        hint="Three lines: se_a, se_b, then (b.mean() - a.mean()) / sqrt(se_a**2 + se_b**2). It should be exactly 3.0.",
    ),
    dict(
        prompt="Now get the same comparison from scipy: run Welch's two-sample t-test of `b` against `a` (unequal variances) and assign the **two-sided p-value** to `answer`.",
        setup="a = np.array([10.0, 12.0, 11.0, 13.0, 9.0])\nb = np.array([14.0, 15.0, 13.0, 16.0, 12.0])",
        solution="answer = stats.ttest_ind(b, a, equal_var=False).pvalue",
        hint="stats.ttest_ind(b, a, equal_var=False) returns an object with .statistic and .pvalue.",
    ),
    dict(
        prompt="Compute the Pearson correlation and the Spearman (rank) correlation between `u` and `v`. Assign them to `pearson` and `spearman` (the coefficients only, not the p-values).",
        setup="u = np.array([1.0, 2.0, 3.0, 4.0, 5.0, 6.0])\nv = np.array([1.0, 2.0, 3.0, 4.0, 5.0, 60.0])",
        solution="pearson = stats.pearsonr(u, v)[0]\nspearman = stats.spearmanr(u, v)[0]",
        hint="stats.pearsonr and stats.spearmanr each return (coefficient, pvalue). The outlier 60 makes them differ a lot.",
        answer_var=["pearson", "spearman"],
    ),
    dict(
        prompt="Fit a straight line y = slope·x + intercept to `x`, `y` with `stats.linregress` and assign the slope to `slope` and the intercept to `intercept`.",
        setup="x = np.array([0.0, 1.0, 2.0, 3.0, 4.0])\ny = np.array([1.0, 3.2, 4.8, 7.1, 9.0])",
        solution="res = stats.linregress(x, y)\nslope = res.slope\nintercept = res.intercept",
        hint="stats.linregress(x, y) returns an object with .slope, .intercept, .stderr.",
        answer_var=["slope", "intercept"],
    ),
    dict(
        prompt="Build a 95% confidence interval for the mean of `x` **by hand** using the t distribution: mean ± t_crit · sd/sqrt(n), with sd using ddof=1 and t_crit = the 0.975 quantile of t with n−1 degrees of freedom. Assign the lower and upper bounds to `lower` and `upper`.",
        setup="x = np.array([2.0, 4.0, 4.0, 5.0, 9.0, 12.0])",
        solution="n = len(x)\nse = x.std(ddof=1) / np.sqrt(n)\nt_crit = stats.t.ppf(0.975, df=n - 1)\nlower = x.mean() - t_crit * se\nupper = x.mean() + t_crit * se",
        hint="se = sd/sqrt(n); t_crit = stats.t.ppf(0.975, df=n-1); then mean ± t_crit*se.",
        answer_var=["lower", "upper"],
    ),
    dict(
        prompt="Compute the lag-1 autocorrelation of `x` **by hand** using the standard estimator: centre `x` with its overall mean, then divide the sum of products of neighbouring centred values by the total sum of squared centred values.",
        setup="x = np.array([1.0, 3.0, 5.0, 7.0, 5.0, 3.0])",
        solution="xc = x - x.mean()\nanswer = (xc[1:] * xc[:-1]).sum() / (xc**2).sum()",
        hint="xc = x - x.mean(); numerator = (xc[1:] * xc[:-1]).sum(); denominator = (xc**2).sum().",
    ),
    dict(
        prompt="Run the Mann-Whitney U test of `a` against `b` (two-sided) and assign the U statistic to `answer`.",
        setup="a = np.array([10.0, 12.0, 11.0, 13.0, 9.0])\nb = np.array([14.0, 15.0, 13.0, 16.0, 12.0])",
        solution="answer = stats.mannwhitneyu(a, b, alternative='two-sided').statistic",
        hint="stats.mannwhitneyu(a, b, alternative='two-sided').statistic",
    ),
    dict(
        prompt="You tested 8 features and got the p-values in `pvals`. With a family-wise error rate of 5%, how many of them are significant after a **Bonferroni** correction (threshold 0.05 / 8)? Assign the count to `answer` as an integer.",
        setup="pvals = np.array([0.001, 0.004, 0.010, 0.020, 0.030, 0.045, 0.200, 0.700])",
        solution="answer = int((pvals < 0.05 / len(pvals)).sum())",
        hint="Bonferroni threshold is alpha / number_of_tests. Count how many p-values are below it.",
    ),
    dict(
        prompt="Apply the Benjamini–Hochberg correction to the same `pvals` with `stats.false_discovery_control` and assign the **adjusted** p-values array to `answer`.",
        setup="pvals = np.array([0.001, 0.004, 0.010, 0.020, 0.030, 0.045, 0.200, 0.700])",
        solution="answer = stats.false_discovery_control(pvals)",
        hint="stats.false_discovery_control(pvals) returns the BH-adjusted p-values (default method='bh').",
    ),
]
