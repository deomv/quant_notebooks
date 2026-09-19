"""Exercises for 03_scipy/03_sampling_distributions_standard_errors_bootstrap.ipynb"""
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
        prompt="For the sample `x`, compute the sample mean, the sample standard deviation (ddof=1) and the **standard error of the mean** sd/sqrt(n). Assign them to `mean`, `sd`, `se`.",
        setup="x = np.array([12.0, 15.0, 9.0, 14.0, 10.0, 12.0, 11.0, 13.0])",
        solution="mean = x.mean()\nsd = x.std(ddof=1)\nse = sd / np.sqrt(len(x))",
        hint="sd with ddof=1; se = sd / sqrt(n). sd describes the data, se describes the estimate of the mean.",
        answer_var=["mean", "sd", "se"],
    ),
    dict(
        prompt="If the population sd is 20, how many observations `n` do you need for the standard error of the mean to be at most 2? Assign the smallest such integer to `answer`.",
        setup="",
        solution="answer = int(np.ceil((20 / 2) ** 2))",
        hint="se = sd / sqrt(n) ≤ 2 → n ≥ (sd / 2)². Round up.",
    ),
    dict(
        prompt="Simulate the sampling distribution of the mean: draw a `(2000, 5)` array of normal(loc=50, scale=10) values with `rng.normal(50, 10, size=(2000, 5))`, take the mean of each row, and assign the **standard deviation (ddof=1) of the 2000 row means** to `answer`. It should be close to 10/sqrt(5).",
        setup="rng = np.random.default_rng(42)",
        solution="samples = rng.normal(50, 10, size=(2000, 5))\nanswer = samples.mean(axis=1).std(ddof=1)",
        hint="samples.mean(axis=1) gives one mean per row; then .std(ddof=1). Use rng exactly once, with size=(2000, 5).",
        tol=1e-9,
    ),
    dict(
        prompt="Build a 95% **z**-interval for the mean of `x`: mean ± 1.96 · se (se = sd/sqrt(n), sd with ddof=1). Assign `lower` and `upper`.",
        setup="x = np.array([12.0, 15.0, 9.0, 14.0, 10.0, 12.0, 11.0, 13.0])",
        solution="se = x.std(ddof=1) / np.sqrt(len(x))\nlower = x.mean() - 1.96 * se\nupper = x.mean() + 1.96 * se",
        hint="Three lines: se, lower, upper.",
        answer_var=["lower", "upper"],
    ),
    dict(
        prompt="Same interval but with the **t** critical value for n−1 degrees of freedom instead of 1.96 (use `stats.t.ppf(0.975, df=n-1)`). Assign `lower_t` and `upper_t`. Notice they are wider.",
        setup="x = np.array([12.0, 15.0, 9.0, 14.0, 10.0, 12.0, 11.0, 13.0])",
        solution="n = len(x)\nse = x.std(ddof=1) / np.sqrt(n)\nt_crit = stats.t.ppf(0.975, df=n - 1)\nlower_t = x.mean() - t_crit * se\nupper_t = x.mean() + t_crit * se",
        hint="t_crit = stats.t.ppf(0.975, df=n-1) is about 2.36 for n=8, wider than 1.96.",
        answer_var=["lower_t", "upper_t"],
    ),
    dict(
        prompt="Check interval coverage by simulation: `sims = rng.normal(0, 1, size=(1000, 10))` (use rng once, this shape). For each row build the z-interval mean ± 1.96·sd/sqrt(10) with ddof=1, and assign the **fraction of rows whose interval contains the true mean 0** to `answer`.",
        setup="rng = np.random.default_rng(7)",
        solution="sims = rng.normal(0, 1, size=(1000, 10))\nm = sims.mean(axis=1)\nse = sims.std(axis=1, ddof=1) / np.sqrt(10)\ncovered = (m - 1.96 * se <= 0) & (0 <= m + 1.96 * se)\nanswer = covered.mean()",
        hint="Row means and row sds via axis=1; build lower/upper arrays; a boolean array of 'contains 0'; its mean is the coverage.",
        tol=1e-9,
    ),
    dict(
        prompt="One bootstrap of the mean: with `rng`, draw resample indices `idx = rng.integers(0, len(x), size=(500, len(x)))`, compute the mean of `x[idx]` along axis 1 for each resample, and assign the **standard deviation (ddof=1) of the 500 bootstrap means** (the bootstrap SE) to `answer`.",
        setup="x = np.array([12.0, 15.0, 9.0, 14.0, 10.0, 12.0, 11.0, 13.0])\nrng = np.random.default_rng(1)",
        solution="idx = rng.integers(0, len(x), size=(500, len(x)))\nboot_means = x[idx].mean(axis=1)\nanswer = boot_means.std(ddof=1)",
        hint="x[idx] is a (500, 8) array of resampled values; mean(axis=1), then std(ddof=1).",
        tol=1e-9,
    ),
    dict(
        prompt="Using the same bootstrap draw (`idx = rng.integers(0, len(x), size=(500, len(x)))` with `rng` seeded 1), compute the **percentile 95% bootstrap interval** for the mean: the 2.5th and 97.5th percentiles of the bootstrap means (use `np.percentile`). Assign `lower` and `upper`.",
        setup="x = np.array([12.0, 15.0, 9.0, 14.0, 10.0, 12.0, 11.0, 13.0])\nrng = np.random.default_rng(1)",
        solution="idx = rng.integers(0, len(x), size=(500, len(x)))\nboot_means = x[idx].mean(axis=1)\nlower = np.percentile(boot_means, 2.5)\nupper = np.percentile(boot_means, 97.5)",
        hint="np.percentile(boot_means, 2.5) and np.percentile(boot_means, 97.5).",
        answer_var=["lower", "upper"],
        tol=1e-9,
    ),
    dict(
        prompt="Bootstrap a **median**: with `idx = rng.integers(0, len(x), size=(500, len(x)))` (rng seeded 1), take the median of each resample and assign the standard deviation (ddof=1) of the 500 bootstrap medians to `answer`.",
        setup="x = np.array([12.0, 15.0, 9.0, 14.0, 10.0, 12.0, 11.0, 13.0])\nrng = np.random.default_rng(1)",
        solution="idx = rng.integers(0, len(x), size=(500, len(x)))\nanswer = np.median(x[idx], axis=1).std(ddof=1)",
        hint="np.median(x[idx], axis=1) then .std(ddof=1). There is no simple formula for this SE — that is why you bootstrap.",
        tol=1e-9,
    ),
    dict(
        prompt="A series has lag-1 autocorrelation ρ = 0.9 and n = 17,520 hourly observations. Compute the **effective sample size** n·(1−ρ)/(1+ρ) and assign it to `answer` (a float).",
        setup="n = 17520\nrho = 0.9",
        solution="answer = n * (1 - rho) / (1 + rho)",
        hint="n * (1 - rho) / (1 + rho); with rho = 0.9 it is about 5% of n.",
    ),
    dict(
        prompt="The iid standard error of the mean of `y` is sd/sqrt(n). Compute the **autocorrelation-adjusted** SE sd/sqrt(n_eff) where n_eff = n·(1−ρ)/(1+ρ) and ρ is the lag-1 autocorrelation of `y` estimated with `pd.Series(y).autocorr(1)`. Assign `se_iid` and `se_adj` (sd with ddof=1).",
        setup="y = np.array([1.0, 1.5, 2.0, 2.4, 2.6, 2.5, 2.2, 1.8, 1.4, 1.1])",
        solution="n = len(y)\nsd = y.std(ddof=1)\nrho = pd.Series(y).autocorr(1)\nn_eff = n * (1 - rho) / (1 + rho)\nse_iid = sd / np.sqrt(n)\nse_adj = sd / np.sqrt(n_eff)",
        hint="rho = pd.Series(y).autocorr(1); n_eff = n*(1-rho)/(1+rho); se_adj = sd/sqrt(n_eff).",
        answer_var=["se_iid", "se_adj"],
    ),
    dict(
        prompt="Block bootstrap indices: split the 12 positions 0..11 into 4 consecutive blocks of length 3, then with `rng` (seeded 3) draw 4 block numbers with `rng.integers(0, 4, size=4)` and build the resampled index array by concatenating the chosen blocks in the drawn order. Assign the length-12 integer array to `answer`.",
        setup="n = 12\nblock_len = 3\nrng = np.random.default_rng(3)",
        solution="blocks = np.arange(n).reshape(n // block_len, block_len)\nchosen = rng.integers(0, n // block_len, size=n // block_len)\nanswer = blocks[chosen].ravel()",
        hint="blocks = np.arange(12).reshape(4, 3); chosen = rng.integers(0, 4, size=4); blocks[chosen].ravel().",
    ),
]
