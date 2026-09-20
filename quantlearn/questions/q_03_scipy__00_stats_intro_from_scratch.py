"""Exercises for 03_scipy/00_stats_intro_from_scratch.ipynb

Compute each quantity by hand from the formula first; the library call is for checking.
"""
import numpy as np
import pandas as pd
from scipy import stats

COMMON_SETUP = """
import numpy as np
import pandas as pd
from scipy import stats
"""

INTRO = """
Formulas you will need (from the cheat sheet): $s^2 = \\sum(x_i-\\bar x)^2/(n-1)$,
$\\text{SE} = s/\\sqrt n$, $t = \\bar x/\\text{SE}$, $\\rho_k = \\sum(x_t-\\bar x)(x_{t-k}-\\bar x)/\\sum(x_t-\\bar x)^2$,
$n_{\\text{eff}} = n(1-\\rho)/(1+\\rho)$, $\\hat b = S_{xy}/S_{xx}$, $\\text{SE}(\\hat b) = \\hat\\sigma/\\sqrt{S_{xx}}$ with $\\hat\\sigma^2 = \\text{RSS}/(n-2)$.
"""

_X = "x = np.array([3.0, -1.0, 2.0, 5.0, 0.0, 3.0])"
_REG = "xr = np.array([0.0, 1.0, 2.0, 3.0, 4.0])\nyr = np.array([1.0, 2.9, 5.2, 6.8, 9.1])"

QUESTIONS = [
    dict(
        prompt="Compute the sample variance of `x` with the $n-1$ denominator, by hand (deviations, squares, sum, divide). Assign the number.",
        setup=_X,
        solution="answer = ((x - x.mean()) ** 2).sum() / (len(x) - 1)",
        hint="Deviations from the mean, squared, summed, divided by n-1. np.var(x, ddof=1) checks it.",
    ),
    dict(
        prompt="Compute the standard error of the mean of `x`, $s/\\sqrt n$ (with $s$ using $n-1$). Assign the number.",
        setup=_X,
        solution="answer = x.std(ddof=1) / np.sqrt(len(x))",
        hint="stats.sem(x) gives the same; the point is to write s / sqrt(n) yourself.",
    ),
    dict(
        prompt="Compute the t-statistic for $H_0: \\mu = 0$ on `x`, and the **two-sided** p-value using the t distribution with $n-1$ degrees of freedom. Assign `t_stat` and `p_two`.",
        setup=_X,
        solution="t_stat = x.mean() / (x.std(ddof=1) / np.sqrt(len(x)))\np_two = 2 * stats.t.sf(abs(t_stat), df=len(x) - 1)",
        hint="t = xbar / SE. p = 2 * stats.t.sf(|t|, df=n-1). Compare with stats.ttest_1samp(x, 0).",
        answer_var=["t_stat", "p_two"],
        tol=1e-6,
    ),
    dict(
        prompt="Compute the **one-sided** p-value for $H_0: \\mu \\le 0$ vs $H_1: \\mu > 0$ on `x`. Assign the number.",
        setup=_X,
        solution="t_stat = x.mean() / (x.std(ddof=1) / np.sqrt(len(x)))\nanswer = stats.t.sf(t_stat, df=len(x) - 1)",
        hint="One tail only: stats.t.sf(t, df), no factor 2 and no abs().",
        tol=1e-6,
    ),
    dict(
        prompt="Compute the 95% confidence interval for the mean of `x`: $\\bar x \\pm t_{n-1,0.975}\\, s/\\sqrt n$. Assign `ci_low` and `ci_high`.",
        setup=_X,
        solution="se = x.std(ddof=1) / np.sqrt(len(x))\ncrit = stats.t.ppf(0.975, df=len(x) - 1)\nci_low = x.mean() - crit * se\nci_high = x.mean() + crit * se",
        hint="stats.t.ppf(0.975, df=n-1) is the critical value (2.571 for df=5).",
        answer_var=["ci_low", "ci_high"],
        tol=1e-6,
    ),
    dict(
        prompt="A sample of 400 daily returns has mean 0.05 and sd 1.2. What is the t-statistic for a zero mean? Assign the number.",
        setup="mean, sd, n = 0.05, 1.2, 400",
        solution="answer = mean / (sd / np.sqrt(n))",
        hint="SE = sd / sqrt(n) = 0.06. t = 0.05 / 0.06.",
        tol=1e-6,
    ),
    dict(
        prompt="Simulate the sampling distribution: draw `sims = rng.normal(0, 3, size=(5000, 25))` (exactly this call, once), take the mean of each row, and assign the standard deviation (ddof=1) of those 5000 means to `sd_means`, and the formula value $\\sigma/\\sqrt n$ to `se_formula`.",
        setup="rng = np.random.default_rng(3)",
        solution="sims = rng.normal(0, 3, size=(5000, 25))\nsd_means = sims.mean(axis=1).std(ddof=1)\nse_formula = 3 / np.sqrt(25)",
        hint="sims.mean(axis=1) is one mean per row. The two numbers should be close to 0.6.",
        answer_var=["sd_means", "se_formula"],
        tol=1e-9,
    ),
    dict(
        prompt="Compute the lag-1 autocorrelation of `z` with the standard estimator $\\rho_1 = \\sum_{t\\ge2}(z_t-\\bar z)(z_{t-1}-\\bar z) / \\sum_t (z_t-\\bar z)^2$ (one overall mean in both numerator and denominator). Assign the number.",
        setup="z = np.array([4.0, 6.0, 5.0, 8.0, 7.0, 9.0, 6.0, 8.0])",
        solution="zc = z - z.mean()\nanswer = (zc[1:] * zc[:-1]).sum() / (zc ** 2).sum()",
        hint="Centre once, multiply neighbouring deviations, sum, divide by the total sum of squares. This is NOT pd.Series(z).autocorr(1).",
        tol=1e-9,
    ),
    dict(
        prompt="A series has n = 5000 observations and lag-1 autocorrelation 0.9. Assign the effective sample size $n_{\\text{eff}} = n(1-\\rho)/(1+\\rho)$ to `n_eff`, and the factor by which the naive standard error of the mean must be multiplied, $\\sqrt{(1+\\rho)/(1-\\rho)}$, to `factor`.",
        setup="n, rho = 5000, 0.9",
        solution="n_eff = n * (1 - rho) / (1 + rho)\nfactor = np.sqrt((1 + rho) / (1 - rho))",
        hint="n_eff ≈ 263, factor ≈ 4.36.",
        answer_var=["n_eff", "factor"],
        tol=1e-6,
    ),
    dict(
        prompt="`y` is an AR(1) series with true mean 0. Its naive t-statistic is large. Compute the corrected t-statistic: naive $t$ divided by $\\sqrt{(1+\\hat\\rho_1)/(1-\\hat\\rho_1)}$ where $\\hat\\rho_1$ is the standard lag-1 estimator. Assign `t_naive` and `t_corrected`.",
        setup="""rng = np.random.default_rng(5)
eps = rng.normal(0, 1, 400)
y = np.zeros(400)
for t in range(1, 400):
    y[t] = 0.85 * y[t - 1] + eps[t]""",
        solution="""t_naive = y.mean() / (y.std(ddof=1) / np.sqrt(len(y)))
yc = y - y.mean()
rho1 = (yc[1:] * yc[:-1]).sum() / (yc ** 2).sum()
t_corrected = t_naive / np.sqrt((1 + rho1) / (1 - rho1))""",
        hint="First the naive t, then rho1 from the formula in the previous task, then divide.",
        answer_var=["t_naive", "t_corrected"],
        tol=1e-6,
    ),
    dict(
        prompt="Compute the Newey–West standard error of the mean of `y` with $L = 10$ Bartlett lags: $\\widehat{\\text{Var}} = \\frac{1}{n}\\left(\\hat\\gamma_0 + 2\\sum_{k=1}^{L}(1-\\frac{k}{L+1})\\hat\\gamma_k\\right)$ with $\\hat\\gamma_k = \\frac{1}{n}\\sum_{t>k}(y_t-\\bar y)(y_{t-k}-\\bar y)$. Assign the standard error (square root).",
        setup="""rng = np.random.default_rng(5)
eps = rng.normal(0, 1, 400)
y = np.zeros(400)
for t in range(1, 400):
    y[t] = 0.85 * y[t - 1] + eps[t]""",
        solution="""n = len(y)
yc = y - y.mean()
L = 10
gamma = [(yc[k:] * yc[:-k]).sum() / n if k > 0 else (yc ** 2).sum() / n for k in range(L + 1)]
weights = [1 - k / (L + 1) for k in range(L + 1)]
var_sum = gamma[0] + 2 * sum(w * g for w, g in zip(weights[1:], gamma[1:]))
answer = np.sqrt(var_sum / n)""",
        hint="gamma_0 is the (ddof=0) variance; weights are 1 - k/(L+1); then sqrt(var_sum / n).",
        tol=1e-8,
    ),
    dict(
        prompt="OLS by hand on `xr`, `yr`: compute $S_{xy}$, $S_{xx}$, the slope $\\hat b = S_{xy}/S_{xx}$ and the intercept $\\hat a = \\bar y - \\hat b \\bar x$. Assign `b_hat` and `a_hat`.",
        setup=_REG,
        solution="Sxy = ((xr - xr.mean()) * (yr - yr.mean())).sum()\nSxx = ((xr - xr.mean()) ** 2).sum()\nb_hat = Sxy / Sxx\na_hat = yr.mean() - b_hat * xr.mean()",
        hint="np.polyfit(xr, yr, 1) checks it, but write the sums yourself.",
        answer_var=["b_hat", "a_hat"],
        tol=1e-9,
    ),
    dict(
        prompt="Compute the standard error of the slope: residuals $e = y - \\hat a - \\hat b x$, $\\hat\\sigma^2 = \\sum e^2/(n-2)$, $\\text{SE}(\\hat b) = \\hat\\sigma/\\sqrt{S_{xx}}$. Assign the number.",
        setup=_REG,
        solution="Sxx = ((xr - xr.mean()) ** 2).sum()\nb_hat = ((xr - xr.mean()) * (yr - yr.mean())).sum() / Sxx\na_hat = yr.mean() - b_hat * xr.mean()\ne = yr - a_hat - b_hat * xr\nsigma2 = (e ** 2).sum() / (len(xr) - 2)\nanswer = np.sqrt(sigma2 / Sxx)",
        hint="stats.linregress(xr, yr).stderr gives the same number.",
        tol=1e-9,
    ),
    dict(
        prompt="Compute the 95% confidence interval for the slope: $\\hat b \\pm t_{n-2,0.975}\\,\\text{SE}(\\hat b)$. Assign `ci_low` and `ci_high`.",
        setup=_REG,
        solution="r = stats.linregress(xr, yr)\ncrit = stats.t.ppf(0.975, df=len(xr) - 2)\nci_low = r.slope - crit * r.stderr\nci_high = r.slope + crit * r.stderr",
        hint="Degrees of freedom are n-2 for a simple regression (two estimated parameters).",
        answer_var=["ci_low", "ci_high"],
        tol=1e-9,
    ),
    dict(
        prompt="Multiple regression by matrix algebra: `X` has a column of ones, `x1` and `x2`. Compute $\\hat\\beta = (X^\\top X)^{-1}X^\\top y$ and the standard errors $\\sqrt{\\text{diag}(\\hat\\sigma^2 (X^\\top X)^{-1})}$ with $\\hat\\sigma^2 = \\text{RSS}/(n-3)$. Assign `beta` (array of 3) and `se` (array of 3).",
        setup="""x1 = np.array([1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0])
x2 = np.array([0.0, 1.0, 0.0, 1.0, 0.0, 1.0, 0.0, 1.0])
y = np.array([3.1, 6.2, 7.1, 10.3, 11.0, 13.9, 15.2, 18.1])
X = np.column_stack([np.ones(8), x1, x2])""",
        solution="""XtX_inv = np.linalg.inv(X.T @ X)
beta = XtX_inv @ X.T @ y
resid = y - X @ beta
sigma2 = (resid ** 2).sum() / (len(y) - 3)
se = np.sqrt(np.diag(sigma2 * XtX_inv))""",
        hint="np.linalg.inv(X.T @ X) @ X.T @ y for beta; the SEs are the square roots of the diagonal of sigma2 * inv(X'X).",
        answer_var=["beta", "se"],
        tol=1e-8,
    ),
    dict(
        prompt="Real data: hourly price changes for 2023 from `../data/hourly_power_clean.csv`. Assign the naive t-statistic for a zero mean to `t_naive` and the standard lag-1 autocorrelation of the changes to `rho1`.",
        setup="""df = pd.read_csv("../data/hourly_power_clean.csv", parse_dates=["time"]).set_index("time")
dp = df.loc["2023", "price_eur_mwh"].diff().dropna().to_numpy()""",
        solution="""t_naive = dp.mean() / (dp.std(ddof=1) / np.sqrt(len(dp)))
dc = dp - dp.mean()
rho1 = (dc[1:] * dc[:-1]).sum() / (dc ** 2).sum()""",
        hint="Same two formulas as tasks 3 and 8, on 8,759 numbers.",
        answer_var=["t_naive", "rho1"],
        tol=1e-6,
    ),
    dict(
        prompt="Real data: regress 2023 hourly consumption (with its hour-of-day mean removed) on heating degrees $\\max(15 - T, 0)$. Assign the slope to `slope` and the lag-1 autocorrelation of the residuals to `resid_rho1`. (If `resid_rho1` is far above zero, the naive SE of the slope is not to be trusted.)",
        setup="""df = pd.read_csv("../data/hourly_power_clean.csv", parse_dates=["time"]).set_index("time")
h = df.loc["2023"]
hdd = np.clip(15 - h["temp_c"], 0, None).to_numpy()
y = (h["consumption_mwh"] - h["consumption_mwh"].groupby(h.index.hour).transform("mean")).to_numpy()""",
        solution="""r = stats.linregress(hdd, y)
slope = r.slope
e = y - (r.intercept + r.slope * hdd)
ec = e - e.mean()
resid_rho1 = (ec[1:] * ec[:-1]).sum() / (ec ** 2).sum()""",
        hint="stats.linregress for the fit; residuals = y - (intercept + slope * hdd); then the lag-1 formula.",
        answer_var=["slope", "resid_rho1"],
        tol=1e-4,
    ),
]
