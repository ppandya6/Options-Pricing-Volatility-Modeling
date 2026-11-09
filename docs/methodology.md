# Methodology

## Pricing and inversion

For time to expiry T, spot S, strike K, continuous rate r and dividend yield q,

```text
d1 = [log(S/K) + (r-q+sigma²/2)T] / (sigma sqrt(T))
d2 = d1 - sigma sqrt(T)
C  = S exp(-qT) N(d1) - K exp(-rT) N(d2)
P  = K exp(-rT) N(-d2) - S exp(-qT) N(-d1)
```

At expiry the payoff is intrinsic value. At zero volatility the discounted deterministic payoff is used. Vega and rho are derivatives per unit change (1.0); divide by 100 for a one percentage-point change. Theta is per calendar year of elapsed time. IV uses Brent's bracketed root solver between zero and `max_vol` (default 10, i.e. 1000% annual volatility). IV is undefined at expiry and at the infinite-volatility upper price bound; the lower price bound maps to zero. Near-boundary IV can be ill-conditioned even when a numerical root exists.

## Volatility and evaluation

Historical volatility is sample standard deviation of daily log returns times sqrt(252). EWMA assumes zero mean and recursively updates variance with decay 0.94, initialized from the first window's mean squared returns. Annualization assumes approximately 252 trading observations per year. The realized forward target uses a zero-mean second moment rather than a demeaned sample standard deviation; the estimators differ slightly in finite samples.

The quote experiment uses x = log(K / [S exp((r-q)T)]). Flat BS fits one volatility. The flexible benchmark fits sigma(x) = exp(a + b x + c x²), clipped to [0.0001, 5], using price-space least squares. Parameter bounds stabilize the descriptive fit. No IV target from a test row is used to fit either model. Quotes can span maturities, but this simple smile has no independent term-structure parameters; analyze individual expiries when studying smile shape.

## Simulation

GBM uses the exact risk-neutral terminal lognormal distribution. Independent normal draws support the usual sample standard error, `sample_std(discounted_payoff)/sqrt(paths)`. Approximate 95% intervals use 1.96 standard errors; they are asymptotic, not finite-sample guarantees.

Heston dynamics under the risk-neutral measure are:

```text
dS/S = (r-q)dt + sqrt(v)dW_s
dv   = kappa(theta-v)dt + xi sqrt(v)dW_v
corr(dW_s,dW_v) = rho
```

The log-stock Euler update and full-truncation variance update use the old positive part of variance in both drift and diffusion. Auxiliary variance can become negative; subsequent steps truncate it for coefficients, not by reflecting it. Memory scales with path count, not path count times steps. Runtime scales with paths times steps. The sufficient Feller condition `2*kappa*theta >= xi²` prevents zero variance in the continuous CIR process but is not necessary for this numerical scheme to run.

The demo evaluates two time grids with the same seed. Those estimates are not constructed as a coupled multilevel estimator, and their difference has no supplied significance test. Refine steps and increase paths independently; discretization bias is excluded from reported sampling intervals. Heston is not automatically more accurate than BS without calibration and empirical validation.

## References

- Black, F. & Scholes, M. (1973). The Pricing of Options and Corporate Liabilities. *Journal of Political Economy*, 81(3), 637–654. DOI: 10.1086/260062.
- Heston, S. (1993). A Closed-Form Solution for Options with Stochastic Volatility with Applications to Bond and Currency Options. *Review of Financial Studies*, 6(2), 327–343. DOI: 10.1093/rfs/6.2.327.
- Lord, R., Koekkoek, R. & van Dijk, D. (2010). A Comparison of Biased Simulation Schemes for Stochastic Volatility Models. *Quantitative Finance*, 10(2), 177–194. DOI: 10.1080/14697680802392496.
- Glasserman, P. (2004). *Monte Carlo Methods in Financial Engineering*. Springer. DOI: 10.1007/978-0-387-21617-1.
