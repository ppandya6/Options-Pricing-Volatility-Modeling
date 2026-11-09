"""Risk-neutral Monte Carlo with independent paths and sampling uncertainty."""
from dataclasses import dataclass
import math
import numpy as np
from .pricing import Option, _vol


@dataclass(frozen=True)
class MonteCarloResult:
    price: float
    standard_error: float
    ci_low: float
    ci_high: float
    paths: int


@dataclass(frozen=True)
class Heston:
    v0: float = 0.04
    kappa: float = 2.0
    theta: float = 0.04
    vol_of_vol: float = 0.4
    rho: float = -0.7

    def __post_init__(self):
        if not all(math.isfinite(x) for x in
                   (self.v0, self.kappa, self.theta, self.vol_of_vol, self.rho)):
            raise ValueError("Heston parameters must be finite")
        if min(self.v0, self.theta, self.vol_of_vol) < 0 or self.kappa <= 0:
            raise ValueError("Variances and vol_of_vol must be nonnegative; kappa positive")
        if abs(self.rho) > 1:
            raise ValueError("rho must be between -1 and 1")

    @property
    def feller_satisfied(self):
        return 2*self.kappa*self.theta >= self.vol_of_vol**2


def _paths(paths):
    if not isinstance(paths, int) or isinstance(paths, bool) or paths < 2:
        raise ValueError("paths must be an integer >= 2")


def _estimate(option, terminal):
    payoff = np.exp(-option.rate*option.maturity) * np.maximum(
        option.sign * (terminal - option.strike), 0)
    price = float(payoff.mean())
    se = float(payoff.std(ddof=1) / math.sqrt(len(payoff)))
    return MonteCarloResult(price, se, price - 1.96*se, price + 1.96*se, len(payoff))


def gbm_price(option: Option, sigma: float, paths: int = 100_000,
              seed: int = 42) -> MonteCarloResult:
    """Exact terminal GBM simulation; no time-discretization bias."""
    _vol(sigma)
    _paths(paths)
    rng = np.random.default_rng(seed)
    t = option.maturity
    terminal = option.spot * np.exp((option.rate-option.dividend-sigma*sigma/2)*t
                                    + sigma*math.sqrt(t)*rng.standard_normal(paths))
    return _estimate(option, terminal)


def heston_price(option: Option, model: Heston = Heston(), paths: int = 50_000,
                 steps: int = 252, seed: int = 42) -> MonteCarloResult:
    """Log-stock Euler with full-truncation variance Euler.

    Confidence intervals measure sampling error only. Repeat with finer steps
    to investigate discretization bias, especially near zero variance.
    """
    _paths(paths)
    if not isinstance(steps, int) or isinstance(steps, bool) or steps < 1:
        raise ValueError("steps must be a positive integer")
    rng = np.random.default_rng(seed)
    dt = option.maturity / steps
    log_s = np.full(paths, math.log(option.spot))
    variance = np.full(paths, model.v0, dtype=float)
    for _ in range(steps):
        z_s, z_ind = rng.standard_normal((2, paths))
        z_v = model.rho*z_s + math.sqrt(1-model.rho**2)*z_ind
        positive = np.maximum(variance, 0)
        diffusion = np.sqrt(positive*dt)
        log_s += (option.rate-option.dividend-positive/2)*dt + diffusion*z_s
        variance += model.kappa*(model.theta-positive)*dt + model.vol_of_vol*diffusion*z_v
    return _estimate(option, np.exp(log_s))
