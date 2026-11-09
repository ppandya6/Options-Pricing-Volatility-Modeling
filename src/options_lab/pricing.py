"""Dividend-adjusted Black–Scholes for European calls and puts.

Time is in years; rates, yield and volatility are annualized decimals.
"""
from dataclasses import dataclass
import math
from scipy.optimize import brentq
from scipy.special import ndtr


@dataclass(frozen=True)
class Option:
    spot: float
    strike: float
    maturity: float
    rate: float = 0.0
    dividend: float = 0.0
    kind: str = "call"

    def __post_init__(self):
        if not all(math.isfinite(x) for x in
                   (self.spot, self.strike, self.maturity, self.rate, self.dividend)):
            raise ValueError("Option inputs must be finite")
        if self.spot <= 0 or self.strike <= 0 or self.maturity < 0:
            raise ValueError("Spot and strike must be positive; maturity nonnegative")
        if self.kind not in ("call", "put"):
            raise ValueError("kind must be call or put")

    @property
    def sign(self):
        return 1 if self.kind == "call" else -1


def _vol(sigma):
    if not math.isfinite(sigma) or sigma < 0:
        raise ValueError("Volatility must be finite and nonnegative")


def bounds(option):
    s = option.spot * math.exp(-option.dividend * option.maturity)
    k = option.strike * math.exp(-option.rate * option.maturity)
    return max(option.sign * (s - k), 0.0), s if option.kind == "call" else k


def black_scholes(option: Option, sigma: float) -> float:
    _vol(sigma)
    lower, _ = bounds(option)
    if option.maturity == 0 or sigma == 0:
        return lower
    t = option.maturity
    d1 = (math.log(option.spot / option.strike) +
          (option.rate - option.dividend + sigma**2 / 2) * t) / (sigma * math.sqrt(t))
    d2 = d1 - sigma * math.sqrt(t)
    z = option.sign
    return float(z * (option.spot * math.exp(-option.dividend * t) * ndtr(z * d1)
                      - option.strike * math.exp(-option.rate * t) * ndtr(z * d2)))


def greeks(option: Option, sigma: float) -> dict[str, float]:
    """Delta, gamma, vega, calendar theta and rho; vega/rho per unit (1.0)."""
    _vol(sigma)
    if sigma == 0 or option.maturity == 0:
        raise ValueError("Greeks require positive volatility and maturity")
    s, k, t, r, q, z = (option.spot, option.strike, option.maturity,
                         option.rate, option.dividend, option.sign)
    root = math.sqrt(t)
    d1 = (math.log(s / k) + (r - q + sigma**2 / 2) * t) / (sigma * root)
    d2 = d1 - sigma * root
    density = math.exp(-d1*d1/2) / math.sqrt(2*math.pi)
    sq, kr = s * math.exp(-q*t), k * math.exp(-r*t)
    return dict(delta=float(z * math.exp(-q*t) * ndtr(z*d1)),
                gamma=math.exp(-q*t) * density / (s*sigma*root),
                vega=sq*density*root,
                theta=float(-sq*density*sigma/(2*root) - z*r*kr*ndtr(z*d2)
                            + z*q*sq*ndtr(z*d1)),
                rho=float(z*kr*t*ndtr(z*d2)))


def implied_volatility(option: Option, price: float, max_vol: float = 10.0) -> float:
    """Bracketed inversion; no IV exists at expiry or at the upper price bound."""
    if option.maturity == 0:
        raise ValueError("Implied volatility is undefined at expiry")
    if not math.isfinite(price) or not math.isfinite(max_vol) or max_vol <= 0:
        raise ValueError("Price must be finite and max_vol positive")
    lower, upper = bounds(option)
    if price < lower or price >= upper:
        raise ValueError(f"Price must lie in [{lower}, {upper})")
    if price == lower:
        return 0.0
    if black_scholes(option, max_vol) < price:
        raise ValueError("IV exceeds max_vol; increase the search bracket")
    return float(brentq(lambda sigma: black_scholes(option, sigma) - price,
                        0, max_vol, xtol=1e-12))
