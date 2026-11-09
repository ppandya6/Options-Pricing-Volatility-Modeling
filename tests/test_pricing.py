from dataclasses import replace
import numpy as np
import pytest
from options_lab import Option, black_scholes, greeks, implied_volatility


def test_reference_price_and_parity():
    call = Option(100, 100, 1, 0.05)
    assert black_scholes(call, 0.2) == pytest.approx(10.450583572185565)
    put = replace(call, kind="put")
    assert black_scholes(call, 0.2)-black_scholes(put, 0.2) == pytest.approx(100-100*np.exp(-0.05))


@pytest.mark.parametrize("kind", ["call", "put"])
@pytest.mark.parametrize("strike", [80, 100, 120])
@pytest.mark.parametrize("sigma", [0.1, 0.3, 0.8])
def test_iv_roundtrip(kind, strike, sigma):
    option = Option(100, strike, 0.75, -0.01, 0.03, kind)
    assert implied_volatility(option, black_scholes(option, sigma)) == pytest.approx(sigma, abs=1e-8)


@pytest.mark.parametrize("kind", ["call", "put"])
def test_greeks_finite_differences(kind):
    option = Option(100, 105, 0.75, 0.03, 0.01, kind)
    vol, h = 0.24, 1e-3
    p = black_scholes(option, vol)
    up = black_scholes(replace(option, spot=100+h), vol)
    down = black_scholes(replace(option, spot=100-h), vol)
    g = greeks(option, vol)
    assert g['delta'] == pytest.approx((up-down)/(2*h), abs=1e-7)
    assert g['gamma'] == pytest.approx((up-2*p+down)/h**2, abs=1e-7)
    assert g['vega'] == pytest.approx((black_scholes(option, vol+h)-black_scholes(option, vol-h))/(2*h), rel=1e-4)
    assert g['rho'] == pytest.approx((black_scholes(replace(option, rate=option.rate+h), vol)-black_scholes(replace(option, rate=option.rate-h), vol))/(2*h), rel=1e-4)
    assert g['theta'] == pytest.approx((black_scholes(replace(option, maturity=option.maturity-h), vol)-black_scholes(replace(option, maturity=option.maturity+h), vol))/(2*h), rel=1e-4)


def test_boundaries_and_invalid_inputs():
    assert black_scholes(Option(110, 100, 0), 0.2) == 10
    assert black_scholes(Option(100, 110, 1), 0) == 0
    assert implied_volatility(Option(100, 100, 1), 0) == 0
    for price in (-1, 100, float('nan')):
        with pytest.raises(ValueError):
            implied_volatility(Option(100, 100, 1), price)
    with pytest.raises(ValueError):
        implied_volatility(Option(100, 100, 0), 0)
    with pytest.raises(ValueError):
        Option(0, 100, 1)
    with pytest.raises(ValueError):
        black_scholes(Option(100, 100, 1), -0.2)
