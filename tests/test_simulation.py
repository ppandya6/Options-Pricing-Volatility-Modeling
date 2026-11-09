import pytest
from options_lab import Option, Heston, gbm_price, heston_price, black_scholes


@pytest.mark.parametrize('kind', ['call', 'put'])
def test_gbm_matches_analytic(kind):
    option = Option(100, 105, 0.75, 0.03, 0.01, kind)
    result = gbm_price(option, 0.2, paths=100_000)
    assert abs(result.price-black_scholes(option, 0.2)) < 4*result.standard_error
    assert result.ci_low < result.price < result.ci_high
    assert result == gbm_price(option, 0.2, paths=100_000)


def test_heston_constant_variance_limit():
    option = Option(100, 100, 1, 0.03, 0.01)
    result = heston_price(option, Heston(vol_of_vol=0), paths=60_000, steps=32)
    assert abs(result.price-black_scholes(option, 0.2)) < 4*result.standard_error


def test_heston_variance_and_validation():
    result = heston_price(Option(100, 100, 0.5), paths=2000, steps=30)
    assert result.price > 0 and result.standard_error > 0
    assert Heston().feller_satisfied is False
    with pytest.raises(ValueError):
        Heston(rho=2)
    with pytest.raises(ValueError):
        gbm_price(Option(100, 100, 1), 0.2, paths=1)
