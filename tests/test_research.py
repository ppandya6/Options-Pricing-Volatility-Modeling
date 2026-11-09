import numpy as np
import pandas as pd
import pytest
from options_lab.research import synthetic_quotes, analyze_quotes, compare_models
from options_lab.volatility import forecasts, forecast_scores


def test_holdout_improvement_and_no_leakage():
    quotes = analyze_quotes(synthetic_quotes())
    fitted, scores = compare_models(quotes)
    assert scores['quadratic_smile']['test']['rmse'] < 1e-4
    assert scores['flat_bs']['test']['rmse'] > 0.05
    changed = quotes.copy()
    changed.loc[changed.split == 'test', 'mid'] += 0.2
    refit, _ = compare_models(changed)
    assert np.allclose(fitted.quadratic_smile_price, refit.quadratic_smile_price)
    assert np.allclose(fitted.flat_bs_price, refit.flat_bs_price)


@pytest.mark.parametrize('problem', ['crossed', 'duplicate', 'nan', 'arbitrage', 'expiry', 'split'])
def test_reject_bad_quotes(problem):
    frame = synthetic_quotes()
    if problem == 'crossed':
        frame.loc[0, 'bid'] = 1000
    elif problem == 'duplicate':
        frame = pd.concat([frame, frame.iloc[[0]]])
    elif problem == 'nan':
        frame.loc[0, 'ask'] = np.nan
    elif problem == 'arbitrage':
        frame.loc[0, ['bid', 'ask']] = 101
    elif problem == 'expiry':
        frame.loc[0, 'maturity'] = 0
    else:
        frame.loc[0, 'split'] = 'unknown'
    with pytest.raises(ValueError):
        analyze_quotes(frame)


def test_forecast_alignment_and_no_lookahead():
    returns = np.array([0.01, -0.02, 0.03, -0.04, 0.05, -0.06, 0.07, -0.08])
    prices = pd.Series(100*np.exp(np.r_[0, returns].cumsum()))
    frame = forecasts(prices, window=3, horizon=2)
    assert frame.loc[3, 'forward_realized'] == pytest.approx(np.sqrt(252*np.mean(returns[3:5]**2)))
    assert frame.loc[3, 'historical'] == pytest.approx(np.std(returns[:3], ddof=1)*np.sqrt(252))
    changed = prices.copy()
    changed.iloc[5:] *= 2
    new = forecasts(changed, window=3, horizon=2)
    assert new.loc[3, 'historical'] == frame.loc[3, 'historical']
    assert new.loc[3, 'ewma'] == frame.loc[3, 'ewma']
    assert frame.forward_realized.iloc[-2:].isna().all()
    assert forecast_scores(frame)['historical']['observations'] == 4


def test_matched_implied_alignment():
    from options_lab.volatility import compare_implied
    target = pd.DataFrame({'historical': [0.2, 0.3], 'ewma': [0.25, 0.25],
                           'forward_realized': [0.22, 0.28]}, index=[10, 11])
    joined, scores = compare_implied(target, pd.Series([0.24, 0.9], index=[10, 12]))
    assert list(joined.index) == [10]
    assert scores['implied']['mae'] == pytest.approx(0.02)
    with pytest.raises(ValueError):
        compare_implied(target, pd.Series([0.2], index=[12]))
