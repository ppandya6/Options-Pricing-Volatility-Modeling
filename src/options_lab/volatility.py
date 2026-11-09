"""Daily-close forecasts aligned to subsequent realized volatility."""
import numpy as np
import pandas as pd


def log_returns(prices):
    series = pd.Series(prices, dtype=float)
    if len(series) < 3 or not np.isfinite(series).all() or (series <= 0).any():
        raise ValueError("At least three finite positive closing prices are required")
    return np.log(series).diff().dropna()


def forecasts(prices, window=21, decay=0.94, annualization=252, horizon=21):
    """Forecasts at close t use returns through t; target uses t+1...t+h.

    Historical: sample standard deviation. EWMA: zero-mean second moment,
    initialized with the first window's mean squared return.
    """
    if not isinstance(window, int) or window < 2 or not isinstance(horizon, int) or horizon < 1:
        raise ValueError("window >= 2 and horizon >= 1 must be integers")
    if not 0 < decay < 1 or not np.isfinite(annualization) or annualization <= 0:
        raise ValueError("Require 0 < decay < 1 and positive annualization")
    returns = log_returns(prices)
    if len(returns) < window + horizon:
        raise ValueError("Not enough history for the window and forward horizon")
    historical = returns.rolling(window).std(ddof=1)*np.sqrt(annualization)
    ewma = pd.Series(np.nan, index=returns.index)
    variance = float(np.mean(returns.iloc[:window]**2))
    ewma.iloc[window-1] = np.sqrt(annualization*variance)
    for i in range(window, len(returns)):
        variance = decay*variance + (1-decay)*returns.iloc[i]**2
        ewma.iloc[i] = np.sqrt(annualization*variance)
    # Explicit forward shifts prevent accidental use of the forecast-date return.
    forward_variance = pd.concat([returns.shift(-i)**2 for i in range(1, horizon+1)], axis=1)
    realized = np.sqrt(annualization*forward_variance.mean(axis=1, skipna=False))
    return pd.DataFrame(dict(historical=historical, ewma=ewma, forward_realized=realized))


def forecast_scores(frame):
    aligned = frame.dropna()
    if aligned.empty:
        raise ValueError("No complete forecast/target pairs")
    return {name: {"rmse": float(np.sqrt(np.mean((aligned[name]-aligned.forward_realized)**2))),
                   "mae": float(np.mean(abs(aligned[name]-aligned.forward_realized))),
                   "observations": len(aligned)} for name in ("historical", "ewma")}


def compare_implied(frame, implied):
    """Score same-date, same-forward-horizon IV alongside historical forecasts.

    Caller guarantees IV horizon matches frame's target and quotes were available
    at the forecast close. IV is a risk-neutral quote, not an unbiased physical
    forecast. Index alignment is exact, with no forward fill.
    """
    iv = pd.Series(implied, dtype=float)
    if not iv.index.is_unique or not frame.index.is_unique:
        raise ValueError("Dates must be unique")
    if not np.isfinite(iv).all() or (iv < 0).any():
        raise ValueError("Implied volatility must be finite and nonnegative")
    joined = frame.join(iv.rename('implied'), how='inner').dropna()
    if joined.empty:
        raise ValueError("No complete aligned forecast/IV/target observations")
    scores = {}
    for name in ('historical', 'ewma', 'implied'):
        error = joined[name]-joined.forward_realized
        scores[name] = dict(rmse=float(np.sqrt(np.mean(error**2))),
                            mae=float(np.mean(abs(error))), bias=float(error.mean()),
                            observations=len(joined))
    return joined, scores
