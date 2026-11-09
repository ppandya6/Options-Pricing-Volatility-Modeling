"""Quote ingestion and an explicit calibration/holdout complexity experiment."""
import numpy as np
import pandas as pd
from scipy.optimize import minimize_scalar, least_squares
from .pricing import Option, black_scholes, implied_volatility

REQUIRED = ["spot", "strike", "maturity", "rate", "dividend", "kind", "bid", "ask", "split"]


def analyze_quotes(frame):
    """One underlying and synchronized snapshot; maturity expressed in years.

    Invalid rows fail explicitly instead of silently contaminating calibration.
    The caller must supply train/test membership to avoid hidden data splitting.
    """
    missing = set(REQUIRED) - set(frame.columns)
    if missing or frame.empty:
        raise ValueError(f"Nonempty quotes required with columns: {REQUIRED}; missing {sorted(missing)}")
    result = frame.copy()
    for name in ("spot", "strike", "maturity", "rate", "dividend", "bid", "ask"):
        result[name] = pd.to_numeric(result[name], errors="raise")
        if not np.isfinite(result[name]).all():
            raise ValueError(f"Nonfinite {name}")
    if (result.bid < 0).any() or (result.ask < result.bid).any():
        raise ValueError("Require 0 <= bid <= ask")
    if not result.split.isin(["train", "test"]).all():
        raise ValueError("split must be train or test")
    if result.duplicated(["strike", "maturity", "kind"]).any():
        raise ValueError("Duplicate contracts: supply a single synchronized quote snapshot")
    if any(result[name].nunique() != 1 for name in ("spot", "rate", "dividend")):
        raise ValueError("Use one synchronized underlying/rate/yield snapshot per run")
    result["mid"] = (result.bid + result.ask)/2
    ivs = []
    for idx, row in result.iterrows():
        try:
            option = Option(row.spot, row.strike, row.maturity, row.rate, row.dividend, row.kind)
            ivs.append(implied_volatility(option, row.mid))
        except ValueError as exc:
            raise ValueError(f"Invalid quote at row {idx}: {exc}") from exc
    result["implied_volatility"] = ivs
    result["log_forward_moneyness"] = np.log(result.strike / (
        result.spot*np.exp((result.rate-result.dividend)*result.maturity)))
    return result


def compare_models(quotes):
    """Fit constant BS volatility vs a 3-parameter log-quadratic IV smile.

    Both minimize unweighted price squared error on train rows only. The smile
    is descriptive, not a stochastic process or an arbitrage-free surface.
    """
    train = quotes[quotes.split == "train"]
    test = quotes[quotes.split == "test"]
    if len(train) < 4 or len(test) < 2 or train.log_forward_moneyness.nunique() < 3:
        raise ValueError("Need >=4 train quotes at >=3 moneyness values and >=2 test quotes")
    options = [Option(r.spot, r.strike, r.maturity, r.rate, r.dividend, r.kind)
               for r in quotes.itertuples()]
    train_mask = (quotes.split == "train").to_numpy()
    x = quotes.log_forward_moneyness.to_numpy()
    mids = quotes.mid.to_numpy()

    def prices(vols):
        return np.array([black_scholes(o, float(v)) for o, v in zip(options, vols)])

    flat = minimize_scalar(lambda v: np.mean((prices(np.full(len(x), v))[train_mask]
                                               - mids[train_mask])**2),
                           bounds=(0.0001, 5), method="bounded")
    if not flat.success:
        raise RuntimeError("Constant-volatility fit did not converge")

    def smile_vol(params):
        return np.exp(np.clip(params[0] + params[1]*x + params[2]*x*x,
                              np.log(0.0001), np.log(5)))

    smile = least_squares(lambda p: (prices(smile_vol(p))-mids)[train_mask],
                          [np.log(flat.x), 0, 0], bounds=([-9, -20, -20], [2, 20, 20]))
    if not smile.success:
        raise RuntimeError("Smile fit did not converge")
    output = quotes.copy()
    scores = {}
    for name, vols, count in [("flat_bs", np.full(len(x), flat.x), 1),
                               ("quadratic_smile", smile_vol(smile.x), 3)]:
        prediction = prices(vols)
        output[name + "_price"] = prediction
        output[name + "_vol"] = vols
        scores[name] = {"parameters": count}
        for split in ("train", "test"):
            mask = (quotes.split == split).to_numpy()
            error = prediction[mask] - mids[mask]
            scores[name][split] = dict(rmse=float(np.sqrt(np.mean(error**2))),
                                       mae=float(np.mean(abs(error))), observations=int(mask.sum()),
                                       within_spread=float(np.mean((prediction[mask] >= quotes.bid.to_numpy()[mask]) &
                                                                   (prediction[mask] <= quotes.ask.to_numpy()[mask]))))
    return output, scores


def synthetic_quotes():
    rows = []
    for i, strike in enumerate(np.linspace(75, 125, 17)):
        option = Option(100, float(strike), 0.5, 0.04, 0.01)
        x = np.log(strike/(100*np.exp(0.03*0.5)))
        vol = 0.22*np.exp(-0.8*x + 1.2*x*x)
        mid = black_scholes(option, vol)
        rows.append(dict(spot=100, strike=strike, maturity=0.5, rate=0.04,
                         dividend=0.01, kind="call", bid=max(0, mid-0.03), ask=mid+0.03,
                         split="test" if i % 3 == 1 else "train"))
    return pd.DataFrame(rows)
