# Options Pricing & Volatility Modeling

A reproducible Python research project exploring how theoretical European-option prices relate to market quotes, volatility smiles, and the cost of model complexity.

## Quick start

Requires Python 3.11 or newer. From the repository root:

```bash
python3 -m venv .venv
source .venv/bin/activate
python -m pip install -e '.[dev]'
python -m pytest -q
options-lab demo
```

The demo runs offline with seeded **synthetic data, not observed market prices**. It writes `outputs/demo/summary.json`, `quotes.csv`, `volatility.csv`, and `synthetic_closes.csv`. No API keys or running services are needed. Generated outputs and the virtual environment are ignored by Git.

## Research workflows

| Question | Implementation | Evidence produced |
| --- | --- | --- |
| What does Black–Scholes assume and predict? | Dividend-adjusted calls/puts, analytic Greeks | Prices, sensitivities, parity/reference tests |
| Where are smiles and skews? | Bracketed implied-volatility inversion | IV by strike and log-forward moneyness in `quotes.csv` |
| Does a flexible fit improve pricing? | Flat BS vs three-parameter log-quadratic smile | Train/test RMSE, MAE, and bid/ask coverage |
| How well does Monte Carlo recover known prices? | Exact terminal risk-neutral GBM | Price, standard error, 95% sampling interval |
| What changes under stochastic volatility? | Heston full-truncation variance simulation | Coarse/fine time-step estimates and Feller diagnostic |
| How do volatility estimates compare? | Rolling historical and EWMA forecasts | Forward realized volatility and forecast errors |
| Does IV improve volatility forecasts? | Matched-date, matched-horizon comparison | Common-sample errors for IV, historical, and EWMA |

### Analyze observed option quotes

Supply a **single synchronized snapshot for one underlying**, containing European contracts and explicitly chosen calibration and holdout rows:

```csv
spot,strike,maturity,rate,dividend,kind,bid,ask,split
100,90,0.5,0.04,0.01,call,12.90,13.10,train
```

That single row illustrates the schema; a run needs at least four train contracts at three distinct moneyness values and two test contracts. Generate a full schema example with the demo. Replace synthetic prices with your own licensed or public quotes before interpreting empirical results.

```bash
options-lab quotes path/to/observed_quotes.csv --output outputs/observed
```

- `maturity`: remaining time in years; `rate`, `dividend`, and volatility: annual decimals, rates continuously compounded. `kind`: `call` or `put`.
- Both models fit **training price errors only**, with equal contract weights. Holdout mids never enter calibration. Inspect spread coverage as well as absolute errors.
- CSV validation rejects nonfinite inputs, crossed quotes, duplicate contracts, expired quotes, invalid splits, and mids outside individual European no-arbitrage bounds. It does not establish cross-strike or calendar arbitrage freedom.
- The three-parameter smile is a descriptive benchmark, not Heston calibration or a globally arbitrage-free volatility surface. Holdout strikes within a snapshot measure cross-sectional interpolation, not performance on future dates.

### Compare historical and implied volatility

```bash
python examples/matched_volatility.py path/to/daily.csv --window 21 --horizon 21
```

Input columns are `date,close,implied`, with increasing unique dates and positive split-adjusted closes. `implied` must be an annual decimal, available at that day's close and consistently interpolated to the selected **forward trading-day horizon**. Missing IVs are omitted, never forward-filled. Each row must represent one trading day; the script does not fill missing trading sessions. The forward target is the annualized root mean squared return over the next `horizon` observations. No target return enters its associated forecast.

The output aligns all forecasts on the same dates and reports RMSE, MAE, and signed bias against future realized volatility. Overlapping targets create dependent errors; do not treat the count as independent observations. IV is risk-neutral and includes risk premia, unlike physical historical forecasts. The demo's separate synthetic histories and six-month IV are illustrative and are explicitly **not** a matched-horizon comparison.

### Use the pricing library

```python
from options_lab import Option, Heston, black_scholes, implied_volatility, heston_price

contract = Option(spot=100, strike=105, maturity=0.5, rate=0.04, dividend=0.01)
price = black_scholes(contract, sigma=0.22)
assert abs(implied_volatility(contract, price) - 0.22) < 1e-8
estimate = heston_price(contract, Heston(rho=-0.7), paths=50_000, steps=252, seed=42)
print(estimate.price, estimate.standard_error)
```

## Interpretation and limitations

Black–Scholes assumes European exercise, lognormal continuous prices, constant volatility and rates/yield, frictionless trading, and continuous hedging. Market smiles challenge its constant-volatility assumption. This project does not price American exercise, discrete dividends, jumps, or transaction costs.

Heston adds mean-reverting stochastic variance and correlated price/variance shocks under the risk-neutral measure. The default parameters are illustrative, **not fitted to market data**. Confidence intervals measure Monte Carlo sampling uncertainty only. Finite time steps introduce bias; compare refinements and independent seeds before claiming a pricing improvement. The Feller condition is reported, not enforced; discretization can visit negative auxiliary variance and uses full truncation. See [methodology](docs/methodology.md).

More parameters are useful only if gains survive held-out data, uncertainty, spreads, and computational cost. The synthetic smile is deliberately generated by the flexible benchmark, so its expected holdout advantage is a pipeline check, not market evidence. A credible study needs several market dates, time-based splits, parameter stability checks, and calibrated Heston benchmarks. These are research extensions, not claims made by the demo.

## Project layout

```text
src/options_lab/
  pricing.py       Black–Scholes, Greeks, bounds and IV
  simulation.py    GBM and Heston Monte Carlo
  volatility.py    Historical/EWMA and aligned IV comparisons
  research.py      Quote checks and held-out model comparison
  cli.py           Reproducible demo and observed-quote commands
examples/          Matched-horizon volatility CSV workflow
tests/             Mathematical, simulation, leakage and CLI tests
docs/              Methodology and references
```

CI runs the tests and a small end-to-end demo on Python 3.11–3.13. Dependencies have compatible ranges in `pyproject.toml`; this is not a frozen environment lock. For exact replication, record your resolved environment (`python -m pip freeze`) with your data provenance, parameters, seeds and output artifacts.
