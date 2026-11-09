"""Offline demo and user-supplied snapshot analysis."""
import argparse
from dataclasses import asdict
import json
from pathlib import Path
import numpy as np
import pandas as pd
from .pricing import Option, black_scholes, greeks
from .simulation import Heston, gbm_price, heston_price
from .research import analyze_quotes, compare_models, synthetic_quotes
from .volatility import forecasts, forecast_scores


def _write_json(path, value):
    path.write_text(json.dumps(value, indent=2, allow_nan=False) + "\n")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    sub = parser.add_subparsers(dest="command", required=True)
    demo = sub.add_parser("demo", help="Reproducible synthetic research experiment")
    demo.add_argument("--output", type=Path, default=Path("outputs/demo"))
    demo.add_argument("--paths", type=int, default=30_000)
    demo.add_argument("--steps", type=int, default=126)
    quotes = sub.add_parser("quotes", help="Analyze a synchronized European-option CSV snapshot")
    quotes.add_argument("csv", type=Path)
    quotes.add_argument("--output", type=Path, default=Path("outputs/quotes"))
    args = parser.parse_args()
    try:
        frame = synthetic_quotes() if args.command == "demo" else pd.read_csv(args.csv)
        analyzed = analyze_quotes(frame)
        comparison, scores = compare_models(analyzed)
        args.output.mkdir(parents=True, exist_ok=True)
        comparison.to_csv(args.output/"quotes.csv", index=False)
        summary = {"data_source": "SYNTHETIC — not observed market data" if args.command == "demo"
                   else f"User-supplied snapshot: {args.csv}; provenance not independently verified",
                   "comparison": scores,
                   "interpretation": "Compare test RMSE and spread coverage, not calibration fit alone. "
                   "The smile has 3 parameters vs 1 for flat BS. A single snapshot is not evidence "
                   "of persistent out-of-sample improvement."}
        if args.command == "demo":
            option = Option(100, 100, 0.5, 0.04, 0.01)
            model = Heston()
            rng = np.random.default_rng(7)
            closes = pd.Series(100*np.exp(np.cumsum(rng.normal(0.0002, 0.2/np.sqrt(252), 757))),
                               index=pd.bdate_range("2022-01-03", periods=757))
            closes.index.name = "date"
            history = forecasts(closes)
            history.index.name = "date"
            history.to_csv(args.output/"volatility.csv")
            pd.DataFrame({"close": closes}).to_csv(args.output/"synthetic_closes.csv")
            summary.update(black_scholes=black_scholes(option, 0.2), greeks=greeks(option, 0.2),
                           gbm=asdict(gbm_price(option, 0.2, paths=args.paths)),
                           heston={"parameters": asdict(model), "feller_satisfied": model.feller_satisfied,
                                   "coarse": asdict(heston_price(option, model, args.paths, args.steps)),
                                   "fine": asdict(heston_price(option, model, args.paths, 2*args.steps)),
                                   "note": "Step refinement is diagnostic, not a bias correction; CIs exclude discretization bias."},
                           forecast_scores=forecast_scores(history),
                           latest_volatility={"historical": float(history.historical.iloc[-1]),
                                              "ewma": float(history.ewma.iloc[-1]),
                                              "atm_implied": float(analyzed.iloc[8].implied_volatility)},
                           volatility_note="Synthetic closes and quotes were generated independently. "
                           "Latest HV/EWMA are daily annualized estimates; ATM IV has a six-month horizon. "
                           "They are not matched-horizon forecasts or evidence of a volatility premium.")
        _write_json(args.output/"summary.json", summary)
        print(json.dumps(summary, indent=2, allow_nan=False))
        print(f"\nArtifacts: {args.output.resolve()}")
    except (ValueError, OSError, RuntimeError) as exc:
        parser.exit(2, f"options-lab: {exc}\n")


if __name__ == "__main__":
    main()
