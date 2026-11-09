"""Compare real same-date, fixed-horizon IV against historical forecasts.

Usage: python examples/matched_volatility.py daily.csv --horizon 21
CSV columns: date,close,implied (annual decimal IV, same horizon throughout).
Run from repository root after installing the package.
"""
import argparse
import json
from pathlib import Path
import pandas as pd
from options_lab.volatility import forecasts, compare_implied


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('csv', type=Path)
    parser.add_argument('--horizon', type=int, default=21)
    parser.add_argument('--window', type=int, default=21)
    parser.add_argument('--output', type=Path, default=Path('outputs/matched-volatility'))
    args = parser.parse_args()
    frame = pd.read_csv(args.csv, parse_dates=['date']).set_index('date')
    if not frame.index.is_unique or not frame.index.is_monotonic_increasing or frame.index.hasnans:
        parser.error('Dates must be unique, valid and increasing')
    history = forecasts(frame.close, window=args.window, horizon=args.horizon)
    # Missing IV quotes are omitted, not imputed; invalid observed values fail.
    joined, scores = compare_implied(history, frame.implied.dropna())
    args.output.mkdir(parents=True, exist_ok=True)
    joined.to_csv(args.output/'aligned.csv')
    text = json.dumps(scores, indent=2, allow_nan=False)
    (args.output/'scores.json').write_text(text + '\n')
    print(text)


if __name__ == '__main__':
    main()
