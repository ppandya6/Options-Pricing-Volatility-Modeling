import json
import subprocess
import sys
from pathlib import Path
import pandas as pd


def test_demo_and_csv_workflow(tmp_path):
    demo = tmp_path/'demo'
    subprocess.run([sys.executable, '-m', 'options_lab.cli', 'demo', '--paths', '1000',
                    '--steps', '8', '--output', str(demo)], check=True, capture_output=True)
    summary = json.loads((demo/'summary.json').read_text())
    assert 'SYNTHETIC' in summary['data_source']
    assert summary['gbm']['paths'] == 1000
    output = tmp_path/'quotes'
    subprocess.run([sys.executable, '-m', 'options_lab.cli', 'quotes', str(demo/'quotes.csv'),
                    '--output', str(output)], check=True, capture_output=True)
    assert (output/'summary.json').exists()
    closes = pd.read_csv(demo/'synthetic_closes.csv')
    closes['implied'] = 0.2
    daily = tmp_path/'daily.csv'
    closes.to_csv(daily, index=False)
    matched = tmp_path/'matched'
    script = Path(__file__).resolve().parents[1]/'examples'/'matched_volatility.py'
    subprocess.run([sys.executable, str(script), str(daily), '--output', str(matched)],
                   check=True, capture_output=True)
    scores = json.loads((matched/'scores.json').read_text())
    assert scores['implied']['observations'] > 0
