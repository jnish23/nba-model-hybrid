"""
Step 6: run scripts/props_signal_job.py unchanged with --dry-run, and record what signal.evaluate_item returns (signals and would-be bets) so the model probabilities and expected
minutes can be checked. Nothing in the job is altered; the wrapper only keeps a copy of each return value.

    .venv/Scripts/python.exe reports/props/restore_2026-11/step6_dry_run.py --config ... --db ... --now ... --dry-run
Writes step6_signals.csv and step6_bets.csv in this folder.
"""
import importlib.util
import sys
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parents[3]
HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT))


def main():
    assert '--dry-run' in sys.argv, 'step 6 runs the job in dry-run mode only'
    spec = importlib.util.spec_from_file_location('props_signal_job', ROOT / 'scripts/props_signal_job.py')
    job = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(job)
    sigs, bets = [], []
    original = job.signal.evaluate_item

    def keep(*args, **kwargs):
        sig, new, detail = original(*args, **kwargs)
        sigs.append(sig)
        bets.append(new)
        return sig, new, detail

    job.signal.evaluate_item = keep
    sys.argv = ['props_signal_job.py'] + sys.argv[1:]
    code = job.main()
    s = pd.concat(sigs, ignore_index=True) if sigs else pd.DataFrame()
    b = pd.concat(bets, ignore_index=True) if bets else pd.DataFrame()
    s.to_csv(HERE / 'step6_signals.csv', index=False)
    b.drop(columns=[c for c in ('pmf_json', 'books_json') if c in b.columns]).to_csv(HERE / 'step6_bets.csv', index=False)
    print(f'exit code {code}; {len(s)} signals, {len(b)} would-be bets')
    if len(s):
        print(s.groupby(['game_id', 'slot_minutes']).agg(signals=('signal_id', 'size'), players=('player_id', 'nunique'), qualified=('qualified', 'sum')).to_string())
        print(s[['p_model', 'p_star', 'p_market', 'expected_minutes']].describe().to_string())
        print('p_model in (0,1):', bool(((s['p_model'] > 0) & (s['p_model'] < 1)).all()), '| expected_minutes in [0,48]:',
              bool(s['expected_minutes'].between(0, 48).all()))
    return code


if __name__ == '__main__':
    sys.exit(main())
