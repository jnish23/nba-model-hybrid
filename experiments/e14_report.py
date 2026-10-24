"""Build the E14 HTML report from results.json + e14_games.csv. python -m experiments.e14_report"""
import json
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parent.parent
OUT = ROOT / 'reports' / 'experiments' / 'E14_blowout_under'
SEASONS = ['2019-20', '2020-21', '2021-22', '2022-23', '2023-24', '2024-25', '2025-26']


def main():
    res = json.loads((OUT / 'results.json').read_text())
    g = pd.read_csv(OUT / 'e14_games.csv')
    g = g[g['seasonYear'].isin(SEASONS)].copy()
    day = pd.factorize(pd.to_datetime(g['gameDate']))[0]
    data = {'seasons': SEASONS, 'nd': int(day.max() + 1),
            's': [SEASONS.index(s) for s in g['seasonYear']],
            'a': np.rint(g['abs_spread'] * 10).astype(int).tolist(),
            'r': np.rint(g['r'] * 2).astype(int).tolist(),
            'o': g['ot'].astype(int).tolist(), 'd': day.tolist()}
    html = (ROOT / 'experiments' / 'e14_report_template.html').read_text(encoding='utf-8')
    html = html.replace('__R__', json.dumps(res)).replace('__G__', json.dumps(data, separators=(',', ':')))
    (OUT / 'report.html').write_text(html, encoding='utf-8')
    print(OUT / 'report.html', len(html) // 1024, 'KB')


if __name__ == '__main__':
    main()
