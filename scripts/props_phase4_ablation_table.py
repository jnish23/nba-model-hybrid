"""
Descriptive feature-group ablations (one group dropped, retrained with the frozen v2_base families on the v2 pipeline): paired CRPS change versus the full model
on the dev fold 2023-24 and on the seen season 2024-25, per stat, with game-clustered intervals. Nothing is selected from this table.

    python scripts/props_phase4_ablation_table.py --out reports/props/phase4_ablations.md
"""
import argparse
import sys
from pathlib import Path

import pandas as pd

sys.path.append(str(Path(__file__).resolve().parent.parent))
from props.models import evaluate as ev  # noqa: E402
from props.quality import md_table  # noqa: E402

GROUPS = ['injury', 'teammates', 'opponent', 'team_context', 'rest', 'schedule', 'role', 'conversion']
BASE = {'2023-24': 'v2_base', '2024-25': 'v2_val_2024-25'}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--out', required=True)
    ap.add_argument('--n-boot', type=int, default=500)
    a = ap.parse_args()
    out = []
    for season, base in BASE.items():
        ra, pa = ev.load_rows_light(base, season)
        for g in GROUPS:
            rb, pb = ev.load_rows_light(f'abl_{g}_{season}', season)
            c = ev.compare_versions(ra, pa, rb, pb, n_boot=a.n_boot)            # B minus A: positive = dropping the group hurts
            core = c[c['stat'].isin(['points', 'rebounds', 'assists', 'threes', 'pra'])]
            out.append({'season': season, 'dropped group': g, 'mean CRPS change over 8 stats (%)': 100 * (c['B_minus_A'] / c['crps_A']).mean(),
                        'points': c.loc[c['stat'] == 'points', 'B_minus_A'].iloc[0], 'points lo': c.loc[c['stat'] == 'points', 'lo'].iloc[0], 'points hi': c.loc[c['stat'] == 'points', 'hi'].iloc[0],
                        'pra': c.loc[c['stat'] == 'pra', 'B_minus_A'].iloc[0], 'stats significantly worse (of 8)': int((c['lo'] > 0).sum()), 'stats significantly better (of 8)': int((c['hi'] < 0).sum())})
            print(out[-1], flush=True)
    t = pd.DataFrame(out)
    md = ['# Descriptive feature-group ablations (v2 pipeline)\n',
          'One group dropped at a time, retrained with the frozen v2_base rate families; paired CRPS versus the full v2 model (positive = dropping the group hurts). 2023-24 is a development fold, 2024-25 is seen. '
          'Nothing was selected from this table and no group was removed. H4 (team-level opponent effects) is read from the `opponent` and `team_context` rows.\n', md_table(t.round(4), 4)]
    Path(a.out).write_text('\n'.join(md), encoding='utf-8')


if __name__ == '__main__':
    main()
