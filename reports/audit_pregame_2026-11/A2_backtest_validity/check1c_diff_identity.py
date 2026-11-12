"""Check 1c: derived columns in the 2026-11-01 training set are exactly home-minus-away of their parts (no outcome-dependent derivation)."""
import sys; sys.path.insert(0, __import__('pathlib').Path(__file__).parent.as_posix())
from common import *
t = load_train()
bad = []; n = 0
for c in t.columns:
    if c.startswith('diff_'):
        base = c[5:]
        h, a = f'{base}_home', f'{base}_away'
        if h in t and a in t:
            n += 1
            d = (t[c] - (t[h] - t[a])).abs()
            d = d.where(~(t[c].isna() & (t[h] - t[a]).isna()), 0)
            if d.max() > 1e-9: bad.append((c, float(d.max())))
        else:
            bad.append((c, 'parts missing'))
print('diff_ columns checked:', n, 'violations:', bad)
m = (t['matchup_3pa_vs_opp_3pa_5g'] - (t['off_threePointersAttempted_5g_home'] - t['def_OPP_FG3A_5g_away'])).abs().max()
print('matchup_3pa_vs_opp_3pa_5g == off_3PA_5g_home - def_OPP_FG3A_5g_away, max abs diff', m)
print('matchup_* columns present:', [c for c in t.columns if c.startswith('matchup_') and c not in ('matchup_home','matchup_away')])
print('def_defensiveRating columns present:', [c for c in t.columns if 'def_defensiveRating' in c], ' | off_defensiveRating present:', len([c for c in t.columns if c.startswith('off_defensiveRating')]))
