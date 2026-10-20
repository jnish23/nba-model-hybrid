"""
Parse official NBA injury report PDFs into rows.

The PDF is a table: Game Date | Game Time | Matchup | Team | Player Name | Current Status | Reason.
Date/time/matchup/team cells are only printed on the first row of their group, so they are
carried forward (also across pages). Long reasons wrap onto extra lines, which are ignored.
Two layouts exist: older PDFs keep spaces ("Brooklyn Nets", "Williams III, Robert"); newer ones
drop them ("NewOrleansPelicans", "SamuelsJr.,Jermaine"). Column boundaries are read from each
page's header row, so both work.

Output columns:
  report_ts (datetime, ET as printed in the header), game_date, game_time, matchup (AWAY@HOME),
  team (as printed), player (as printed, "Last,First"), status, reason, not_submitted (bool row
  marking a team whose report was not yet submitted; player/status empty).
"""
import re
from datetime import datetime

import pdfplumber
import pandas as pd

PARSER_VERSION = 2  # bump when parsing changes; cached results from older versions are re-parsed

STATUSES = {'out', 'doubtful', 'questionable', 'probable', 'available'}
COLUMNS = ['date', 'time', 'matchup', 'team', 'player', 'status', 'reason']
LINE_TOL = 2.5  # points: words within this vertical distance are on the same line

_HEADER_RE = re.compile(r'Injury Report:\s*(\d\d/\d\d/\d\d)\s*(\d\d:\d\d)\s*(AM|PM)', re.I)
_DATE_RE = re.compile(r'^\d\d/\d\d/\d{4}$')
_MATCHUP_RE = re.compile(r'^[A-Z]{2,4}@[A-Z]{2,4}$')


def _lines(words):
    """Group words into lines by vertical position, each sorted left to right."""
    lines = []
    for w in sorted(words, key=lambda w: (w['top'], w['x0'])):
        if lines and abs(lines[-1][0] - w['top']) <= LINE_TOL:
            lines[-1][1].append(w)
        else:
            lines.append([w['top'], [w]])
    return [sorted(ws, key=lambda w: w['x0']) for _, ws in lines]


# Header tokens -> column names. Older (2019-20) reports add Category / Previous Status /
# Previous Reason columns and reorder Reason and Current Status, so columns are mapped by name.
_HEADER_PAIRS = {('Game', 'Date'): 'date', ('Game', 'Time'): 'time', ('Player', 'Name'): 'player',
                 ('Current', 'Status'): 'status', ('Previous', 'Status'): 'prev_status',
                 ('Previous', 'Reason'): 'prev_reason'}
_HEADER_SINGLES = {'GameDate': 'date', 'GameTime': 'time', 'Matchup': 'matchup', 'Team': 'team',
                   'PlayerName': 'player', 'CurrentStatus': 'status', 'Reason': 'reason',
                   'Category': 'category', 'PreviousStatus': 'prev_status', 'PreviousReason': 'prev_reason'}
REQUIRED = {'date', 'time', 'matchup', 'team', 'player', 'status'}


def _column_starts(lines):
    """Find the header row and return [(x_start, column_name), ...] sorted by x, or None."""
    for ws in lines:
        texts = [w['text'] for w in ws]
        if 'Matchup' not in texts:
            continue
        cols, i = [], 0
        while i < len(ws):
            t = ws[i]['text']
            nxt = ws[i + 1]['text'] if i + 1 < len(ws) else None
            if (t, nxt) in _HEADER_PAIRS:
                cols.append((ws[i]['x0'] - 4, _HEADER_PAIRS[(t, nxt)]))
                i += 2
            else:
                if t in _HEADER_SINGLES:
                    cols.append((ws[i]['x0'] - 4, _HEADER_SINGLES[t]))
                i += 1
        if REQUIRED <= {name for _, name in cols}:
            return sorted(cols)
    return None


def _split(ws, starts):
    cols = {}
    for w in ws:
        name = starts[0][1]
        for x, n in starts:
            if w['x0'] >= x:
                name = n
        cols.setdefault(name, []).append(w['text'])
    out = {c: ' '.join(v) for c, v in cols.items()}
    return {c: out.get(c, '') for c in COLUMNS}


def parse_report(path):
    """Parse one injury report PDF. Returns a DataFrame (possibly empty)."""
    rows = []
    report_ts = None
    starts = None
    cur = {'date': None, 'time': None, 'matchup': None, 'team': None}
    with pdfplumber.open(path) as pdf:
        for page in pdf.pages:
            words = page.extract_words()
            lines = _lines(words)
            page_starts = _column_starts(lines)
            starts = page_starts or starts
            for ws in lines:
                text = ' '.join(w['text'] for w in ws)
                if report_ts is None:
                    m = _HEADER_RE.search(text.replace('Injury Report:', 'Injury Report: '))
                    if m:
                        report_ts = datetime.strptime(' '.join(m.groups()), '%m/%d/%y %I:%M %p')
                if starts is None or text.startswith('Injury Report') or 'Matchup' in text or text.startswith('Page'):
                    continue
                c = _split(ws, starts)
                d = c['date'].replace(' ', '')
                if _DATE_RE.match(d):
                    cur['date'] = d
                t = c['time'].replace(' ', '')
                if '(ET)' in t:
                    cur['time'] = t.replace('(ET)', '')
                mu = c['matchup'].replace(' ', '')
                if _MATCHUP_RE.match(mu):
                    cur['matchup'] = mu
                if c['team']:
                    cur['team'] = c['team']
                if c['reason'].replace(' ', '').upper() == 'NOTYETSUBMITTED':
                    rows.append({**cur, 'player': None, 'status': None, 'reason': None, 'not_submitted': True})
                    continue
                status = c['status'].strip()
                if status.lower() in STATUSES and c['player']:
                    rows.append({**cur, 'player': c['player'], 'status': status.capitalize(),
                                 'reason': c['reason'], 'not_submitted': False})
    df = pd.DataFrame(rows, columns=['date', 'time', 'matchup', 'team', 'player', 'status', 'reason', 'not_submitted'])
    df = df.rename(columns={'date': 'game_date', 'time': 'game_time'})
    if report_ts is None or starts is None:
        raise ValueError(f"{path}: no report timestamp or table header found")
    df.insert(0, 'report_ts', report_ts)
    if not df.empty:
        df['game_date'] = pd.to_datetime(df['game_date'], format='%m/%d/%Y').dt.date
    return df
