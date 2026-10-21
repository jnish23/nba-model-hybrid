import datetime as dt

import pandas as pd

from props import injury_archive as ia


def test_plan_skips_what_the_manifest_already_holds(tmp_path):
    days = [dt.date(2024, 1, 15), dt.date(2024, 1, 16)]
    man = pd.DataFrame([{'date': '2024-01-15', 'slot': '08AM', 'stamp': '', 'status': 'missing', 'path': ''}])
    todo = ia.plan(days, manifest=man)
    assert len(todo) == 2 * len(ia.NEW_SLOTS) - 1 and (days[0], (8, 'AM')) not in todo


def test_download_writes_files_records_misses_and_resumes(tmp_path):
    days = [dt.date(2024, 1, 15)]
    calls = []

    def fake_fetch(session, stamp):
        calls.append(stamp)
        # the old hourly name exists for 2 PM only; everything else 403s
        return (200, b'%PDF-fake') if stamp.endswith('_02PM') else (403, None)

    out = ia.download(days, out_dir=tmp_path / 'r', manifest_path=tmp_path / 'm.csv', fetch_fn=fake_fetch, sleep=lambda s: None,
                      session=type('S', (), {'headers': {}})())
    assert out == {'requested': len(ia.NEW_SLOTS), 'downloaded': 1}
    man = pd.read_csv(tmp_path / 'm.csv', dtype=str)
    assert set(man['status']) == {'ok', 'missing'} and len(man) == len(ia.NEW_SLOTS)
    assert (tmp_path / 'r' / '2024' / 'Injury-Report_2024-01-15_02PM.pdf').read_bytes() == b'%PDF-fake'
    n = len(calls)
    ia.download(days, out_dir=tmp_path / 'r', manifest_path=tmp_path / 'm.csv', fetch_fn=fake_fetch, sleep=lambda s: None,
                session=type('S', (), {'headers': {}})())
    assert len(calls) == n                                   # nothing requested twice


def test_team_archive_folder_is_not_used():
    assert 'injury_reports_hourly' in str(ia.HOURLY_DIR) and ia.HOURLY_DIR.name != 'injury_reports'
