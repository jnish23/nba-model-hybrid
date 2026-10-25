import json
import sqlite3

import pandas as pd
import pytest

from props.live import config, store


def bet(**kw):
    row = {'bet_id': 'S|g1|7|assists', 'season': 'S', 'game_id': 'g1', 'player_id': 7, 'stat': 'assists', 'side': 'over', 'line': 4.5, 'created_utc': '2026-10-23T00:41:00Z', 'book': 'fanduel'}
    row.update(kw)
    return pd.DataFrame([row])


def test_a_recorded_bet_cannot_be_updated_or_deleted(tmp_path):
    conn = store.connect(tmp_path / 'live.db')
    store.write_slot(conn, 'g1', 60, 'done', 'now', 'r1', bets=bet())
    with pytest.raises(sqlite3.DatabaseError, match='immutable'):
        conn.execute("UPDATE prop_bets SET side = 'under'")
    with pytest.raises(sqlite3.DatabaseError, match='immutable'):
        conn.execute('DELETE FROM prop_bets')
    assert conn.execute('SELECT side FROM prop_bets').fetchone()[0] == 'over'


def test_a_second_write_of_the_same_player_game_stat_keeps_the_first_bet(tmp_path):
    conn = store.connect(tmp_path / 'live.db')
    store.write_slot(conn, 'g1', 240, 'done', 'now', 'r1', bets=bet(line=4.5))
    n_sig, n_bet = store.write_slot(conn, 'g1', 60, 'done', 'later', 'r2', bets=bet(bet_id='S|g1|7|assists|b', line=5.5, side='under'))
    assert n_bet == 0
    assert conn.execute('SELECT count(*), min(line), min(side) FROM prop_bets').fetchone() == (1, 4.5, 'over')
    assert store.slot_done(conn, 'g1', 60)[0] == 'done' and store.slot_done(conn, 'g1', 5) is None


def test_existing_bet_keys_and_health(tmp_path):
    conn = store.connect(tmp_path / 'live.db')
    store.write_slot(conn, 'g1', 60, 'done', 'now', 'r1', bets=bet())
    assert store.existing_bet_keys(conn, ['g1', 'g2']) == {('g1', 7, 'assists')}
    store.health(conn, 'r1', 'signal', 'a', 'b', 'ok', 1, {'x': 1})
    assert conn.execute('SELECT status FROM pipeline_health').fetchone()[0] == 'ok'


def _pinned(tmp_path):
    rule = tmp_path / 'rule.json'
    rule.write_text('{"k": 1}')
    bundle_dir = tmp_path / 'v' / '2026-27'
    bundle_dir.mkdir(parents=True)
    (bundle_dir / 'bundle.pkl').write_bytes(b'bundle')
    feat = tmp_path / 'f.db'
    feat.write_bytes(b'x' * 10)
    from props.models import registry
    cfg = {'rule_file': str(rule), 'rule_sha256': registry.sha256_file(rule), 'model_version': 'v', 'model_season': '2026-27',
           'model_bundle_sha256': registry.sha256_file(bundle_dir / 'bundle.pkl'), 'features_db': str(feat), 'features_db_bytes': 10, 'features_db_sha256': 'unused'}
    return cfg, rule, bundle_dir / 'bundle.pkl', feat


def test_the_guard_refuses_a_changed_rule_bundle_or_features_file(tmp_path, monkeypatch):
    cfg, rule, bundle, feat = _pinned(tmp_path)
    monkeypatch.setattr(config, 'bundle_path', lambda c: bundle)
    assert config.verify(cfg) is cfg
    rule.write_text('{"k": 2}')
    with pytest.raises(config.ConfigRefused, match='rule file'):
        config.verify(cfg)
    rule.write_text('{"k": 1}')
    bundle.write_bytes(b'other')
    with pytest.raises(config.ConfigRefused, match='model bundle'):
        config.verify(cfg)
    bundle.write_bytes(b'bundle')
    feat.write_bytes(b'x' * 11)
    with pytest.raises(config.ConfigRefused, match='features database'):
        config.verify(cfg)


def test_a_bet_that_violates_a_constraint_aborts_the_slot_instead_of_vanishing(tmp_path):
    conn = store.connect(tmp_path / 'live.db')
    bad = bet(side=None)                                                  # NOT NULL side missing
    with pytest.raises(sqlite3.IntegrityError):
        store.write_slot(conn, 'g1', 60, 'done', 'now', 'r1', bets=bad)
    assert store.slot_done(conn, 'g1', 60) is None and conn.execute('SELECT count(*) FROM prop_bets').fetchone()[0] == 0


def test_the_writer_stamps_created_utc_and_run_id(tmp_path):
    conn = store.connect(tmp_path / 'live.db')
    b = bet().drop(columns=['created_utc'])
    store.write_slot(conn, 'g1', 60, 'done', '2026-10-23T00:42:52Z', 'r9', bets=b)
    assert conn.execute('SELECT created_utc, run_id FROM prop_bets').fetchone() == ('2026-10-23T00:42:52Z', 'r9')
