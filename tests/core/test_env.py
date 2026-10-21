import os

from core import env


def test_parse_env_text():
    text = '# c\r\nODDS_API_KEY="abc123"\r\nexport OTHER=x # note\r\n\r\nBAD LINE\r\nEMPTY=\r\n'
    assert env.parse_env_text(text) == {'ODDS_API_KEY': 'abc123', 'OTHER': 'x', 'EMPTY': ''}


def test_load_env_does_not_override_and_returns_names_only(tmp_path, monkeypatch):
    (tmp_path / '.env').write_text('ODDS_TEST_A=1\nODDS_TEST_B=2\n')
    monkeypatch.setenv('ODDS_TEST_A', 'real')
    monkeypatch.delenv('ODDS_TEST_B', raising=False)
    assert env.load_env(tmp_path) == ['ODDS_TEST_B']
    assert os.environ['ODDS_TEST_A'] == 'real' and os.environ['ODDS_TEST_B'] == '2'
    monkeypatch.delenv('ODDS_TEST_B')


def test_worktree_falls_back_to_the_main_checkouts_env(tmp_path):
    main = tmp_path / 'main'
    (main / '.git' / 'worktrees' / 'wt').mkdir(parents=True)
    wt = tmp_path / 'wt'
    wt.mkdir()
    (wt / '.git').write_text(f"gitdir: {main / '.git' / 'worktrees' / 'wt'}\n")
    assert env.candidate_files(wt) == [wt / '.env', main / '.env']
    assert env.candidate_files(main) == [main / '.env']
