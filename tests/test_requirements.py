from pathlib import Path

REQ = Path(__file__).resolve().parent.parent / 'requirements.txt'


def _lines():
    raw = REQ.read_bytes()
    assert not raw.startswith((b'\xff\xfe', b'\xfe\xff', b'\xef\xbb\xbf')), 'requirements.txt must be plain UTF-8 (no BOM / UTF-16)'
    text = raw.decode('utf-8')                 # raises on UTF-16
    return [l.strip() for l in text.splitlines() if l.strip() and not l.startswith('#')]


def test_requirements_is_utf8_and_pip_parsable():
    lines = _lines()
    assert len(lines) > 30 and all('==' in l for l in lines)


def test_no_pathlib_backport_and_needed_packages_present():
    names = {l.split('==')[0].lower() for l in _lines()}
    assert 'pathlib' not in names, 'the PyPI pathlib backport shadows the standard library module'
    assert {'pypdf', 'pytest', 'pdfplumber', 'lightgbm', 'flaml', 'pandas', 'scipy'} <= names
