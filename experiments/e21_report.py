"""Build the E21 HTML report: E14's page head/CSS and chart helpers + the E21 body. python -m experiments.e21_report"""
import json
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
OUT = ROOT / 'reports' / 'experiments' / 'E21_pace'


def main():
    e14 = (ROOT / 'experiments' / 'e14_report_template.html').read_text(encoding='utf-8')
    head = e14[:e14.index('</style>') + len('</style>')].replace('<title>Blowout Under Test</title>', '<title>Pace Mismatch Test</title>')
    a, b = e14.index("const NS = 'http://www.w3.org/2000/svg';"), e14.index('// seasons')
    chart_js = e14[a:b]
    body = (ROOT / 'experiments' / 'e21_report_template.html').read_text(encoding='utf-8')
    res = json.loads((OUT / 'results.json').read_text())
    html = head + '\n\n' + body.replace('__R__', json.dumps(res)).replace('__CHART_JS__', chart_js)
    (OUT / 'report.html').write_text(html, encoding='utf-8')
    print(OUT / 'report.html', len(html) // 1024, 'KB')


if __name__ == '__main__':
    main()
