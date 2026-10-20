"""
Live injury status from the official NBA injury-report PDFs.

This module used to scrape ESPN's injuries page (a different source, with different statuses and no game/time stamp, from
the one every training feature is built on). The training and live inj_* features both come from the PDF pipeline in
models/injury_report_features.py; this class is now a thin accessor over that same pipeline. ESPN survives only as a
side-by-side comparator in models/injury_source_compare.py.
"""
import logging

import pandas as pd

from models import injury_report_features as irf
from models.injury_source_compare import latest_pdf_rows


class LiveRotationBuilder:
    def __init__(self, db_path):
        self.db_path = db_path
        self.injury_map = {}          # normalised player key -> status (official report)

    def fetch_injury_report(self, as_of=None, download=True):
        """Newest official report at/before `as_of` (US/Eastern, default now) for as_of's date -> {player_key: status}."""
        from scripts.scrape_injury_reports import live as fetch_report, now_et
        as_of = pd.Timestamp(as_of) if as_of is not None else pd.Timestamp(now_et())
        if download:
            path = fetch_report(as_of=as_of)
            if path is None:
                logging.error("Injury report download FAILED; using only the reports already on disk")
        rows = latest_pdf_rows(irf.parse_all_reports(), as_of.normalize(), as_of)
        self.injury_map = {irf.report_name_key(p): s for p, s in zip(rows['player'], rows['status'])}
        logging.info(f"Official injury report: {len(self.injury_map)} listed players")
        return self.injury_map

    def p_miss(self, player_name, play_rates=None):
        """Probability the player misses, from his official status (statuses never seen -> 0)."""
        status = self.injury_map.get(irf.db_name_key(player_name))
        if status is None:
            return 0.0
        rates = play_rates or irf.load_status_play_rates(self.db_path)['all']
        return 1.0 - float(rates.get(status, {'p_play': 1.0})['p_play'])
