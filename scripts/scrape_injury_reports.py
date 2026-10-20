"""Entry point: download the official NBA injury report PDFs. The implementation lives in core/scrape_injury_reports.py."""
import os
import sys

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from core.scrape_injury_reports import main

if __name__ == "__main__":
    main()
