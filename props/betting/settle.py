"""
Settlement of simulated prop bets under the bounding treatments of reports/props/void_rules.md.

Common rule: a player who does not play voids the bet at every book (stake refunded). A push refunds. What differs is an early exit:
  stands                    the bet stands whenever the player took the court (the common rule)
  early_exit_flagged_books  at books with an early-exit rule an OVER is voided when minutes fall below EARLY_EXIT_FRACTION of expected minutes (a proxy for an
                            injury exit in the 1st/2nd quarter, which the box score cannot tell apart from other short nights); the under stands
  early_exit_all_books      the same at every book: the strict bound, used as a sensitivity
Profit is per unit staked; a void or push is 0 and does not count as a stake.
"""
import numpy as np

EARLY_EXIT_BOOKS = ('fanatics', 'draftkings')        # Fanatics: primary text (house rules); DraftKings: discretionary refunds reported, low confidence
EARLY_EXIT_FRACTION = 0.5
TREATMENTS = ('stands', 'early_exit_flagged_books', 'early_exit_all_books')


def settle(side, line, dec_price, y, played, minutes, expected_minutes, book, treatment='stands'):
    """Vectorized. side: 'over'/'under' per bet. Returns (profit, status) with status in win/loss/push/void."""
    if treatment not in TREATMENTS:
        raise ValueError(treatment)
    side = np.asarray(side)
    line, dec_price = np.asarray(line, float), np.asarray(dec_price, float)
    y, minutes, expected_minutes = np.asarray(y, float), np.asarray(minutes, float), np.asarray(expected_minutes, float)
    played = np.asarray(played, bool) & ~np.isnan(y)
    over = side == 'over'
    win = np.where(over, y > line, y < line)
    push = y == line
    void = ~played
    if treatment != 'stands':
        flagged = np.ones(len(side), bool) if treatment == 'early_exit_all_books' else np.isin(np.asarray(book), EARLY_EXIT_BOOKS)
        early = minutes < EARLY_EXIT_FRACTION * expected_minutes
        void = void | (over & flagged & early)
    status = np.where(void, 'void', np.where(push, 'push', np.where(win, 'win', 'loss')))
    profit = np.where(status == 'win', dec_price - 1.0, np.where(status == 'loss', -1.0, 0.0))
    return profit, status
