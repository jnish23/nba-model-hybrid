# E12_line_shopping

`odds_snapshots` rows in the database: **0** -> no timestamped multi-book snapshots exist; the multi-book source is the per-book columns of `nba_odds_opening` (opening quotes, up to 6 books). Coverage below is therefore the opening-line coverage.

## Coverage (share of test games with >= 2 books quoting the market; mean number of books)

| seasonYear   |   games |   spread >=2 books |   total >=2 books |   moneyline >=2 books |   spread mean books |   total mean books |   moneyline mean books |
|:-------------|--------:|-------------------:|------------------:|----------------------:|--------------------:|-------------------:|-----------------------:|
| 2019-20      |    1143 |               1    |              1    |                  0.99 |                4.03 |               4.02 |                   3.8  |
| 2020-21      |    1171 |               1    |              1    |                  0.99 |                3.96 |               3.98 |                   3.71 |
| 2021-22      |    1323 |               1    |              1    |                  1    |                4.94 |               4.93 |                   4.65 |
| 2022-23      |    1320 |               1    |              1    |                  1    |                4.97 |               4.96 |                   4.74 |
| 2023-24      |    1318 |               0.99 |              0.99 |                  0.99 |                4.89 |               4.92 |                   4.59 |
| 2024-25      |    1320 |               1    |              1    |                  1    |                4.95 |               4.98 |                   4.71 |
| 2025-26      |    1321 |               1    |              1    |                  1    |                4.97 |               4.99 |                   4.8  |

## Spread / total bets settled at the best available line across books

| market   |   bets with data |   bets where best line is better |   mean improvement (pts) all bets |   ROI DK line |   ROI consensus median line |   ROI best line @ -110 | best line vs DK diff [95% CI]   |   ROI best line @ quoted price | quoted-price diff [95% CI]   |
|:---------|-----------------:|---------------------------------:|----------------------------------:|--------------:|----------------------------:|-----------------------:|:--------------------------------|-------------------------------:|:-----------------------------|
| spread   |              644 |                              289 |                             0.273 |         -3.66 |                       -3.77 |                  -0.35 | +3.30 [+1.40, +5.32]            |                          -0.34 | +3.32 [+1.44, +5.35]         |
| total    |             1973 |                              947 |                             0.446 |          1.12 |                       -0.02 |                   2.77 | +1.66 [+0.61, +2.71]            |                           2.69 | +1.57 [+0.52, +2.62]         |

## Realised price improvement by key number (spread bets; improvement measured in points of handicap)

| bucket                 |   bets |   mean_improvement_pts |   result_changes |   roi_dk |   roi_best_line |   roi_gain |
|:-----------------------|-------:|-----------------------:|-----------------:|---------:|----------------:|-----------:|
| crosses 3              |      5 |                  1.2   |            0.2   |   14.545 |          52.727 |     38.182 |
| crosses 7              |     26 |                  1.346 |            0.115 |  -11.888 |           3.147 |     15.035 |
| no improvement         |    355 |                 -0.22  |            0.003 |   -0.512 |          -1.05  |     -0.538 |
| other half-point shift |    258 |                  0.824 |            0.043 |   -7.505 |          -0.775 |      6.73  |

Total bets: improvement distribution (points): {'count': 1973.0, 'mean': 0.446, 'std': 1.969, 'min': -36.0, '25%': 0.0, '50%': 0.0, '75%': 1.0, 'max': 7.0}

## Moneyline bets settled at the best price across books

| market   |   bets with data |   ROI DK price |   ROI median price |   ROI best price | best vs DK diff [95% CI]   |
|:---------|-----------------:|---------------:|-------------------:|-----------------:|:---------------------------|
| ml       |             1765 |           0.46 |              -0.84 |              1.9 | +1.44 [+0.45, +2.35]       |

## Control: bet BOTH sides of every game at the best available price/line (an efficient market should lose here)

- moneyline, both sides at best price: ROI -1.34% over 8884 games
- spread, both sides at the best line (-110): ROI -1.62% over 8898 games

## Consensus de-vigged probability as the reference (8366 games with open and close consensus)

- log loss: model 0.6193 | consensus open 0.6076 | consensus close 0.6001
- mean |model - open consensus| = 0.045; share of games with model-open gap > 5 pts: 36.0%

## Decision (pre-registered: adopt best-price settlement as a reporting option only if ROI improves with a CI excluding zero AND the both-sides control has ROI <= 0)

- spread: **ADOPT as reporting option**
- total: **ADOPT as reporting option**
- ml: **ADOPT as reporting option**