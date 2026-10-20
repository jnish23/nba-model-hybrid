# Phase 3 training report (r1_2023-24)

Version `r1_2023-24`, feature set `base`, excluded groups [], folds ['2023-24'], 4000 simulations per player-game. The stored pmfs are dispersion-calibrated (Stage F0); the isotonic layer is reported (calibrated columns) but not applied. Smoke check only: no confidence intervals, no ablations, no market comparison. Phase 4 is the formal evaluation.

## Rate-family selection (mean loss across the development folds; lower is better)

| component | lgb_loss | hier_loss | chosen |
|---|---|---|---|
| fg2a | 1.11956 | 1.11802 | hier |
| fg3a | 0.90405 | 0.93067 | lgb |
| fta | 1.95228 | 1.95031 | hier |
| reb | 1.1158 | 1.11561 | hier |
| ast | 1.12885 | 1.13011 | lgb |
| p2 | 0.68481 | 0.68429 | hier |
| p3 | 0.65221 | 0.65251 | hier |
| pft | 0.51027 | 0.5106 | hier |

## Test season 2023-24

### Full-distribution CRPS against rolling-average baselines (played rows, all three decision times)

| stat | n | mean pred | mean obs | CRPS model | CRPS poisson | CRPS normal | PIT<.1 | PIT>.9 | var ratio (obs/pred) |
|---|---|---|---|---|---|---|---|---|---|
| points | 84306 | 11.253 | 10.6519 | 3.1184 | 3.4948 | 3.2799 | 0.1049 | 0.0774 | 0.9364 |
| rebounds | 84306 | 4.1337 | 4.0699 | 1.2792 | 1.3448 | 1.3513 | 0.082 | 0.0906 | 0.9768 |
| assists | 84306 | 2.5155 | 2.4789 | 0.8899 | 0.9317 | 0.9522 | 0.0535 | 0.0899 | 0.9608 |
| threes | 84306 | 1.2563 | 1.1961 | 0.5567 | 0.572 | 0.6011 | 0.032 | 0.0736 | 0.925 |
| pra | 84306 | 17.9017 | 17.2008 | 4.1472 | 4.7543 | 4.4066 | 0.1075 | 0.081 | 0.9541 |
| pr | 84306 | 15.3864 | 14.7219 | 3.8067 | 4.3176 | 4.0161 | 0.1069 | 0.0812 | 0.955 |
| pa | 84306 | 13.7687 | 13.1308 | 3.4464 | 3.8999 | 3.6564 | 0.1045 | 0.0779 | 0.9367 |
| ra | 84306 | 6.6482 | 6.5488 | 1.7733 | 1.906 | 1.8805 | 0.0912 | 0.0908 | 0.9775 |

### Calibration of P(X > line), raw vs calibrated (calibration window to test season)

| stat | logloss_raw | ece_raw | logloss_cal | ece_cal |
|---|---|---|---|---|
| points | 0.6577 | 0.0349 | 0.6582 | 0.035 |
| rebounds | 0.5801 | 0.0133 | 0.5805 | 0.014 |
| assists | 0.5398 | 0.0055 | 0.5411 | 0.0138 |
| threes | 0.4489 | 0.0158 | 0.4495 | 0.0131 |
| pra | 0.6696 | 0.0361 | 0.6699 | 0.0355 |
| pr | 0.6661 | 0.0367 | 0.6665 | 0.0356 |
| pa | 0.6626 | 0.0338 | 0.6632 | 0.0353 |
| ra | 0.6243 | 0.0153 | 0.6251 | 0.0185 |
