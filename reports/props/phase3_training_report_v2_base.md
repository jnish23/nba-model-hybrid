# Phase 3 training report (v2_base)

Version `v2_base`, feature set `base`, excluded groups [], folds ['2021-22', '2022-23', '2023-24'], 4000 simulations per player-game. The stored pmfs are dispersion-calibrated (Stage F0); the isotonic layer is reported (calibrated columns) but not applied. Smoke check only: no confidence intervals, no ablations, no market comparison. Phase 4 is the formal evaluation.

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

## Test season 2021-22

### Full-distribution CRPS against rolling-average baselines (played rows, all three decision times)

| stat | n | mean pred | mean obs | CRPS model | CRPS poisson | CRPS normal | PIT<.1 | PIT>.9 | var ratio (obs/pred) |
|---|---|---|---|---|---|---|---|---|---|
| points | 83757 | 10.632 | 10.4435 | 3.1704 | 3.5418 | 3.3101 | 0.0915 | 0.0905 | 0.9529 |
| rebounds | 83757 | 4.1398 | 4.1914 | 1.3134 | 1.3728 | 1.3752 | 0.0731 | 0.0975 | 0.9763 |
| assists | 83757 | 2.3398 | 2.321 | 0.8651 | 0.8982 | 0.9229 | 0.0476 | 0.0923 | 0.9715 |
| threes | 83757 | 1.2629 | 1.1758 | 0.5634 | 0.573 | 0.6028 | 0.0358 | 0.0761 | 0.9383 |
| pra | 83757 | 17.1109 | 16.9558 | 4.1909 | 4.7663 | 4.4105 | 0.0866 | 0.0936 | 0.9488 |
| pr | 83757 | 14.7716 | 14.6348 | 3.8636 | 4.3589 | 4.0469 | 0.0863 | 0.0937 | 0.9507 |
| pa | 83757 | 12.9715 | 12.7645 | 3.4825 | 3.9121 | 3.6571 | 0.088 | 0.0905 | 0.9487 |
| ra | 83757 | 6.4787 | 6.5124 | 1.7844 | 1.904 | 1.8769 | 0.0794 | 0.0956 | 0.9659 |

### Calibration of P(X > line), raw vs calibrated (calibration window to test season)

| stat | logloss_raw | ece_raw | logloss_cal | ece_cal |
|---|---|---|---|---|
| points | 0.6641 | 0.0068 | 0.6654 | 0.0064 |
| rebounds | 0.5845 | 0.0066 | 0.5875 | 0.0107 |
| assists | 0.537 | 0.0071 | 0.5411 | 0.0164 |
| threes | 0.4549 | 0.0204 | 0.4574 | 0.0185 |
| pra | 0.675 | 0.0072 | 0.6763 | 0.0105 |
| pr | 0.6718 | 0.0065 | 0.6727 | 0.0105 |
| pa | 0.6682 | 0.0089 | 0.6688 | 0.0084 |
| ra | 0.6267 | 0.0028 | 0.6289 | 0.0162 |

## Test season 2022-23

### Full-distribution CRPS against rolling-average baselines (played rows, all three decision times)

| stat | n | mean pred | mean obs | CRPS model | CRPS poisson | CRPS normal | PIT<.1 | PIT>.9 | var ratio (obs/pred) |
|---|---|---|---|---|---|---|---|---|---|
| points | 82956 | 10.941 | 10.9038 | 3.1859 | 3.5907 | 3.3648 | 0.0918 | 0.094 | 0.9867 |
| rebounds | 82956 | 4.1708 | 4.1398 | 1.3034 | 1.3731 | 1.3786 | 0.0841 | 0.0952 | 1.0143 |
| assists | 82956 | 2.3794 | 2.4045 | 0.8661 | 0.9033 | 0.9276 | 0.0403 | 0.0889 | 0.9179 |
| threes | 82956 | 1.196 | 1.1745 | 0.5586 | 0.5737 | 0.604 | 0.0303 | 0.0832 | 0.9934 |
| pra | 82956 | 17.4911 | 17.4481 | 4.189 | 4.8236 | 4.4654 | 0.0964 | 0.0984 | 0.9946 |
| pr | 82956 | 15.1121 | 15.0436 | 3.864 | 4.4068 | 4.0945 | 0.0974 | 0.0967 | 0.9905 |
| pa | 82956 | 13.3206 | 13.3083 | 3.4973 | 3.975 | 3.7217 | 0.0916 | 0.097 | 0.9901 |
| ra | 82956 | 6.5494 | 6.5443 | 1.7742 | 1.9075 | 1.8824 | 0.0854 | 0.096 | 0.9984 |

### Calibration of P(X > line), raw vs calibrated (calibration window to test season)

| stat | logloss_raw | ece_raw | logloss_cal | ece_cal |
|---|---|---|---|---|
| points | 0.6638 | 0.0035 | 0.6644 | 0.0095 |
| rebounds | 0.5827 | 0.0069 | 0.5833 | 0.0068 |
| assists | 0.5413 | 0.0127 | 0.5415 | 0.0084 |
| threes | 0.4546 | 0.0057 | 0.4555 | 0.0082 |
| pra | 0.6749 | 0.0046 | 0.6753 | 0.0069 |
| pr | 0.6724 | 0.0036 | 0.6731 | 0.0065 |
| pa | 0.6688 | 0.0038 | 0.6695 | 0.0093 |
| ra | 0.6251 | 0.0035 | 0.6256 | 0.0088 |

## Test season 2023-24

### Full-distribution CRPS against rolling-average baselines (played rows, all three decision times)

| stat | n | mean pred | mean obs | CRPS model | CRPS poisson | CRPS normal | PIT<.1 | PIT>.9 | var ratio (obs/pred) |
|---|---|---|---|---|---|---|---|---|---|
| points | 84306 | 11.2126 | 10.6519 | 3.1141 | 3.4948 | 3.2799 | 0.1058 | 0.0787 | 0.9433 |
| rebounds | 84306 | 4.1358 | 4.0699 | 1.2784 | 1.3448 | 1.3513 | 0.0834 | 0.0909 | 0.982 |
| assists | 84306 | 2.5089 | 2.4789 | 0.8895 | 0.9317 | 0.9522 | 0.0536 | 0.0913 | 0.9663 |
| threes | 84306 | 1.2513 | 1.1961 | 0.5564 | 0.572 | 0.6011 | 0.0319 | 0.0748 | 0.9307 |
| pra | 84306 | 17.8567 | 17.2008 | 4.1408 | 4.7543 | 4.4066 | 0.1078 | 0.0828 | 0.963 |
| pr | 84306 | 15.348 | 14.7219 | 3.8007 | 4.3176 | 4.0161 | 0.1082 | 0.0823 | 0.9626 |
| pa | 84306 | 13.7217 | 13.1308 | 3.4417 | 3.8999 | 3.6564 | 0.105 | 0.0797 | 0.9453 |
| ra | 84306 | 6.6437 | 6.5488 | 1.7715 | 1.906 | 1.8805 | 0.0918 | 0.0913 | 0.9829 |

### Calibration of P(X > line), raw vs calibrated (calibration window to test season)

| stat | logloss_raw | ece_raw | logloss_cal | ece_cal |
|---|---|---|---|---|
| points | 0.658 | 0.0329 | 0.6578 | 0.0301 |
| rebounds | 0.5797 | 0.0136 | 0.5803 | 0.0138 |
| assists | 0.54 | 0.005 | 0.5408 | 0.012 |
| threes | 0.4485 | 0.015 | 0.4483 | 0.0123 |
| pra | 0.67 | 0.0343 | 0.6698 | 0.0308 |
| pr | 0.6662 | 0.035 | 0.6658 | 0.0314 |
| pa | 0.6629 | 0.0316 | 0.663 | 0.0299 |
| ra | 0.6241 | 0.0153 | 0.6245 | 0.0172 |
