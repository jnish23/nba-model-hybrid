# Phase 3 training report (v1_base)

Version `v1_base`, feature set `base`, excluded groups [], folds ['2021-22', '2022-23', '2023-24'], 4000 simulations per player-game. Smoke check only: no confidence intervals, no ablations, no market comparison. Phase 4 is the formal evaluation.

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
| points | 83757 | 10.6318 | 10.4435 | 3.1746 | 3.5418 | 3.3101 | 0.115 | 0.1056 | 1.0951 |
| rebounds | 83757 | 4.1396 | 4.1914 | 1.3132 | 1.3728 | 1.3752 | 0.0871 | 0.1055 | 1.0757 |
| assists | 83757 | 2.3393 | 2.321 | 0.8654 | 0.8982 | 0.9229 | 0.0538 | 0.0973 | 1.0322 |
| threes | 83757 | 1.2622 | 1.1758 | 0.5645 | 0.573 | 0.6028 | 0.0471 | 0.0825 | 1.058 |
| pra | 83757 | 17.1109 | 16.9558 | 4.199 | 4.7663 | 4.4105 | 0.1304 | 0.1153 | 1.1679 |
| pr | 83757 | 14.7716 | 14.6348 | 3.8707 | 4.3589 | 4.0469 | 0.127 | 0.1144 | 1.1615 |
| pa | 83757 | 12.9714 | 12.7645 | 3.4878 | 3.9121 | 3.6571 | 0.1191 | 0.1061 | 1.1023 |
| ra | 83757 | 6.4787 | 6.5124 | 1.7854 | 1.904 | 1.8769 | 0.102 | 0.1097 | 1.1153 |

### Calibration of P(X > line), raw vs calibrated (calibration window to test season)

| stat | logloss_raw | ece_raw | logloss_cal | ece_cal |
|---|---|---|---|---|
| points | 0.6645 | 0.0136 | 0.6655 | 0.0066 |
| rebounds | 0.5845 | 0.0044 | 0.5872 | 0.0102 |
| assists | 0.5373 | 0.0107 | 0.5409 | 0.017 |
| threes | 0.456 | 0.026 | 0.4566 | 0.0164 |
| pra | 0.6754 | 0.0114 | 0.6764 | 0.011 |
| pr | 0.6722 | 0.0109 | 0.6729 | 0.0111 |
| pa | 0.6686 | 0.015 | 0.6688 | 0.0092 |
| ra | 0.627 | 0.0085 | 0.628 | 0.0163 |

## Test season 2022-23

### Full-distribution CRPS against rolling-average baselines (played rows, all three decision times)

| stat | n | mean pred | mean obs | CRPS model | CRPS poisson | CRPS normal | PIT<.1 | PIT>.9 | var ratio (obs/pred) |
|---|---|---|---|---|---|---|---|---|---|
| points | 82956 | 10.9408 | 10.9038 | 3.188 | 3.5907 | 3.3648 | 0.1066 | 0.1045 | 1.0875 |
| rebounds | 82956 | 4.1705 | 4.1398 | 1.3036 | 1.3731 | 1.3786 | 0.0902 | 0.0988 | 1.0523 |
| assists | 82956 | 2.3791 | 2.4045 | 0.865 | 0.9033 | 0.9276 | 0.047 | 0.0969 | 0.9835 |
| threes | 82956 | 1.1953 | 1.1745 | 0.559 | 0.5737 | 0.604 | 0.0391 | 0.089 | 1.1022 |
| pra | 82956 | 17.4911 | 17.4481 | 4.195 | 4.8236 | 4.4654 | 0.1228 | 0.1134 | 1.1367 |
| pr | 82956 | 15.112 | 15.0436 | 3.8694 | 4.4068 | 4.0945 | 0.1229 | 0.1097 | 1.1343 |
| pa | 82956 | 13.3206 | 13.3083 | 3.5002 | 3.975 | 3.7217 | 0.112 | 0.1079 | 1.0876 |
| ra | 82956 | 6.5494 | 6.5443 | 1.7744 | 1.9075 | 1.8824 | 0.1017 | 0.1044 | 1.078 |

### Calibration of P(X > line), raw vs calibrated (calibration window to test season)

| stat | logloss_raw | ece_raw | logloss_cal | ece_cal |
|---|---|---|---|---|
| points | 0.6639 | 0.0034 | 0.6645 | 0.0107 |
| rebounds | 0.5828 | 0.0086 | 0.5832 | 0.0061 |
| assists | 0.5404 | 0.006 | 0.5411 | 0.0076 |
| threes | 0.4551 | 0.0115 | 0.4554 | 0.0084 |
| pra | 0.675 | 0.0047 | 0.6753 | 0.0072 |
| pr | 0.6725 | 0.0062 | 0.6731 | 0.0063 |
| pa | 0.6688 | 0.0042 | 0.6695 | 0.0093 |
| ra | 0.6252 | 0.0069 | 0.6255 | 0.0087 |

## Test season 2023-24

### Full-distribution CRPS against rolling-average baselines (played rows, all three decision times)

| stat | n | mean pred | mean obs | CRPS model | CRPS poisson | CRPS normal | PIT<.1 | PIT>.9 | var ratio (obs/pred) |
|---|---|---|---|---|---|---|---|---|---|
| points | 84306 | 11.2125 | 10.6519 | 3.1175 | 3.4948 | 3.2799 | 0.1203 | 0.0873 | 1.0234 |
| rebounds | 84306 | 4.1352 | 4.0699 | 1.2785 | 1.3448 | 1.3513 | 0.0882 | 0.0933 | 1.0104 |
| assists | 84306 | 2.5087 | 2.4789 | 0.8897 | 0.9317 | 0.9522 | 0.0578 | 0.0944 | 1.0013 |
| threes | 84306 | 1.2509 | 1.1961 | 0.557 | 0.572 | 0.6011 | 0.0399 | 0.0807 | 1.0366 |
| pra | 84306 | 17.8567 | 17.2008 | 4.1482 | 4.7543 | 4.4066 | 0.1368 | 0.095 | 1.0857 |
| pr | 84306 | 15.348 | 14.7219 | 3.8075 | 4.3176 | 4.0161 | 0.1351 | 0.094 | 1.0827 |
| pa | 84306 | 13.7216 | 13.1308 | 3.4458 | 3.8999 | 3.6564 | 0.126 | 0.0886 | 1.0264 |
| ra | 84306 | 6.6436 | 6.5488 | 1.7724 | 1.906 | 1.8805 | 0.1049 | 0.0985 | 1.0558 |

### Calibration of P(X > line), raw vs calibrated (calibration window to test season)

| stat | logloss_raw | ece_raw | logloss_cal | ece_cal |
|---|---|---|---|---|
| points | 0.6586 | 0.036 | 0.658 | 0.0303 |
| rebounds | 0.5798 | 0.0145 | 0.5803 | 0.0139 |
| assists | 0.5402 | 0.0087 | 0.5408 | 0.0119 |
| threes | 0.449 | 0.0192 | 0.4485 | 0.0122 |
| pra | 0.6704 | 0.0359 | 0.6699 | 0.0308 |
| pr | 0.6666 | 0.0368 | 0.6659 | 0.0315 |
| pa | 0.6634 | 0.0342 | 0.6632 | 0.03 |
| ra | 0.6244 | 0.0169 | 0.6248 | 0.0173 |
