# Phase 3 training report (a11_env)

Version `a11_env`, feature set `with_env`, excluded groups [], folds ['2021-22', '2022-23', '2023-24', '2024-25'], 4000 simulations per player-game. The stored pmfs are dispersion-calibrated (Stage F0); the isotonic layer is reported (calibrated columns) but not applied. Smoke check only: no confidence intervals, no ablations, no market comparison. Phase 4 is the formal evaluation.

## Rate-family selection (mean loss across the development folds; lower is better)

| component | chosen |
|---|---|
| fg2a | hier |
| fg3a | lgb |
| fta | hier |
| reb | hier |
| ast | lgb |
| p2 | hier |
| p3 | hier |
| pft | hier |

## Test season 2021-22

### Full-distribution CRPS against rolling-average baselines (played rows, all three decision times)

| stat | n | mean pred | mean obs | CRPS model | CRPS poisson | CRPS normal | PIT<.1 | PIT>.9 | var ratio (obs/pred) |
|---|---|---|---|---|---|---|---|---|---|
| points | 83757 | 10.9224 | 10.4435 | 3.149 | 3.5418 | 3.3101 | 0.0961 | 0.0793 | 0.899 |
| rebounds | 83757 | 4.2332 | 4.1914 | 1.3063 | 1.3728 | 1.3752 | 0.0775 | 0.0887 | 0.9222 |
| assists | 83757 | 2.404 | 2.321 | 0.86 | 0.8982 | 0.9229 | 0.0524 | 0.0866 | 0.9264 |
| threes | 83757 | 1.3103 | 1.1758 | 0.5638 | 0.573 | 0.6028 | 0.0392 | 0.0702 | 0.8981 |
| pra | 83757 | 17.559 | 16.9558 | 4.1514 | 4.7663 | 4.4105 | 0.0926 | 0.0797 | 0.8836 |
| pr | 83757 | 15.1556 | 14.6348 | 3.8334 | 4.3589 | 4.0469 | 0.0924 | 0.0802 | 0.8897 |
| pa | 83757 | 13.3262 | 12.7645 | 3.4521 | 3.9121 | 3.6571 | 0.0937 | 0.0786 | 0.8902 |
| ra | 83757 | 6.6361 | 6.5124 | 1.7713 | 1.904 | 1.8769 | 0.0839 | 0.0854 | 0.907 |

### Calibration of P(X > line), raw vs calibrated (calibration window to test season)

| stat | logloss_raw | ece_raw | logloss_cal | ece_cal |
|---|---|---|---|---|
| points | 0.6602 | 0.0245 | 0.6601 | 0.0132 |
| rebounds | 0.5826 | 0.0078 | 0.584 | 0.0073 |
| assists | 0.5346 | 0.0147 | 0.5373 | 0.012 |
| threes | 0.452 | 0.0305 | 0.4533 | 0.0234 |
| pra | 0.6722 | 0.0265 | 0.6726 | 0.0171 |
| pr | 0.6688 | 0.0251 | 0.6692 | 0.0139 |
| pa | 0.6652 | 0.0274 | 0.6651 | 0.0166 |
| ra | 0.6244 | 0.0136 | 0.6267 | 0.0089 |

## Test season 2022-23

### Full-distribution CRPS against rolling-average baselines (played rows, all three decision times)

| stat | n | mean pred | mean obs | CRPS model | CRPS poisson | CRPS normal | PIT<.1 | PIT>.9 | var ratio (obs/pred) |
|---|---|---|---|---|---|---|---|---|---|
| points | 82956 | 10.7972 | 10.9038 | 3.1619 | 3.5907 | 3.3648 | 0.0935 | 0.1025 | 1.0312 |
| rebounds | 82956 | 4.1284 | 4.1398 | 1.2949 | 1.3731 | 1.3786 | 0.0852 | 0.1003 | 1.0462 |
| assists | 82956 | 2.3708 | 2.4045 | 0.858 | 0.9033 | 0.9276 | 0.0432 | 0.0921 | 0.9453 |
| threes | 82956 | 1.1806 | 1.1745 | 0.5562 | 0.5737 | 0.604 | 0.0302 | 0.087 | 1.0172 |
| pra | 82956 | 17.2964 | 17.4481 | 4.1469 | 4.8236 | 4.4654 | 0.105 | 0.1076 | 1.0571 |
| pr | 82956 | 14.9258 | 15.0436 | 3.8301 | 4.4068 | 4.0945 | 0.1039 | 0.1055 | 1.0464 |
| pa | 82956 | 13.1682 | 13.3083 | 3.4658 | 3.975 | 3.7217 | 0.096 | 0.1045 | 1.027 |
| ra | 82956 | 6.499 | 6.5443 | 1.7573 | 1.9075 | 1.8824 | 0.0913 | 0.1031 | 1.0402 |

### Calibration of P(X > line), raw vs calibrated (calibration window to test season)

| stat | logloss_raw | ece_raw | logloss_cal | ece_cal |
|---|---|---|---|---|
| points | 0.6645 | 0.0085 | 0.665 | 0.0135 |
| rebounds | 0.5808 | 0.0036 | 0.5821 | 0.0118 |
| assists | 0.5378 | 0.0094 | 0.5383 | 0.0085 |
| threes | 0.453 | 0.0039 | 0.4536 | 0.008 |
| pra | 0.6754 | 0.008 | 0.6763 | 0.0192 |
| pr | 0.6727 | 0.0076 | 0.6736 | 0.0179 |
| pa | 0.6692 | 0.0094 | 0.6699 | 0.0143 |
| ra | 0.6237 | 0.0032 | 0.6244 | 0.0121 |

## Test season 2023-24

### Full-distribution CRPS against rolling-average baselines (played rows, all three decision times)

| stat | n | mean pred | mean obs | CRPS model | CRPS poisson | CRPS normal | PIT<.1 | PIT>.9 | var ratio (obs/pred) |
|---|---|---|---|---|---|---|---|---|---|
| points | 84306 | 10.5382 | 10.6519 | 3.0948 | 3.4948 | 3.2799 | 0.0906 | 0.1018 | 1.0188 |
| rebounds | 84306 | 4.0751 | 4.0699 | 1.2749 | 1.3448 | 1.3513 | 0.0788 | 0.0954 | 0.9946 |
| assists | 84306 | 2.4746 | 2.4789 | 0.8851 | 0.9317 | 0.9522 | 0.0512 | 0.0939 | 0.9738 |
| threes | 84306 | 1.121 | 1.1961 | 0.5572 | 0.572 | 0.6011 | 0.0225 | 0.0935 | 1.0509 |
| pra | 84306 | 17.0876 | 17.2008 | 4.1127 | 4.7543 | 4.4066 | 0.0984 | 0.106 | 1.0361 |
| pr | 84306 | 14.6132 | 14.7219 | 3.7774 | 4.3176 | 4.0161 | 0.0968 | 0.1066 | 1.0379 |
| pa | 84306 | 13.0126 | 13.1308 | 3.4178 | 3.8999 | 3.6564 | 0.0907 | 0.1018 | 1.0156 |
| ra | 84306 | 6.5497 | 6.5488 | 1.764 | 1.906 | 1.8805 | 0.0897 | 0.0996 | 1.0083 |

### Calibration of P(X > line), raw vs calibrated (calibration window to test season)

| stat | logloss_raw | ece_raw | logloss_cal | ece_cal |
|---|---|---|---|---|
| points | 0.664 | 0.01 | 0.6641 | 0.0043 |
| rebounds | 0.581 | 0.0042 | 0.5817 | 0.0112 |
| assists | 0.5405 | 0.0027 | 0.5415 | 0.0101 |
| threes | 0.4549 | 0.0171 | 0.4553 | 0.0175 |
| pra | 0.6751 | 0.0058 | 0.6754 | 0.0039 |
| pr | 0.6719 | 0.0072 | 0.6726 | 0.0024 |
| pa | 0.6685 | 0.0106 | 0.6689 | 0.0057 |
| ra | 0.6243 | 0.0047 | 0.6254 | 0.0126 |

## Test season 2024-25

### Full-distribution CRPS against rolling-average baselines (played rows, all three decision times)

| stat | n | mean pred | mean obs | CRPS model | CRPS poisson | CRPS normal | PIT<.1 | PIT>.9 | var ratio (obs/pred) |
|---|---|---|---|---|---|---|---|---|---|
| points | 84399 | 10.7793 | 10.6404 | 3.1414 | 3.5505 | 3.3224 | 0.0996 | 0.097 | 1.0166 |
| rebounds | 84399 | 4.1899 | 4.1243 | 1.2963 | 1.3704 | 1.3729 | 0.0821 | 0.0916 | 0.9812 |
| assists | 84399 | 2.5436 | 2.4659 | 0.8852 | 0.9307 | 0.9504 | 0.0589 | 0.0854 | 0.9492 |
| threes | 84399 | 1.3047 | 1.2624 | 0.582 | 0.6006 | 0.6249 | 0.0402 | 0.0828 | 0.9957 |
| pra | 84399 | 17.5108 | 17.2306 | 4.1313 | 4.781 | 4.4279 | 0.106 | 0.0929 | 0.9866 |
| pr | 84399 | 14.9679 | 14.7647 | 3.8021 | 4.349 | 4.0424 | 0.1031 | 0.0936 | 0.9924 |
| pa | 84399 | 13.3226 | 13.1064 | 3.4547 | 3.9471 | 3.6905 | 0.1034 | 0.0946 | 1.0046 |
| ra | 84399 | 6.732 | 6.5902 | 1.7817 | 1.9287 | 1.9006 | 0.0906 | 0.0911 | 0.9734 |

### Calibration of P(X > line), raw vs calibrated (calibration window to test season)

| stat | logloss_raw | ece_raw | logloss_cal | ece_cal |
|---|---|---|---|---|
| points | 0.6628 | 0.0098 | 0.6628 | 0.0052 |
| rebounds | 0.5831 | 0.0117 | 0.5833 | 0.0053 |
| assists | 0.5355 | 0.0127 | 0.536 | 0.0089 |
| threes | 0.4553 | 0.0121 | 0.4555 | 0.0051 |
| pra | 0.6733 | 0.0146 | 0.673 | 0.0029 |
| pr | 0.6703 | 0.0129 | 0.6701 | 0.0039 |
| pa | 0.6669 | 0.0116 | 0.6671 | 0.0037 |
| ra | 0.6246 | 0.0195 | 0.6242 | 0.0098 |
