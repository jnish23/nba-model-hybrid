# Phase 3 training report (a11_pg2)

Version `a11_pg2`, feature set `base`, excluded groups [], folds ['2021-22', '2022-23', '2023-24', '2024-25'], 4000 simulations per player-game. The stored pmfs are dispersion-calibrated (Stage F0); the isotonic layer is reported (calibrated columns) but not applied. Smoke check only: no confidence intervals, no ablations, no market comparison. Phase 4 is the formal evaluation.

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
| points | 83757 | 10.9347 | 10.4435 | 3.1493 | 3.5418 | 3.3101 | 0.0981 | 0.0803 | 0.903 |
| rebounds | 83757 | 4.2405 | 4.1914 | 1.3064 | 1.3728 | 1.3752 | 0.079 | 0.0892 | 0.9291 |
| assists | 83757 | 2.395 | 2.321 | 0.8602 | 0.8982 | 0.9229 | 0.0531 | 0.0882 | 0.9343 |
| threes | 83757 | 1.3006 | 1.1758 | 0.563 | 0.573 | 0.6028 | 0.0382 | 0.0717 | 0.9039 |
| pra | 83757 | 17.5696 | 16.9558 | 4.1523 | 4.7663 | 4.4105 | 0.0943 | 0.0802 | 0.8915 |
| pr | 83757 | 15.1752 | 14.6348 | 3.8341 | 4.3589 | 4.0469 | 0.0935 | 0.0812 | 0.8961 |
| pa | 83757 | 13.3295 | 12.7645 | 3.4526 | 3.9121 | 3.6571 | 0.0946 | 0.0794 | 0.8957 |
| ra | 83757 | 6.6344 | 6.5124 | 1.7717 | 1.904 | 1.8769 | 0.0856 | 0.0869 | 0.918 |

### Calibration of P(X > line), raw vs calibrated (calibration window to test season)

| stat | logloss_raw | ece_raw | logloss_cal | ece_cal |
|---|---|---|---|---|
| points | 0.6603 | 0.0252 | 0.6602 | 0.0135 |
| rebounds | 0.5828 | 0.0073 | 0.5844 | 0.0083 |
| assists | 0.5348 | 0.0122 | 0.5372 | 0.0148 |
| threes | 0.4521 | 0.0286 | 0.4533 | 0.0216 |
| pra | 0.6721 | 0.0266 | 0.6741 | 0.0186 |
| pr | 0.6689 | 0.0257 | 0.669 | 0.0144 |
| pa | 0.6653 | 0.027 | 0.6656 | 0.0169 |
| ra | 0.6243 | 0.0132 | 0.6257 | 0.0077 |

## Test season 2022-23

### Full-distribution CRPS against rolling-average baselines (played rows, all three decision times)

| stat | n | mean pred | mean obs | CRPS model | CRPS poisson | CRPS normal | PIT<.1 | PIT>.9 | var ratio (obs/pred) |
|---|---|---|---|---|---|---|---|---|---|
| points | 82956 | 10.8185 | 10.9038 | 3.1651 | 3.5907 | 3.3648 | 0.0936 | 0.1018 | 1.0314 |
| rebounds | 82956 | 4.13 | 4.1398 | 1.2963 | 1.3731 | 1.3786 | 0.0854 | 0.0996 | 1.0443 |
| assists | 82956 | 2.3691 | 2.4045 | 0.8584 | 0.9033 | 0.9276 | 0.0436 | 0.092 | 0.9486 |
| threes | 82956 | 1.1856 | 1.1745 | 0.5563 | 0.5737 | 0.604 | 0.0308 | 0.0863 | 1.0159 |
| pra | 82956 | 17.3175 | 17.4481 | 4.1525 | 4.8236 | 4.4654 | 0.1052 | 0.1066 | 1.0529 |
| pr | 82956 | 14.9488 | 15.0436 | 3.8347 | 4.4068 | 4.0945 | 0.1041 | 0.1044 | 1.0449 |
| pa | 82956 | 13.1877 | 13.3083 | 3.4701 | 3.975 | 3.7217 | 0.0958 | 0.103 | 1.0249 |
| ra | 82956 | 6.4993 | 6.5443 | 1.7591 | 1.9075 | 1.8824 | 0.0906 | 0.1026 | 1.0375 |

### Calibration of P(X > line), raw vs calibrated (calibration window to test season)

| stat | logloss_raw | ece_raw | logloss_cal | ece_cal |
|---|---|---|---|---|
| points | 0.6645 | 0.0074 | 0.6653 | 0.015 |
| rebounds | 0.5808 | 0.0029 | 0.5816 | 0.0118 |
| assists | 0.5377 | 0.0095 | 0.5387 | 0.009 |
| threes | 0.4527 | 0.0054 | 0.4535 | 0.0084 |
| pra | 0.6754 | 0.0071 | 0.6764 | 0.0199 |
| pr | 0.673 | 0.0067 | 0.6738 | 0.019 |
| pa | 0.669 | 0.0081 | 0.6696 | 0.0157 |
| ra | 0.6235 | 0.0029 | 0.6239 | 0.012 |

## Test season 2023-24

### Full-distribution CRPS against rolling-average baselines (played rows, all three decision times)

| stat | n | mean pred | mean obs | CRPS model | CRPS poisson | CRPS normal | PIT<.1 | PIT>.9 | var ratio (obs/pred) |
|---|---|---|---|---|---|---|---|---|---|
| points | 84306 | 10.9287 | 10.6519 | 3.0966 | 3.4948 | 3.2799 | 0.0971 | 0.0885 | 0.9679 |
| rebounds | 84306 | 4.0713 | 4.0699 | 1.275 | 1.3448 | 1.3513 | 0.0783 | 0.0962 | 0.9963 |
| assists | 84306 | 2.4642 | 2.4789 | 0.8861 | 0.9317 | 0.9522 | 0.0505 | 0.0958 | 0.9781 |
| threes | 84306 | 1.2125 | 1.1961 | 0.5555 | 0.572 | 0.6011 | 0.0285 | 0.0799 | 0.9574 |
| pra | 84306 | 17.4638 | 17.2008 | 4.1145 | 4.7543 | 4.4066 | 0.103 | 0.0944 | 0.9955 |
| pr | 84306 | 15 | 14.7219 | 3.7796 | 4.3176 | 4.0161 | 0.1016 | 0.0936 | 0.9939 |
| pa | 84306 | 13.3927 | 13.1308 | 3.4191 | 3.8999 | 3.6564 | 0.0965 | 0.0905 | 0.9692 |
| ra | 84306 | 6.5353 | 6.5488 | 1.7652 | 1.906 | 1.8805 | 0.088 | 0.1002 | 1.0068 |

### Calibration of P(X > line), raw vs calibrated (calibration window to test season)

| stat | logloss_raw | ece_raw | logloss_cal | ece_cal |
|---|---|---|---|---|
| points | 0.6612 | 0.0141 | 0.662 | 0.0208 |
| rebounds | 0.5814 | 0.0036 | 0.5821 | 0.0118 |
| assists | 0.5416 | 0.0044 | 0.5425 | 0.0101 |
| threes | 0.4494 | 0.0068 | 0.4496 | 0.0062 |
| pra | 0.6731 | 0.0132 | 0.674 | 0.0221 |
| pr | 0.6699 | 0.015 | 0.671 | 0.0222 |
| pa | 0.6657 | 0.0114 | 0.6665 | 0.0205 |
| ra | 0.6247 | 0.0033 | 0.6255 | 0.0132 |

## Test season 2024-25

### Full-distribution CRPS against rolling-average baselines (played rows, all three decision times)

| stat | n | mean pred | mean obs | CRPS model | CRPS poisson | CRPS normal | PIT<.1 | PIT>.9 | var ratio (obs/pred) |
|---|---|---|---|---|---|---|---|---|---|
| points | 84399 | 10.7395 | 10.6404 | 3.1425 | 3.5505 | 3.3224 | 0.0996 | 0.0995 | 1.0309 |
| rebounds | 84399 | 4.1808 | 4.1243 | 1.2956 | 1.3704 | 1.3729 | 0.0812 | 0.0919 | 0.9808 |
| assists | 84399 | 2.5408 | 2.4659 | 0.8853 | 0.9307 | 0.9504 | 0.0583 | 0.0856 | 0.9428 |
| threes | 84399 | 1.3009 | 1.2624 | 0.5822 | 0.6006 | 0.6249 | 0.0396 | 0.083 | 0.996 |
| pra | 84399 | 17.4593 | 17.2306 | 4.1321 | 4.781 | 4.4279 | 0.1051 | 0.0954 | 0.9969 |
| pr | 84399 | 14.9188 | 14.7647 | 3.8029 | 4.349 | 4.0424 | 0.103 | 0.0963 | 1.0042 |
| pa | 84399 | 13.2802 | 13.1064 | 3.4557 | 3.9471 | 3.6905 | 0.104 | 0.0969 | 1.0164 |
| ra | 84399 | 6.7205 | 6.5902 | 1.7813 | 1.9287 | 1.9006 | 0.0898 | 0.0922 | 0.9705 |

### Calibration of P(X > line), raw vs calibrated (calibration window to test season)

| stat | logloss_raw | ece_raw | logloss_cal | ece_cal |
|---|---|---|---|---|
| points | 0.6631 | 0.0075 | 0.6634 | 0.0129 |
| rebounds | 0.5829 | 0.0104 | 0.583 | 0.006 |
| assists | 0.5361 | 0.0117 | 0.5366 | 0.0104 |
| threes | 0.456 | 0.0111 | 0.4561 | 0.0069 |
| pra | 0.6738 | 0.0123 | 0.6738 | 0.0056 |
| pr | 0.6707 | 0.0105 | 0.6709 | 0.009 |
| pa | 0.6674 | 0.0093 | 0.6677 | 0.0084 |
| ra | 0.6252 | 0.0179 | 0.6251 | 0.0115 |
