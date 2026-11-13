# Phase 3 training report (a11_r0)

Version `a11_r0`, feature set `base`, excluded groups [], folds ['2021-22', '2022-23', '2023-24', '2024-25'], 4000 simulations per player-game. The stored pmfs are dispersion-calibrated (Stage F0); the isotonic layer is reported (calibrated columns) but not applied. Smoke check only: no confidence intervals, no ablations, no market comparison. Phase 4 is the formal evaluation.

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
| points | 83757 | 10.4206 | 10.4435 | 3.1647 | 3.5418 | 3.3101 | 0.0806 | 0.0942 | 0.9316 |
| rebounds | 83757 | 4.0584 | 4.1914 | 1.3138 | 1.3728 | 1.3752 | 0.0643 | 0.1006 | 0.9532 |
| assists | 83757 | 2.2808 | 2.321 | 0.865 | 0.8982 | 0.9229 | 0.0424 | 0.0992 | 0.9717 |
| threes | 83757 | 1.2383 | 1.1758 | 0.5618 | 0.573 | 0.6028 | 0.0333 | 0.0795 | 0.9405 |
| pra | 83757 | 16.7589 | 16.9558 | 4.1878 | 4.7663 | 4.4105 | 0.0721 | 0.0977 | 0.9114 |
| pr | 83757 | 14.4786 | 14.6348 | 3.8609 | 4.3589 | 4.0469 | 0.0742 | 0.0965 | 0.9171 |
| pa | 83757 | 12.7011 | 12.7645 | 3.4766 | 3.9121 | 3.6571 | 0.0754 | 0.0952 | 0.9218 |
| ra | 83757 | 6.3381 | 6.5124 | 1.7851 | 1.904 | 1.8769 | 0.0677 | 0.1002 | 0.9418 |

### Calibration of P(X > line), raw vs calibrated (calibration window to test season)

| stat | logloss_raw | ece_raw | logloss_cal | ece_cal |
|---|---|---|---|---|
| points | 0.6678 | 0.0125 | 0.6698 | 0.0209 |
| rebounds | 0.5869 | 0.0214 | 0.5904 | 0.0196 |
| assists | 0.5402 | 0.0109 | 0.5499 | 0.0283 |
| threes | 0.4556 | 0.0129 | 0.4593 | 0.0104 |
| pra | 0.6792 | 0.0177 | 0.6839 | 0.0287 |
| pr | 0.6758 | 0.0165 | 0.6799 | 0.0267 |
| pa | 0.6725 | 0.0141 | 0.6754 | 0.0247 |
| ra | 0.6288 | 0.0218 | 0.6319 | 0.0299 |

## Test season 2022-23

### Full-distribution CRPS against rolling-average baselines (played rows, all three decision times)

| stat | n | mean pred | mean obs | CRPS model | CRPS poisson | CRPS normal | PIT<.1 | PIT>.9 | var ratio (obs/pred) |
|---|---|---|---|---|---|---|---|---|---|
| points | 82956 | 10.9634 | 10.9038 | 3.1828 | 3.5907 | 3.3648 | 0.0947 | 0.0939 | 0.9944 |
| rebounds | 82956 | 4.1767 | 4.1398 | 1.302 | 1.3731 | 1.3786 | 0.0874 | 0.0949 | 1.0182 |
| assists | 82956 | 2.4035 | 2.4045 | 0.8633 | 0.9033 | 0.9276 | 0.042 | 0.0855 | 0.9017 |
| threes | 82956 | 1.1941 | 1.1745 | 0.5576 | 0.5737 | 0.604 | 0.0301 | 0.0832 | 0.9976 |
| pra | 82956 | 17.5431 | 17.4481 | 4.1841 | 4.8236 | 4.4654 | 0.1016 | 0.0982 | 1.0075 |
| pr | 82956 | 15.1397 | 15.0436 | 3.8604 | 4.4068 | 4.0945 | 0.1007 | 0.0966 | 1.0025 |
| pa | 82956 | 13.3674 | 13.3083 | 3.493 | 3.975 | 3.7217 | 0.0962 | 0.0952 | 0.9935 |
| ra | 82956 | 6.5789 | 6.5443 | 1.7701 | 1.9075 | 1.8824 | 0.0893 | 0.0946 | 0.9922 |

### Calibration of P(X > line), raw vs calibrated (calibration window to test season)

| stat | logloss_raw | ece_raw | logloss_cal | ece_cal |
|---|---|---|---|---|
| points | 0.6625 | 0.0038 | 0.6632 | 0.0106 |
| rebounds | 0.5819 | 0.0085 | 0.5823 | 0.0063 |
| assists | 0.5378 | 0.0106 | 0.5392 | 0.0136 |
| threes | 0.4535 | 0.0071 | 0.4541 | 0.009 |
| pra | 0.6736 | 0.007 | 0.674 | 0.0108 |
| pr | 0.6716 | 0.006 | 0.672 | 0.008 |
| pa | 0.6668 | 0.0046 | 0.6677 | 0.0136 |
| ra | 0.6234 | 0.0078 | 0.6239 | 0.0132 |

## Test season 2023-24

### Full-distribution CRPS against rolling-average baselines (played rows, all three decision times)

| stat | n | mean pred | mean obs | CRPS model | CRPS poisson | CRPS normal | PIT<.1 | PIT>.9 | var ratio (obs/pred) |
|---|---|---|---|---|---|---|---|---|---|
| points | 84306 | 11.2098 | 10.6519 | 3.1146 | 3.4948 | 3.2799 | 0.107 | 0.0798 | 0.9516 |
| rebounds | 84306 | 4.1641 | 4.0699 | 1.2796 | 1.3448 | 1.3513 | 0.0858 | 0.0892 | 0.9797 |
| assists | 84306 | 2.541 | 2.4789 | 0.8882 | 0.9317 | 0.9522 | 0.0561 | 0.0866 | 0.9484 |
| threes | 84306 | 1.2495 | 1.1961 | 0.5564 | 0.572 | 0.6011 | 0.0308 | 0.0745 | 0.9346 |
| pra | 84306 | 17.9142 | 17.2008 | 4.1445 | 4.7543 | 4.4066 | 0.1148 | 0.0834 | 0.9743 |
| pr | 84306 | 15.3738 | 14.7219 | 3.8047 | 4.3176 | 4.0161 | 0.1128 | 0.0836 | 0.9749 |
| pa | 84306 | 13.7505 | 13.1308 | 3.442 | 3.8999 | 3.6564 | 0.1083 | 0.0798 | 0.9487 |
| ra | 84306 | 6.7045 | 6.5488 | 1.7729 | 1.906 | 1.8805 | 0.0985 | 0.0901 | 0.9871 |

### Calibration of P(X > line), raw vs calibrated (calibration window to test season)

| stat | logloss_raw | ece_raw | logloss_cal | ece_cal |
|---|---|---|---|---|
| points | 0.658 | 0.0347 | 0.6576 | 0.0273 |
| rebounds | 0.5801 | 0.0179 | 0.5802 | 0.0146 |
| assists | 0.5389 | 0.0109 | 0.5397 | 0.0144 |
| threes | 0.4483 | 0.0154 | 0.448 | 0.0105 |
| pra | 0.6693 | 0.038 | 0.6684 | 0.0292 |
| pr | 0.6662 | 0.0377 | 0.6653 | 0.0286 |
| pa | 0.6626 | 0.0352 | 0.6623 | 0.0279 |
| ra | 0.624 | 0.0227 | 0.6241 | 0.019 |

## Test season 2024-25

### Full-distribution CRPS against rolling-average baselines (played rows, all three decision times)

| stat | n | mean pred | mean obs | CRPS model | CRPS poisson | CRPS normal | PIT<.1 | PIT>.9 | var ratio (obs/pred) |
|---|---|---|---|---|---|---|---|---|---|
| points | 84399 | 10.7973 | 10.6404 | 3.1574 | 3.5505 | 3.3224 | 0.1047 | 0.1 | 1.0489 |
| rebounds | 84399 | 4.1935 | 4.1243 | 1.3001 | 1.3704 | 1.3729 | 0.0865 | 0.0937 | 1.0051 |
| assists | 84399 | 2.5594 | 2.4659 | 0.889 | 0.9307 | 0.9504 | 0.0623 | 0.0848 | 0.9664 |
| threes | 84399 | 1.311 | 1.2624 | 0.5834 | 0.6006 | 0.6249 | 0.0426 | 0.0832 | 1.0119 |
| pra | 84399 | 17.5498 | 17.2306 | 4.1605 | 4.781 | 4.4279 | 0.117 | 0.0962 | 1.026 |
| pr | 84399 | 14.9907 | 14.7647 | 3.8247 | 4.349 | 4.0424 | 0.113 | 0.0973 | 1.0303 |
| pa | 84399 | 13.3563 | 13.1064 | 3.477 | 3.9471 | 3.6905 | 0.11 | 0.0972 | 1.0338 |
| ra | 84399 | 6.7522 | 6.5902 | 1.7909 | 1.9287 | 1.9006 | 0.0975 | 0.0941 | 1.0024 |

### Calibration of P(X > line), raw vs calibrated (calibration window to test season)

| stat | logloss_raw | ece_raw | logloss_cal | ece_cal |
|---|---|---|---|---|
| points | 0.6628 | 0.0107 | 0.6646 | 0.0276 |
| rebounds | 0.583 | 0.0124 | 0.5832 | 0.0085 |
| assists | 0.5366 | 0.0169 | 0.537 | 0.0054 |
| threes | 0.4565 | 0.0143 | 0.4566 | 0.0087 |
| pra | 0.6736 | 0.016 | 0.675 | 0.0257 |
| pr | 0.6704 | 0.0138 | 0.6717 | 0.0266 |
| pa | 0.6673 | 0.0136 | 0.6687 | 0.0257 |
| ra | 0.6259 | 0.0227 | 0.6251 | 0.0067 |
