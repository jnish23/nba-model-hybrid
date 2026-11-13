# Phase 3 training report (a11_pg1)

Version `a11_pg1`, feature set `base`, excluded groups [], folds ['2021-22', '2022-23', '2023-24', '2024-25'], 4000 simulations per player-game. The stored pmfs are dispersion-calibrated (Stage F0); the isotonic layer is reported (calibrated columns) but not applied. Smoke check only: no confidence intervals, no ablations, no market comparison. Phase 4 is the formal evaluation.

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
| points | 83757 | 10.5195 | 10.4435 | 3.1792 | 3.5418 | 3.3101 | 0.0847 | 0.0921 | 0.9278 |
| rebounds | 83757 | 4.0812 | 4.1914 | 1.3179 | 1.3728 | 1.3752 | 0.0649 | 0.0978 | 0.9454 |
| assists | 83757 | 2.3084 | 2.321 | 0.8692 | 0.8982 | 0.9229 | 0.0439 | 0.0972 | 0.9578 |
| threes | 83757 | 1.2522 | 1.1758 | 0.5633 | 0.573 | 0.6028 | 0.0346 | 0.0784 | 0.9359 |
| pra | 83757 | 16.9082 | 16.9558 | 4.2152 | 4.7663 | 4.4105 | 0.0754 | 0.0939 | 0.9088 |
| pr | 83757 | 14.6005 | 14.6348 | 3.8817 | 4.3589 | 4.0469 | 0.0775 | 0.0933 | 0.9146 |
| pa | 83757 | 12.8274 | 12.7645 | 3.4978 | 3.9121 | 3.6571 | 0.0791 | 0.0924 | 0.9187 |
| ra | 83757 | 6.3884 | 6.5124 | 1.7943 | 1.904 | 1.8769 | 0.0689 | 0.0972 | 0.9325 |

### Calibration of P(X > line), raw vs calibrated (calibration window to test season)

| stat | logloss_raw | ece_raw | logloss_cal | ece_cal |
|---|---|---|---|---|
| points | 0.667 | 0.0059 | 0.671 | 0.0168 |
| rebounds | 0.5871 | 0.018 | 0.5895 | 0.0148 |
| assists | 0.5404 | 0.0065 | 0.5465 | 0.0212 |
| threes | 0.4556 | 0.0164 | 0.459 | 0.0124 |
| pra | 0.6782 | 0.0096 | 0.6819 | 0.0219 |
| pr | 0.6752 | 0.0096 | 0.6791 | 0.0215 |
| pa | 0.6718 | 0.0061 | 0.6745 | 0.0193 |
| ra | 0.629 | 0.0158 | 0.6321 | 0.0221 |

## Test season 2022-23

### Full-distribution CRPS against rolling-average baselines (played rows, all three decision times)

| stat | n | mean pred | mean obs | CRPS model | CRPS poisson | CRPS normal | PIT<.1 | PIT>.9 | var ratio (obs/pred) |
|---|---|---|---|---|---|---|---|---|---|
| points | 82956 | 11.0227 | 10.9038 | 3.1984 | 3.5907 | 3.3648 | 0.0958 | 0.0921 | 0.9929 |
| rebounds | 82956 | 4.1919 | 4.1398 | 1.3071 | 1.3731 | 1.3786 | 0.0883 | 0.0943 | 1.0122 |
| assists | 82956 | 2.4207 | 2.4045 | 0.8674 | 0.9033 | 0.9276 | 0.0429 | 0.0822 | 0.8925 |
| threes | 82956 | 1.2016 | 1.1745 | 0.5589 | 0.5737 | 0.604 | 0.0315 | 0.0826 | 0.9966 |
| pra | 82956 | 17.6352 | 17.4481 | 4.2139 | 4.8236 | 4.4654 | 0.1016 | 0.0956 | 1.0007 |
| pr | 82956 | 15.2148 | 15.0436 | 3.8839 | 4.4068 | 4.0945 | 0.1018 | 0.0947 | 0.9989 |
| pa | 82956 | 13.4437 | 13.3083 | 3.5146 | 3.975 | 3.7217 | 0.0951 | 0.092 | 0.98 |
| ra | 82956 | 6.6117 | 6.5443 | 1.7811 | 1.9075 | 1.8824 | 0.0906 | 0.0928 | 0.9818 |

### Calibration of P(X > line), raw vs calibrated (calibration window to test season)

| stat | logloss_raw | ece_raw | logloss_cal | ece_cal |
|---|---|---|---|---|
| points | 0.6619 | 0.0062 | 0.6628 | 0.0122 |
| rebounds | 0.5819 | 0.0103 | 0.5828 | 0.006 |
| assists | 0.538 | 0.0104 | 0.5392 | 0.0156 |
| threes | 0.454 | 0.0088 | 0.4545 | 0.0083 |
| pra | 0.6734 | 0.0114 | 0.6739 | 0.0114 |
| pr | 0.6711 | 0.0097 | 0.6714 | 0.0091 |
| pa | 0.6664 | 0.0083 | 0.6672 | 0.0147 |
| ra | 0.6245 | 0.0115 | 0.6252 | 0.0146 |

## Test season 2023-24

### Full-distribution CRPS against rolling-average baselines (played rows, all three decision times)

| stat | n | mean pred | mean obs | CRPS model | CRPS poisson | CRPS normal | PIT<.1 | PIT>.9 | var ratio (obs/pred) |
|---|---|---|---|---|---|---|---|---|---|
| points | 84306 | 11.139 | 10.6519 | 3.122 | 3.4948 | 3.2799 | 0.1053 | 0.0823 | 0.9565 |
| rebounds | 84306 | 4.147 | 4.0699 | 1.2838 | 1.3448 | 1.3513 | 0.0835 | 0.0901 | 0.9799 |
| assists | 84306 | 2.5241 | 2.4789 | 0.8901 | 0.9317 | 0.9522 | 0.0535 | 0.0893 | 0.9547 |
| threes | 84306 | 1.2386 | 1.1961 | 0.5573 | 0.572 | 0.6011 | 0.0303 | 0.0771 | 0.9478 |
| pra | 84306 | 17.8095 | 17.2008 | 4.1609 | 4.7543 | 4.4066 | 0.1131 | 0.086 | 0.9775 |
| pr | 84306 | 15.2859 | 14.7219 | 3.8175 | 4.3176 | 4.0161 | 0.1118 | 0.0865 | 0.979 |
| pa | 84306 | 13.6628 | 13.1308 | 3.4526 | 3.8999 | 3.6564 | 0.105 | 0.0831 | 0.952 |
| ra | 84306 | 6.6703 | 6.5488 | 1.7799 | 1.906 | 1.8805 | 0.0964 | 0.0918 | 0.9906 |

### Calibration of P(X > line), raw vs calibrated (calibration window to test season)

| stat | logloss_raw | ece_raw | logloss_cal | ece_cal |
|---|---|---|---|---|
| points | 0.6584 | 0.0304 | 0.658 | 0.0197 |
| rebounds | 0.5805 | 0.0152 | 0.5807 | 0.0113 |
| assists | 0.5401 | 0.0087 | 0.5408 | 0.0097 |
| threes | 0.4493 | 0.0132 | 0.4497 | 0.0076 |
| pra | 0.6697 | 0.0334 | 0.6688 | 0.0211 |
| pr | 0.6663 | 0.0332 | 0.6655 | 0.0213 |
| pa | 0.6628 | 0.0304 | 0.6623 | 0.0198 |
| ra | 0.6245 | 0.0191 | 0.6245 | 0.0132 |

## Test season 2024-25

### Full-distribution CRPS against rolling-average baselines (played rows, all three decision times)

| stat | n | mean pred | mean obs | CRPS model | CRPS poisson | CRPS normal | PIT<.1 | PIT>.9 | var ratio (obs/pred) |
|---|---|---|---|---|---|---|---|---|---|
| points | 84399 | 10.6897 | 10.6404 | 3.1687 | 3.5505 | 3.3224 | 0.1005 | 0.1027 | 1.0588 |
| rebounds | 84399 | 4.1637 | 4.1243 | 1.3037 | 1.3704 | 1.3729 | 0.0832 | 0.0961 | 1.0099 |
| assists | 84399 | 2.5321 | 2.4659 | 0.8911 | 0.9307 | 0.9504 | 0.0606 | 0.0875 | 0.9748 |
| threes | 84399 | 1.2985 | 1.2624 | 0.5837 | 0.6006 | 0.6249 | 0.0404 | 0.0846 | 1.0134 |
| pra | 84399 | 17.3849 | 17.2306 | 4.1817 | 4.781 | 4.4279 | 0.1134 | 0.0997 | 1.04 |
| pr | 84399 | 14.8533 | 14.7647 | 3.842 | 4.349 | 4.0424 | 0.1092 | 0.1007 | 1.0428 |
| pa | 84399 | 13.2216 | 13.1064 | 3.4924 | 3.9471 | 3.6905 | 0.1064 | 0.1007 | 1.0487 |
| ra | 84399 | 6.6951 | 6.5902 | 1.7974 | 1.9287 | 1.9006 | 0.0941 | 0.0969 | 1.0102 |

### Calibration of P(X > line), raw vs calibrated (calibration window to test season)

| stat | logloss_raw | ece_raw | logloss_cal | ece_cal |
|---|---|---|---|---|
| points | 0.6639 | 0.005 | 0.6658 | 0.0285 |
| rebounds | 0.5836 | 0.0088 | 0.5842 | 0.007 |
| assists | 0.5387 | 0.012 | 0.5391 | 0.0053 |
| threes | 0.4572 | 0.0109 | 0.4579 | 0.0085 |
| pra | 0.6738 | 0.0088 | 0.6754 | 0.0262 |
| pr | 0.6714 | 0.0073 | 0.6732 | 0.027 |
| pa | 0.6682 | 0.0063 | 0.6702 | 0.027 |
| ra | 0.6267 | 0.0168 | 0.6266 | 0.0046 |
