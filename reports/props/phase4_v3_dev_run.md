# Phase 3 training report (v3_dev)

Version `v3_dev`, feature set `base`, excluded groups [], folds ['2021-22', '2022-23', '2023-24'], 4000 simulations per player-game. The stored pmfs are dispersion-calibrated (Stage F0); the isotonic layer is reported (calibrated columns) but not applied. Smoke check only: no confidence intervals, no ablations, no market comparison. Phase 4 is the formal evaluation.

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
| points | 83757 | 10.4206 | 10.4435 | 3.1647 | 3.5418 | 3.3101 | 0.0805 | 0.0942 | 0.9316 |
| rebounds | 83757 | 4.0584 | 4.1914 | 1.3138 | 1.3728 | 1.3752 | 0.0643 | 0.1006 | 0.9532 |
| assists | 83757 | 2.2808 | 2.321 | 0.865 | 0.8982 | 0.9229 | 0.0424 | 0.0992 | 0.9717 |
| threes | 83757 | 1.2383 | 1.1758 | 0.5618 | 0.573 | 0.6028 | 0.0333 | 0.0795 | 0.9405 |
| pra | 83757 | 16.7589 | 16.9558 | 4.1878 | 4.7663 | 4.4105 | 0.0721 | 0.0977 | 0.9114 |
| pr | 83757 | 14.4786 | 14.6348 | 3.8609 | 4.3589 | 4.0469 | 0.0741 | 0.0965 | 0.9171 |
| pa | 83757 | 12.701 | 12.7645 | 3.4766 | 3.9121 | 3.6571 | 0.0754 | 0.0952 | 0.9217 |
| ra | 83757 | 6.3381 | 6.5124 | 1.7851 | 1.904 | 1.8769 | 0.0677 | 0.1002 | 0.9418 |

### Calibration of P(X > line), raw vs calibrated (calibration window to test season)

| stat | logloss_raw | ece_raw | logloss_cal | ece_cal |
|---|---|---|---|---|
| points | 0.6678 | 0.0125 | 0.6698 | 0.0209 |
| rebounds | 0.5869 | 0.0214 | 0.5904 | 0.0196 |
| assists | 0.5402 | 0.0109 | 0.5499 | 0.0283 |
| threes | 0.4556 | 0.0129 | 0.4593 | 0.0104 |
| pra | 0.6792 | 0.0177 | 0.6839 | 0.0287 |
| pr | 0.6758 | 0.0165 | 0.68 | 0.0266 |
| pa | 0.6725 | 0.0141 | 0.6754 | 0.0247 |
| ra | 0.6288 | 0.0218 | 0.6319 | 0.0299 |

## Test season 2022-23

### Full-distribution CRPS against rolling-average baselines (played rows, all three decision times)

| stat | n | mean pred | mean obs | CRPS model | CRPS poisson | CRPS normal | PIT<.1 | PIT>.9 | var ratio (obs/pred) |
|---|---|---|---|---|---|---|---|---|---|
| points | 82956 | 10.9576 | 10.9038 | 3.1827 | 3.5907 | 3.3648 | 0.0947 | 0.0944 | 0.9962 |
| rebounds | 82956 | 4.1766 | 4.1398 | 1.302 | 1.3731 | 1.3786 | 0.0877 | 0.095 | 1.0195 |
| assists | 82956 | 2.4035 | 2.4045 | 0.8633 | 0.9033 | 0.9276 | 0.042 | 0.0855 | 0.9018 |
| threes | 82956 | 1.1936 | 1.1745 | 0.5575 | 0.5737 | 0.604 | 0.0301 | 0.0831 | 0.9986 |
| pra | 82956 | 17.5373 | 17.4481 | 4.184 | 4.8236 | 4.4654 | 0.1017 | 0.0988 | 1.0102 |
| pr | 82956 | 15.134 | 15.0436 | 3.8603 | 4.4068 | 4.0945 | 0.101 | 0.0969 | 1.005 |
| pa | 82956 | 13.3615 | 13.3083 | 3.4929 | 3.975 | 3.7217 | 0.0961 | 0.0956 | 0.9946 |
| ra | 82956 | 6.579 | 6.5443 | 1.7701 | 1.9075 | 1.8824 | 0.0896 | 0.0948 | 0.9934 |

### Calibration of P(X > line), raw vs calibrated (calibration window to test season)

| stat | logloss_raw | ece_raw | logloss_cal | ece_cal |
|---|---|---|---|---|
| points | 0.6625 | 0.0046 | 0.6632 | 0.0102 |
| rebounds | 0.5819 | 0.0085 | 0.5822 | 0.0062 |
| assists | 0.5378 | 0.0106 | 0.5391 | 0.0134 |
| threes | 0.4534 | 0.007 | 0.4541 | 0.0085 |
| pra | 0.6736 | 0.0067 | 0.6741 | 0.0102 |
| pr | 0.6714 | 0.0057 | 0.6721 | 0.0081 |
| pa | 0.667 | 0.0045 | 0.6677 | 0.0137 |
| ra | 0.6234 | 0.0078 | 0.6239 | 0.013 |

## Test season 2023-24

### Full-distribution CRPS against rolling-average baselines (played rows, all three decision times)

| stat | n | mean pred | mean obs | CRPS model | CRPS poisson | CRPS normal | PIT<.1 | PIT>.9 | var ratio (obs/pred) |
|---|---|---|---|---|---|---|---|---|---|
| points | 84306 | 11.2075 | 10.6519 | 3.1141 | 3.4948 | 3.2799 | 0.107 | 0.0798 | 0.9509 |
| rebounds | 84306 | 4.1638 | 4.0699 | 1.2795 | 1.3448 | 1.3513 | 0.0857 | 0.0889 | 0.9782 |
| assists | 84306 | 2.5407 | 2.4789 | 0.8882 | 0.9317 | 0.9522 | 0.0559 | 0.0865 | 0.9471 |
| threes | 84306 | 1.249 | 1.1961 | 0.5563 | 0.572 | 0.6011 | 0.0311 | 0.0744 | 0.9354 |
| pra | 84306 | 17.9114 | 17.2008 | 4.144 | 4.7543 | 4.4066 | 0.1145 | 0.0832 | 0.9727 |
| pr | 84306 | 15.3712 | 14.7219 | 3.8042 | 4.3176 | 4.0161 | 0.1129 | 0.0835 | 0.9736 |
| pa | 84306 | 13.748 | 13.1308 | 3.4415 | 3.8999 | 3.6564 | 0.108 | 0.0796 | 0.9476 |
| ra | 84306 | 6.7037 | 6.5488 | 1.7727 | 1.906 | 1.8805 | 0.0986 | 0.0901 | 0.986 |

### Calibration of P(X > line), raw vs calibrated (calibration window to test season)

| stat | logloss_raw | ece_raw | logloss_cal | ece_cal |
|---|---|---|---|---|
| points | 0.6581 | 0.0343 | 0.6576 | 0.0265 |
| rebounds | 0.5803 | 0.0177 | 0.5804 | 0.0143 |
| assists | 0.539 | 0.0109 | 0.5397 | 0.0142 |
| threes | 0.4481 | 0.0153 | 0.448 | 0.0105 |
| pra | 0.6693 | 0.0377 | 0.6683 | 0.0283 |
| pr | 0.6662 | 0.0374 | 0.6652 | 0.028 |
| pa | 0.6627 | 0.0347 | 0.6622 | 0.0272 |
| ra | 0.624 | 0.0225 | 0.6245 | 0.0187 |
