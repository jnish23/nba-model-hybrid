# Phase 3 training report (v2_val_2024-25)

Version `v2_val_2024-25`, feature set `base`, excluded groups [], folds ['2024-25'], 4000 simulations per player-game. The stored pmfs are dispersion-calibrated (Stage F0); the isotonic layer is reported (calibrated columns) but not applied. Smoke check only: no confidence intervals, no ablations, no market comparison. Phase 4 is the formal evaluation.

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

## Test season 2024-25

### Full-distribution CRPS against rolling-average baselines (played rows, all three decision times)

| stat | n | mean pred | mean obs | CRPS model | CRPS poisson | CRPS normal | PIT<.1 | PIT>.9 | var ratio (obs/pred) |
|---|---|---|---|---|---|---|---|---|---|
| points | 84399 | 11.1113 | 10.6404 | 3.1677 | 3.5505 | 3.3224 | 0.113 | 0.0888 | 1.0152 |
| rebounds | 84399 | 4.2498 | 4.1243 | 1.3011 | 1.3704 | 1.3729 | 0.0884 | 0.0891 | 0.9843 |
| assists | 84399 | 2.596 | 2.4659 | 0.8923 | 0.9307 | 0.9504 | 0.0655 | 0.0827 | 0.9546 |
| threes | 84399 | 1.3419 | 1.2624 | 0.5851 | 0.6006 | 0.6249 | 0.0445 | 0.0788 | 0.985 |
| pra | 84399 | 17.9559 | 17.2306 | 4.1765 | 4.781 | 4.4279 | 0.1211 | 0.085 | 0.9929 |
| pr | 84399 | 15.3608 | 14.7647 | 3.8361 | 4.349 | 4.0424 | 0.1183 | 0.0868 | 0.9948 |
| pa | 84399 | 13.7068 | 13.1064 | 3.4921 | 3.9471 | 3.6905 | 0.1177 | 0.0859 | 1.0053 |
| ra | 84399 | 6.8444 | 6.5902 | 1.7948 | 1.9287 | 1.9006 | 0.0996 | 0.0875 | 0.9802 |

### Calibration of P(X > line), raw vs calibrated (calibration window to test season)

| stat | logloss_raw | ece_raw | logloss_cal | ece_cal |
|---|---|---|---|---|
| points | 0.66 | 0.0315 | 0.6581 | 0.0027 |
| rebounds | 0.5828 | 0.0205 | 0.5822 | 0.0059 |
| assists | 0.5363 | 0.0233 | 0.5361 | 0.0101 |
| threes | 0.4562 | 0.0214 | 0.4552 | 0.0061 |
| pra | 0.6712 | 0.037 | 0.6685 | 0.0047 |
| pr | 0.668 | 0.0345 | 0.6657 | 0.004 |
| pa | 0.665 | 0.0349 | 0.6627 | 0.0038 |
| ra | 0.6249 | 0.0326 | 0.6231 | 0.012 |
