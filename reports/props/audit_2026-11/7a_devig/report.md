# 7a: de-vig method and consensus set

Step S2 of `WP6_WP7_experiment_plan.md`, produced by `scripts/a11_devig_calibration.py` on 2026-11-12 21:07. Seasons: 2023-24, 2024-25. Quotes: `data/props_experiments/audit_2026-11/quotes`. Outcomes: `player_game_labels` of `data/props_features_v3.db`. Date-clustered bootstrap: 2000 replicates, seed 20261112.

## Decision

**Chosen p_market method: `shin_all_mean` (de-vig shin, consensus set all_mean).** shin_all_mean passes the 7a keep rule in 2023-24 and 2024-25; it has the lowest pooled log loss (0.68441) of 1 passing variant(s).

`choice.json` carries devig_method, consensus_set, passed and reason for the later steps.

## Keep rule, variant by variant

Rule (plan 7a item 7, judged on the four bet stats by amendment 5 of 2026-11-12): in each of 2023-24 and 2024-25 the paired log-loss difference (variant minus current; negative is better) has a 95% upper bound below 0, |mean bias| is no larger than the current method's, and coverage is at least 95% of the current method's. Bonferroni intervals use 0.05/23 = 0.0022 (99.78%, plan A.5); they are printed for reference and the keep rule uses the 95% bound.

| variant | season | d logloss | 95% interval | Bonferroni interval | abs bias pts (current) | coverage | logloss ok | bias ok | coverage ok |
|---|---|---|---|---|---|---|---|---|---|
| multiplicative_bettable_mean | 2023-24 | +0.00011 | [+0.00006, +0.00017] | [+0.00001, +0.00020] | 1.631 (1.638) | 83.3% | no | yes | no |
| multiplicative_all_median | 2023-24 | +0.00010 | [+0.00004, +0.00015] | [-0.00000, +0.00018] | 1.648 (1.638) | 100.0% | no | no | yes |
| power_all_mean | 2023-24 | -0.00018 | [-0.00029, -0.00007] | [-0.00038, -0.00002] | 1.588 (1.638) | 100.0% | yes | yes | yes |
| power_bettable_mean | 2023-24 | -0.00003 | [-0.00017, +0.00009] | [-0.00026, +0.00016] | 1.581 (1.638) | 83.3% | no | yes | no |
| power_all_median | 2023-24 | -0.00006 | [-0.00020, +0.00007] | [-0.00030, +0.00014] | 1.600 (1.638) | 100.0% | no | yes | yes |
| shin_all_mean | 2023-24 | -0.00014 | [-0.00021, -0.00007] | [-0.00026, -0.00004] | 1.606 (1.638) | 100.0% | yes | yes | yes |
| shin_bettable_mean | 2023-24 | -0.00000 | [-0.00011, +0.00009] | [-0.00016, +0.00013] | 1.598 (1.638) | 83.3% | no | yes | no |
| shin_all_median | 2023-24 | -0.00003 | [-0.00013, +0.00007] | [-0.00019, +0.00012] | 1.617 (1.638) | 100.0% | no | yes | yes |
| multiplicative_bettable_mean | 2024-25 | +0.00006 | [+0.00000, +0.00012] | [-0.00003, +0.00015] | 1.497 (1.486) | 88.2% | no | no | no |
| multiplicative_all_median | 2024-25 | +0.00003 | [-0.00003, +0.00008] | [-0.00005, +0.00011] | 1.501 (1.486) | 100.0% | no | no | yes |
| power_all_mean | 2024-25 | -0.00008 | [-0.00019, +0.00002] | [-0.00025, +0.00008] | 1.409 (1.486) | 100.0% | no | yes | yes |
| power_bettable_mean | 2024-25 | -0.00003 | [-0.00014, +0.00008] | [-0.00020, +0.00015] | 1.420 (1.486) | 88.2% | no | yes | no |
| power_all_median | 2024-25 | -0.00005 | [-0.00017, +0.00007] | [-0.00024, +0.00013] | 1.424 (1.486) | 100.0% | no | yes | yes |
| shin_all_mean | 2024-25 | -0.00007 | [-0.00014, -0.00000] | [-0.00018, +0.00003] | 1.436 (1.486) | 100.0% | yes | yes | yes |
| shin_bettable_mean | 2024-25 | -0.00001 | [-0.00009, +0.00007] | [-0.00013, +0.00011] | 1.448 (1.486) | 88.2% | no | yes | no |
| shin_all_median | 2024-25 | -0.00004 | [-0.00013, +0.00005] | [-0.00018, +0.00010] | 1.451 (1.486) | 100.0% | no | yes | yes |

| variant | keep rule passes | pooled log loss | reason if not |
|---|---|---|---|
| multiplicative_bettable_mean | no | 0.68460 (current 0.68452) | 2023-24: log-loss upper bound +0.00017 not below 0; 2023-24: coverage 83.3% of current, below 95%; 2024-25: log-loss upper bound +0.00012 not below 0; 2024-25: |bias| 1.497 pts above the current method's 1.486; 2024-25: coverage 88.2% of current, below 95% |
| multiplicative_all_median | no | 0.68458 (current 0.68452) | 2023-24: log-loss upper bound +0.00015 not below 0; 2023-24: |bias| 1.648 pts above the current method's 1.638; 2024-25: log-loss upper bound +0.00008 not below 0; 2024-25: |bias| 1.501 pts above the current method's 1.486 |
| power_all_mean | no | 0.68438 (current 0.68452) | 2024-25: log-loss upper bound +0.00002 not below 0 |
| power_bettable_mean | no | 0.68448 (current 0.68452) | 2023-24: log-loss upper bound +0.00009 not below 0; 2023-24: coverage 83.3% of current, below 95%; 2024-25: log-loss upper bound +0.00008 not below 0; 2024-25: coverage 88.2% of current, below 95% |
| power_all_median | no | 0.68446 (current 0.68452) | 2023-24: log-loss upper bound +0.00007 not below 0; 2024-25: log-loss upper bound +0.00007 not below 0 |
| shin_all_mean | yes | 0.68441 (current 0.68452) |  |
| shin_bettable_mean | no | 0.68451 (current 0.68452) | 2023-24: log-loss upper bound +0.00009 not below 0; 2023-24: coverage 83.3% of current, below 95%; 2024-25: log-loss upper bound +0.00007 not below 0; 2024-25: coverage 88.2% of current, below 95% |
| shin_all_median | no | 0.68448 (current 0.68452) | 2023-24: log-loss upper bound +0.00007 not below 0; 2024-25: log-loss upper bound +0.00005 not below 0 |

Secondary, decides nothing: the same rule on all eight markets (the decision above uses the four bet stats, rebounds, assists, threes, ra) would choose `shin_all_mean`.

## Rows

Rows where all 9 variants price the same line, joined to outcomes of players who played, pushes dropped.

| season | common_rows | no_label | did_not_play | missing_stat | push | used | rows_current_method |
|---|---|---|---|---|---|---|---|
| 2023-24 | 273588 | 1761 | 1726 | 0 | 0 | 270101 | 338227 |
| 2024-25 | 253305 | 2115 | 2337 | 0 | 0 | 248853 | 308754 |

## Metrics of the 9 variants (common rows, pooled over stats and slots)

bias = mean(hit - p_market); recalibration = logistic fit of hit on logit(p) (ideal intercept 0, slope 1).

| season | variant | n | logloss | brier | bias | mean_p | hit_rate | recal_intercept | recal_slope |
|---|---|---|---|---|---|---|---|---|---|
| 2023-24 | multiplicative_all_mean | 270101 | 0.68779 | 0.24735 | -0.01832 | 0.49777 | 0.47946 | -0.07252 | 1.19163 |
| 2023-24 | multiplicative_bettable_mean | 270101 | 0.68788 | 0.24739 | -0.01834 | 0.49780 | 0.47946 | -0.07284 | 1.16649 |
| 2023-24 | multiplicative_all_median | 270101 | 0.68790 | 0.24740 | -0.01832 | 0.49778 | 0.47946 | -0.07271 | 1.17020 |
| 2023-24 | power_all_mean | 270101 | 0.68767 | 0.24729 | -0.01807 | 0.49753 | 0.47946 | -0.07245 | 1.07202 |
| 2023-24 | power_bettable_mean | 270101 | 0.68777 | 0.24734 | -0.01810 | 0.49756 | 0.47946 | -0.07276 | 1.04970 |
| 2023-24 | power_all_median | 270101 | 0.68779 | 0.24735 | -0.01808 | 0.49754 | 0.47946 | -0.07263 | 1.05468 |
| 2023-24 | shin_all_mean | 270101 | 0.68770 | 0.24731 | -0.01816 | 0.49761 | 0.47946 | -0.07246 | 1.11219 |
| 2023-24 | shin_bettable_mean | 270101 | 0.68780 | 0.24735 | -0.01818 | 0.49764 | 0.47946 | -0.07275 | 1.08898 |
| 2023-24 | shin_all_median | 270101 | 0.68782 | 0.24736 | -0.01816 | 0.49762 | 0.47946 | -0.07264 | 1.09341 |
| 2024-25 | multiplicative_all_mean | 248853 | 0.68888 | 0.24789 | -0.01028 | 0.49596 | 0.48568 | -0.03990 | 1.09958 |
| 2024-25 | multiplicative_bettable_mean | 248853 | 0.68893 | 0.24791 | -0.01045 | 0.49612 | 0.48568 | -0.04059 | 1.10354 |
| 2024-25 | multiplicative_all_median | 248853 | 0.68891 | 0.24790 | -0.01034 | 0.49602 | 0.48568 | -0.04032 | 1.09069 |
| 2024-25 | power_all_mean | 248853 | 0.68883 | 0.24787 | -0.00986 | 0.49554 | 0.48568 | -0.03991 | 0.99277 |
| 2024-25 | power_bettable_mean | 248853 | 0.68888 | 0.24789 | -0.01004 | 0.49572 | 0.48568 | -0.04060 | 0.99490 |
| 2024-25 | power_all_median | 248853 | 0.68886 | 0.24788 | -0.00992 | 0.49560 | 0.48568 | -0.04030 | 0.98450 |
| 2024-25 | shin_all_mean | 248853 | 0.68884 | 0.24787 | -0.01001 | 0.49568 | 0.48568 | -0.03990 | 1.02817 |
| 2024-25 | shin_bettable_mean | 248853 | 0.68889 | 0.24790 | -0.01018 | 0.49586 | 0.48568 | -0.04059 | 1.03094 |
| 2024-25 | shin_all_median | 248853 | 0.68887 | 0.24788 | -0.01007 | 0.49575 | 0.48568 | -0.04030 | 1.01968 |
| both | multiplicative_all_mean | 518954 | 0.68832 | 0.24761 | -0.01446 | 0.49690 | 0.48244 | -0.05668 | 1.14716 |
| both | multiplicative_bettable_mean | 518954 | 0.68838 | 0.24764 | -0.01456 | 0.49700 | 0.48244 | -0.05724 | 1.13615 |
| both | multiplicative_all_median | 518954 | 0.68838 | 0.24764 | -0.01450 | 0.49694 | 0.48244 | -0.05701 | 1.13161 |
| both | power_all_mean | 518954 | 0.68823 | 0.24757 | -0.01414 | 0.49657 | 0.48244 | -0.05666 | 1.03377 |
| both | power_bettable_mean | 518954 | 0.68831 | 0.24760 | -0.01424 | 0.49668 | 0.48244 | -0.05721 | 1.02327 |
| both | power_all_median | 518954 | 0.68830 | 0.24760 | -0.01417 | 0.49661 | 0.48244 | -0.05697 | 1.02062 |
| both | shin_all_mean | 518954 | 0.68825 | 0.24758 | -0.01425 | 0.49669 | 0.48244 | -0.05666 | 1.07163 |
| both | shin_bettable_mean | 518954 | 0.68832 | 0.24761 | -0.01435 | 0.49679 | 0.48244 | -0.05720 | 1.06102 |
| both | shin_all_median | 518954 | 0.68832 | 0.24761 | -0.01428 | 0.49672 | 0.48244 | -0.05697 | 1.05764 |

## Paired differences against the current method

Negative is better for log loss and Brier. Date-clustered bootstrap; 95% and Bonferroni intervals.

| season | variant | metric | diff | lo95 | hi95 | lo_bonf | hi_bonf | n_dates |
|---|---|---|---|---|---|---|---|---|
| 2023-24 | multiplicative_bettable_mean | logloss | 0.00008 | 0.00004 | 0.00013 | 0.00001 | 0.00015 | 207 |
| 2023-24 | multiplicative_bettable_mean | brier | 0.00004 | 0.00002 | 0.00006 | 0.00001 | 0.00007 | 207 |
| 2023-24 | multiplicative_all_median | logloss | 0.00010 | 0.00006 | 0.00015 | 0.00003 | 0.00018 | 207 |
| 2023-24 | multiplicative_all_median | brier | 0.00005 | 0.00003 | 0.00008 | 0.00002 | 0.00009 | 207 |
| 2023-24 | power_all_mean | logloss | -0.00012 | -0.00018 | -0.00006 | -0.00023 | -0.00004 | 207 |
| 2023-24 | power_all_mean | brier | -0.00006 | -0.00009 | -0.00003 | -0.00011 | -0.00002 | 207 |
| 2023-24 | power_bettable_mean | logloss | -0.00002 | -0.00009 | 0.00005 | -0.00013 | 0.00010 | 207 |
| 2023-24 | power_bettable_mean | brier | -0.00001 | -0.00004 | 0.00003 | -0.00006 | 0.00005 | 207 |
| 2023-24 | power_all_median | logloss | -0.00000 | -0.00008 | 0.00008 | -0.00013 | 0.00012 | 207 |
| 2023-24 | power_all_median | brier | 0.00000 | -0.00004 | 0.00004 | -0.00006 | 0.00006 | 207 |
| 2023-24 | shin_all_mean | logloss | -0.00009 | -0.00013 | -0.00005 | -0.00016 | -0.00004 | 207 |
| 2023-24 | shin_all_mean | brier | -0.00004 | -0.00006 | -0.00003 | -0.00008 | -0.00002 | 207 |
| 2023-24 | shin_bettable_mean | logloss | 0.00001 | -0.00005 | 0.00006 | -0.00009 | 0.00009 | 207 |
| 2023-24 | shin_bettable_mean | brier | 0.00000 | -0.00002 | 0.00003 | -0.00004 | 0.00004 | 207 |
| 2023-24 | shin_all_median | logloss | 0.00002 | -0.00004 | 0.00009 | -0.00008 | 0.00012 | 207 |
| 2023-24 | shin_all_median | brier | 0.00001 | -0.00002 | 0.00004 | -0.00004 | 0.00006 | 207 |
| 2024-25 | multiplicative_bettable_mean | logloss | 0.00005 | 0.00001 | 0.00009 | -0.00001 | 0.00011 | 213 |
| 2024-25 | multiplicative_bettable_mean | brier | 0.00003 | 0.00001 | 0.00004 | -0.00001 | 0.00005 | 213 |
| 2024-25 | multiplicative_all_median | logloss | 0.00002 | -0.00001 | 0.00006 | -0.00003 | 0.00008 | 213 |
| 2024-25 | multiplicative_all_median | brier | 0.00001 | -0.00001 | 0.00003 | -0.00002 | 0.00004 | 213 |
| 2024-25 | power_all_mean | logloss | -0.00005 | -0.00010 | 0.00001 | -0.00014 | 0.00004 | 213 |
| 2024-25 | power_all_mean | brier | -0.00002 | -0.00005 | 0.00000 | -0.00006 | 0.00002 | 213 |
| 2024-25 | power_bettable_mean | logloss | 0.00000 | -0.00006 | 0.00006 | -0.00009 | 0.00009 | 213 |
| 2024-25 | power_bettable_mean | brier | 0.00000 | -0.00003 | 0.00003 | -0.00004 | 0.00005 | 213 |
| 2024-25 | power_all_median | logloss | -0.00002 | -0.00009 | 0.00005 | -0.00014 | 0.00009 | 213 |
| 2024-25 | power_all_median | brier | -0.00001 | -0.00004 | 0.00003 | -0.00006 | 0.00005 | 213 |
| 2024-25 | shin_all_mean | logloss | -0.00004 | -0.00008 | -0.00000 | -0.00010 | 0.00002 | 213 |
| 2024-25 | shin_all_mean | brier | -0.00002 | -0.00003 | -0.00000 | -0.00005 | 0.00001 | 213 |
| 2024-25 | shin_bettable_mean | logloss | 0.00001 | -0.00004 | 0.00005 | -0.00006 | 0.00008 | 213 |
| 2024-25 | shin_bettable_mean | brier | 0.00001 | -0.00002 | 0.00003 | -0.00003 | 0.00004 | 213 |
| 2024-25 | shin_all_median | logloss | -0.00001 | -0.00007 | 0.00004 | -0.00011 | 0.00007 | 213 |
| 2024-25 | shin_all_median | brier | -0.00000 | -0.00003 | 0.00002 | -0.00005 | 0.00004 | 213 |
| both | multiplicative_bettable_mean | logloss | 0.00007 | 0.00004 | 0.00010 | 0.00002 | 0.00011 | 420 |
| both | multiplicative_bettable_mean | brier | 0.00003 | 0.00002 | 0.00005 | 0.00001 | 0.00006 | 420 |
| both | multiplicative_all_median | logloss | 0.00007 | 0.00003 | 0.00010 | 0.00002 | 0.00011 | 420 |
| both | multiplicative_all_median | brier | 0.00003 | 0.00002 | 0.00005 | 0.00001 | 0.00006 | 420 |
| both | power_all_mean | logloss | -0.00009 | -0.00013 | -0.00005 | -0.00014 | -0.00003 | 420 |
| both | power_all_mean | brier | -0.00004 | -0.00006 | -0.00002 | -0.00007 | -0.00001 | 420 |
| both | power_bettable_mean | logloss | -0.00001 | -0.00006 | 0.00004 | -0.00008 | 0.00006 | 420 |
| both | power_bettable_mean | brier | -0.00000 | -0.00003 | 0.00002 | -0.00004 | 0.00003 | 420 |
| both | power_all_median | logloss | -0.00001 | -0.00006 | 0.00004 | -0.00009 | 0.00007 | 420 |
| both | power_all_median | brier | -0.00000 | -0.00003 | 0.00002 | -0.00004 | 0.00003 | 420 |
| both | shin_all_mean | logloss | -0.00007 | -0.00009 | -0.00004 | -0.00010 | -0.00003 | 420 |
| both | shin_all_mean | brier | -0.00003 | -0.00004 | -0.00002 | -0.00005 | -0.00001 | 420 |
| both | shin_bettable_mean | logloss | 0.00001 | -0.00003 | 0.00004 | -0.00005 | 0.00006 | 420 |
| both | shin_bettable_mean | brier | 0.00001 | -0.00001 | 0.00002 | -0.00002 | 0.00003 | 420 |
| both | shin_all_median | logloss | 0.00001 | -0.00003 | 0.00005 | -0.00005 | 0.00007 | 420 |
| both | shin_all_median | brier | 0.00000 | -0.00002 | 0.00003 | -0.00002 | 0.00004 | 420 |

## How much of the over-bias each variant removes

Bias in points (hit - p_market, times 100); removed = |current| - |variant|.

| season | variant | bias_pts | bias_current_pts | removed_pts | removed_share |
|---|---|---|---|---|---|
| 2023-24 | multiplicative_all_mean | -1.83167 | -1.83167 | 0.00000 | 0.00000 |
| 2023-24 | multiplicative_bettable_mean | -1.83443 | -1.83167 | -0.00276 | -0.00151 |
| 2023-24 | multiplicative_all_median | -1.83231 | -1.83167 | -0.00064 | -0.00035 |
| 2023-24 | power_all_mean | -1.80725 | -1.83167 | 0.02442 | 0.01333 |
| 2023-24 | power_bettable_mean | -1.81006 | -1.83167 | 0.02161 | 0.01180 |
| 2023-24 | power_all_median | -1.80820 | -1.83167 | 0.02348 | 0.01282 |
| 2023-24 | shin_all_mean | -1.81554 | -1.83167 | 0.01613 | 0.00881 |
| 2023-24 | shin_bettable_mean | -1.81807 | -1.83167 | 0.01360 | 0.00743 |
| 2023-24 | shin_all_median | -1.81649 | -1.83167 | 0.01518 | 0.00829 |
| 2024-25 | multiplicative_all_mean | -1.02788 | -1.02788 | 0.00000 | 0.00000 |
| 2024-25 | multiplicative_bettable_mean | -1.04485 | -1.02788 | -0.01698 | -0.01652 |
| 2024-25 | multiplicative_all_median | -1.03425 | -1.02788 | -0.00638 | -0.00620 |
| 2024-25 | power_all_mean | -0.98619 | -1.02788 | 0.04169 | 0.04056 |
| 2024-25 | power_bettable_mean | -1.00437 | -1.02788 | 0.02350 | 0.02287 |
| 2024-25 | power_all_median | -0.99238 | -1.02788 | 0.03550 | 0.03454 |
| 2024-25 | shin_all_mean | -1.00072 | -1.02788 | 0.02716 | 0.02642 |
| 2024-25 | shin_bettable_mean | -1.01847 | -1.02788 | 0.00941 | 0.00915 |
| 2024-25 | shin_all_median | -1.00708 | -1.02788 | 0.02079 | 0.02023 |
| both | multiplicative_all_mean | -1.44623 | -1.44623 | 0.00000 | 0.00000 |
| both | multiplicative_bettable_mean | -1.45581 | -1.44623 | -0.00958 | -0.00662 |
| both | multiplicative_all_median | -1.44962 | -1.44623 | -0.00339 | -0.00234 |
| both | power_all_mean | -1.41353 | -1.44623 | 0.03270 | 0.02261 |
| both | power_bettable_mean | -1.42371 | -1.44623 | 0.02252 | 0.01557 |
| both | power_all_median | -1.41699 | -1.44623 | 0.02924 | 0.02022 |
| both | shin_all_mean | -1.42481 | -1.44623 | 0.02142 | 0.01481 |
| both | shin_bettable_mean | -1.43464 | -1.44623 | 0.01159 | 0.00802 |
| both | shin_all_median | -1.42836 | -1.44623 | 0.01787 | 0.01236 |

## Coverage (rows with a consensus, share of the current method)

| season | variant | n | n_current | share_of_current |
|---|---|---|---|---|
| 2023-24 | multiplicative_all_mean | 338227 | 338227 | 1.0000 |
| 2023-24 | multiplicative_bettable_mean | 281383 | 338227 | 0.8319 |
| 2023-24 | multiplicative_all_median | 338227 | 338227 | 1.0000 |
| 2023-24 | power_all_mean | 338226 | 338227 | 1.0000 |
| 2023-24 | power_bettable_mean | 281381 | 338227 | 0.8319 |
| 2023-24 | power_all_median | 338226 | 338227 | 1.0000 |
| 2023-24 | shin_all_mean | 338225 | 338227 | 1.0000 |
| 2023-24 | shin_bettable_mean | 281372 | 338227 | 0.8319 |
| 2023-24 | shin_all_median | 338225 | 338227 | 1.0000 |
| 2024-25 | multiplicative_all_mean | 308754 | 308754 | 1.0000 |
| 2024-25 | multiplicative_bettable_mean | 259835 | 308754 | 0.8416 |
| 2024-25 | multiplicative_all_median | 308754 | 308754 | 1.0000 |
| 2024-25 | power_all_mean | 308748 | 308754 | 1.0000 |
| 2024-25 | power_bettable_mean | 259825 | 308754 | 0.8415 |
| 2024-25 | power_all_median | 308748 | 308754 | 1.0000 |
| 2024-25 | shin_all_mean | 308749 | 308754 | 1.0000 |
| 2024-25 | shin_bettable_mean | 259821 | 308754 | 0.8415 |
| 2024-25 | shin_all_median | 308749 | 308754 | 1.0000 |

Bettable-only coverage loss by stat (share of the current method):

| season | variant | stat | share_of_current |
|---|---|---|---|
| 2023-24 | multiplicative_bettable_mean | assists | 0.7836 |
| 2023-24 | multiplicative_bettable_mean | pa | 0.8360 |
| 2023-24 | multiplicative_bettable_mean | points | 0.8692 |
| 2023-24 | multiplicative_bettable_mean | pr | 0.8608 |
| 2023-24 | multiplicative_bettable_mean | pra | 0.7482 |
| 2023-24 | multiplicative_bettable_mean | ra | 0.7617 |
| 2023-24 | multiplicative_bettable_mean | rebounds | 0.8529 |
| 2023-24 | multiplicative_bettable_mean | threes | 0.9075 |
| 2023-24 | power_bettable_mean | assists | 0.7837 |
| 2023-24 | power_bettable_mean | pa | 0.8360 |
| 2023-24 | power_bettable_mean | points | 0.8692 |
| 2023-24 | power_bettable_mean | pr | 0.8608 |
| 2023-24 | power_bettable_mean | pra | 0.7481 |
| 2023-24 | power_bettable_mean | ra | 0.7617 |
| 2023-24 | power_bettable_mean | rebounds | 0.8528 |
| 2023-24 | power_bettable_mean | threes | 0.9074 |
| 2023-24 | shin_bettable_mean | assists | 0.7837 |
| 2023-24 | shin_bettable_mean | pa | 0.8360 |
| 2023-24 | shin_bettable_mean | points | 0.8692 |
| 2023-24 | shin_bettable_mean | pr | 0.8608 |
| 2023-24 | shin_bettable_mean | pra | 0.7481 |
| 2023-24 | shin_bettable_mean | ra | 0.7617 |
| 2023-24 | shin_bettable_mean | rebounds | 0.8528 |
| 2023-24 | shin_bettable_mean | threes | 0.9075 |
| 2024-25 | multiplicative_bettable_mean | assists | 0.8804 |
| 2024-25 | multiplicative_bettable_mean | pa | 0.8379 |
| 2024-25 | multiplicative_bettable_mean | points | 0.9153 |
| 2024-25 | multiplicative_bettable_mean | pr | 0.8170 |
| 2024-25 | multiplicative_bettable_mean | pra | 0.6458 |
| 2024-25 | multiplicative_bettable_mean | ra | 0.9063 |
| 2024-25 | multiplicative_bettable_mean | rebounds | 0.8990 |
| 2024-25 | multiplicative_bettable_mean | threes | 0.8489 |
| 2024-25 | power_bettable_mean | assists | 0.8804 |
| 2024-25 | power_bettable_mean | pa | 0.8378 |
| 2024-25 | power_bettable_mean | points | 0.9152 |
| 2024-25 | power_bettable_mean | pr | 0.8170 |
| 2024-25 | power_bettable_mean | pra | 0.6456 |
| 2024-25 | power_bettable_mean | ra | 0.9063 |
| 2024-25 | power_bettable_mean | rebounds | 0.8990 |
| 2024-25 | power_bettable_mean | threes | 0.8489 |
| 2024-25 | shin_bettable_mean | assists | 0.8804 |
| 2024-25 | shin_bettable_mean | pa | 0.8378 |
| 2024-25 | shin_bettable_mean | points | 0.9152 |
| 2024-25 | shin_bettable_mean | pr | 0.8170 |
| 2024-25 | shin_bettable_mean | pra | 0.6456 |
| 2024-25 | shin_bettable_mean | ra | 0.9063 |
| 2024-25 | shin_bettable_mean | rebounds | 0.8990 |
| 2024-25 | shin_bettable_mean | threes | 0.8489 |

## Log loss by stat, 2023-24

| stat | multiplicative_all_mean | multiplicative_bettable_mean | multiplicative_all_median | power_all_mean | power_bettable_mean | power_all_median | shin_all_mean | shin_bettable_mean | shin_all_median |
|---|---|---|---|---|---|---|---|---|---|
| assists | 0.68656 | 0.68685 | 0.68662 | 0.68647 | 0.68680 | 0.68655 | 0.68649 | 0.68680 | 0.68656 |
| pa | 0.69184 | 0.69187 | 0.69199 | 0.69177 | 0.69180 | 0.69193 | 0.69179 | 0.69182 | 0.69195 |
| points | 0.69212 | 0.69225 | 0.69225 | 0.69210 | 0.69224 | 0.69223 | 0.69211 | 0.69224 | 0.69223 |
| pr | 0.69171 | 0.69172 | 0.69178 | 0.69162 | 0.69164 | 0.69170 | 0.69165 | 0.69167 | 0.69173 |
| pra | 0.69207 | 0.69212 | 0.69217 | 0.69201 | 0.69206 | 0.69212 | 0.69203 | 0.69208 | 0.69214 |
| ra | 0.69050 | 0.69058 | 0.69067 | 0.69043 | 0.69053 | 0.69062 | 0.69045 | 0.69054 | 0.69063 |
| rebounds | 0.68806 | 0.68817 | 0.68821 | 0.68797 | 0.68810 | 0.68814 | 0.68799 | 0.68811 | 0.68815 |
| threes | 0.67266 | 0.67269 | 0.67270 | 0.67226 | 0.67234 | 0.67231 | 0.67236 | 0.67242 | 0.67240 |

Mean bias (hit - p) by stat, 2023-24:

| stat | multiplicative_all_mean | multiplicative_bettable_mean | multiplicative_all_median | power_all_mean | power_bettable_mean | power_all_median | shin_all_mean | shin_bettable_mean | shin_all_median |
|---|---|---|---|---|---|---|---|---|---|
| assists | -0.00769 | -0.00753 | -0.00771 | -0.00733 | -0.00716 | -0.00736 | -0.00745 | -0.00728 | -0.00748 |
| pa | -0.01686 | -0.01712 | -0.01681 | -0.01687 | -0.01716 | -0.01680 | -0.01687 | -0.01715 | -0.01680 |
| points | -0.01830 | -0.01834 | -0.01821 | -0.01830 | -0.01834 | -0.01820 | -0.01829 | -0.01832 | -0.01819 |
| pr | -0.02285 | -0.02282 | -0.02275 | -0.02287 | -0.02284 | -0.02276 | -0.02286 | -0.02283 | -0.02276 |
| pra | -0.02322 | -0.02349 | -0.02312 | -0.02322 | -0.02352 | -0.02312 | -0.02321 | -0.02349 | -0.02310 |
| ra | -0.01478 | -0.01480 | -0.01491 | -0.01461 | -0.01462 | -0.01475 | -0.01467 | -0.01469 | -0.01481 |
| rebounds | -0.02229 | -0.02208 | -0.02254 | -0.02214 | -0.02191 | -0.02243 | -0.02218 | -0.02196 | -0.02246 |
| threes | -0.01718 | -0.01725 | -0.01716 | -0.01602 | -0.01609 | -0.01600 | -0.01644 | -0.01651 | -0.01642 |

## Log loss by stat, 2024-25

| stat | multiplicative_all_mean | multiplicative_bettable_mean | multiplicative_all_median | power_all_mean | power_bettable_mean | power_all_median | shin_all_mean | shin_bettable_mean | shin_all_median |
|---|---|---|---|---|---|---|---|---|---|
| assists | 0.68676 | 0.68678 | 0.68681 | 0.68677 | 0.68679 | 0.68684 | 0.68675 | 0.68677 | 0.68681 |
| pa | 0.69269 | 0.69266 | 0.69274 | 0.69268 | 0.69265 | 0.69275 | 0.69268 | 0.69266 | 0.69275 |
| points | 0.69212 | 0.69219 | 0.69209 | 0.69209 | 0.69216 | 0.69205 | 0.69210 | 0.69218 | 0.69207 |
| pr | 0.69274 | 0.69280 | 0.69281 | 0.69274 | 0.69280 | 0.69282 | 0.69274 | 0.69280 | 0.69281 |
| pra | 0.69276 | 0.69279 | 0.69279 | 0.69276 | 0.69279 | 0.69280 | 0.69276 | 0.69279 | 0.69279 |
| ra | 0.69238 | 0.69244 | 0.69236 | 0.69250 | 0.69255 | 0.69248 | 0.69246 | 0.69251 | 0.69243 |
| rebounds | 0.68673 | 0.68678 | 0.68680 | 0.68653 | 0.68658 | 0.68660 | 0.68659 | 0.68664 | 0.68666 |
| threes | 0.67859 | 0.67870 | 0.67858 | 0.67845 | 0.67855 | 0.67844 | 0.67847 | 0.67857 | 0.67846 |

Mean bias (hit - p) by stat, 2024-25:

| stat | multiplicative_all_mean | multiplicative_bettable_mean | multiplicative_all_median | power_all_mean | power_bettable_mean | power_all_median | shin_all_mean | shin_bettable_mean | shin_all_median |
|---|---|---|---|---|---|---|---|---|---|
| assists | -0.01117 | -0.01121 | -0.01132 | -0.01046 | -0.01050 | -0.01062 | -0.01071 | -0.01075 | -0.01086 |
| pa | +0.00185 | +0.00156 | +0.00189 | +0.00182 | +0.00151 | +0.00188 | +0.00183 | +0.00153 | +0.00188 |
| points | -0.00745 | -0.00749 | -0.00741 | -0.00740 | -0.00745 | -0.00736 | -0.00741 | -0.00746 | -0.00737 |
| pr | -0.00823 | -0.00863 | -0.00822 | -0.00825 | -0.00869 | -0.00824 | -0.00824 | -0.00867 | -0.00823 |
| pra | -0.00460 | -0.00490 | -0.00459 | -0.00445 | -0.00478 | -0.00444 | -0.00450 | -0.00482 | -0.00449 |
| ra | -0.01544 | -0.01576 | -0.01545 | -0.01531 | -0.01567 | -0.01532 | -0.01536 | -0.01570 | -0.01536 |
| rebounds | -0.01968 | -0.01976 | -0.01982 | -0.01915 | -0.01925 | -0.01929 | -0.01933 | -0.01943 | -0.01948 |
| threes | -0.01178 | -0.01183 | -0.01201 | -0.01025 | -0.01029 | -0.01050 | -0.01079 | -0.01084 | -0.01104 |

## Log loss by stat, both

| stat | multiplicative_all_mean | multiplicative_bettable_mean | multiplicative_all_median | power_all_mean | power_bettable_mean | power_all_median | shin_all_mean | shin_bettable_mean | shin_all_median |
|---|---|---|---|---|---|---|---|---|---|
| assists | 0.68667 | 0.68682 | 0.68672 | 0.68662 | 0.68679 | 0.68670 | 0.68663 | 0.68679 | 0.68669 |
| pa | 0.69218 | 0.69218 | 0.69229 | 0.69213 | 0.69214 | 0.69226 | 0.69215 | 0.69215 | 0.69227 |
| points | 0.69212 | 0.69222 | 0.69216 | 0.69209 | 0.69220 | 0.69214 | 0.69210 | 0.69221 | 0.69215 |
| pr | 0.69217 | 0.69221 | 0.69224 | 0.69212 | 0.69216 | 0.69220 | 0.69214 | 0.69217 | 0.69221 |
| pra | 0.69240 | 0.69244 | 0.69247 | 0.69237 | 0.69241 | 0.69245 | 0.69238 | 0.69242 | 0.69245 |
| ra | 0.69135 | 0.69142 | 0.69143 | 0.69137 | 0.69144 | 0.69146 | 0.69136 | 0.69143 | 0.69144 |
| rebounds | 0.68738 | 0.68746 | 0.68749 | 0.68723 | 0.68732 | 0.68736 | 0.68728 | 0.68736 | 0.68739 |
| threes | 0.67543 | 0.67549 | 0.67544 | 0.67515 | 0.67524 | 0.67517 | 0.67521 | 0.67529 | 0.67523 |

Mean bias (hit - p) by stat, both:

| stat | multiplicative_all_mean | multiplicative_bettable_mean | multiplicative_all_median | power_all_mean | power_bettable_mean | power_all_median | shin_all_mean | shin_bettable_mean | shin_all_median |
|---|---|---|---|---|---|---|---|---|---|
| assists | -0.00950 | -0.00945 | -0.00959 | -0.00896 | -0.00890 | -0.00906 | -0.00915 | -0.00909 | -0.00924 |
| pa | -0.00943 | -0.00970 | -0.00937 | -0.00944 | -0.00974 | -0.00938 | -0.00944 | -0.00972 | -0.00938 |
| points | -0.01259 | -0.01263 | -0.01253 | -0.01257 | -0.01261 | -0.01249 | -0.01257 | -0.01260 | -0.01250 |
| pr | -0.01630 | -0.01647 | -0.01625 | -0.01633 | -0.01651 | -0.01626 | -0.01632 | -0.01649 | -0.01626 |
| pra | -0.01421 | -0.01450 | -0.01416 | -0.01415 | -0.01446 | -0.01409 | -0.01416 | -0.01446 | -0.01410 |
| ra | -0.01508 | -0.01523 | -0.01516 | -0.01493 | -0.01510 | -0.01501 | -0.01498 | -0.01515 | -0.01506 |
| rebounds | -0.02096 | -0.02090 | -0.02115 | -0.02061 | -0.02055 | -0.02083 | -0.02073 | -0.02067 | -0.02094 |
| threes | -0.01466 | -0.01472 | -0.01476 | -0.01333 | -0.01339 | -0.01344 | -0.01381 | -0.01387 | -0.01391 |

## Bias by p_market bucket, 2023-24 (each variant bucketed on its own p)

| bucket | n (current) | multiplicative_all_mean | multiplicative_bettable_mean | multiplicative_all_median | power_all_mean | power_bettable_mean | power_all_median | shin_all_mean | shin_bettable_mean | shin_all_median |
|---|---|---|---|---|---|---|---|---|---|---|
| <0.35 | 1800 | -0.00851 | -0.00983 | -0.01848 | +0.00256 | +0.00358 | +0.00176 | -0.00675 | -0.00817 | -0.00572 |
| 0.35-0.45 | 25918 | -0.03148 | -0.02897 | -0.02837 | -0.02306 | -0.02024 | -0.02097 | -0.02478 | -0.02250 | -0.02259 |
| 0.45-0.50 | 110691 | -0.02345 | -0.02348 | -0.02258 | -0.02155 | -0.02174 | -0.02053 | -0.02235 | -0.02235 | -0.02149 |
| 0.50-0.55 | 108384 | -0.01466 | -0.01540 | -0.01649 | -0.01677 | -0.01711 | -0.01838 | -0.01604 | -0.01645 | -0.01772 |
| 0.55-0.65 | 22651 | +0.00301 | +0.00327 | +0.00144 | -0.00571 | -0.00735 | -0.00697 | -0.00298 | -0.00420 | -0.00403 |
| >=0.65 | 657 | +0.00006 | +0.00093 | +0.00066 | -0.01349 | -0.01358 | -0.00887 | -0.00602 | -0.00543 | -0.00776 |

## Bias by p_market bucket, 2024-25 (each variant bucketed on its own p)

| bucket | n (current) | multiplicative_all_mean | multiplicative_bettable_mean | multiplicative_all_median | power_all_mean | power_bettable_mean | power_all_median | shin_all_mean | shin_bettable_mean | shin_all_median |
|---|---|---|---|---|---|---|---|---|---|---|
| <0.35 | 812 | -0.08402 | -0.08939 | -0.08689 | -0.04186 | -0.04643 | -0.03904 | -0.05931 | -0.06288 | -0.04336 |
| 0.35-0.45 | 27652 | -0.01837 | -0.01879 | -0.01917 | -0.00945 | -0.00882 | -0.00951 | -0.01202 | -0.01159 | -0.01281 |
| 0.45-0.50 | 103279 | -0.01049 | -0.01035 | -0.00891 | -0.00879 | -0.00877 | -0.00726 | -0.00943 | -0.00940 | -0.00790 |
| 0.50-0.55 | 97302 | -0.00854 | -0.00910 | -0.00974 | -0.00984 | -0.01080 | -0.01087 | -0.00929 | -0.01006 | -0.01057 |
| 0.55-0.65 | 19635 | -0.00417 | -0.00385 | -0.00525 | -0.01393 | -0.01253 | -0.01571 | -0.01162 | -0.01110 | -0.01261 |
| >=0.65 | 173 | +0.08263 | +0.07455 | +0.06225 | +0.03858 | +0.03569 | +0.04774 | +0.08221 | +0.08566 | +0.08833 |

## Bias by p_market bucket, both (each variant bucketed on its own p)

| bucket | n (current) | multiplicative_all_mean | multiplicative_bettable_mean | multiplicative_all_median | power_all_mean | power_bettable_mean | power_all_median | shin_all_mean | shin_bettable_mean | shin_all_median |
|---|---|---|---|---|---|---|---|---|---|---|
| <0.35 | 2612 | -0.03198 | -0.03236 | -0.03955 | -0.01413 | -0.01438 | -0.01367 | -0.02571 | -0.02649 | -0.01967 |
| 0.35-0.45 | 53570 | -0.02471 | -0.02375 | -0.02363 | -0.01607 | -0.01441 | -0.01504 | -0.01823 | -0.01693 | -0.01760 |
| 0.45-0.50 | 213970 | -0.01719 | -0.01714 | -0.01599 | -0.01539 | -0.01548 | -0.01415 | -0.01611 | -0.01610 | -0.01493 |
| 0.50-0.55 | 205686 | -0.01176 | -0.01240 | -0.01329 | -0.01349 | -0.01410 | -0.01481 | -0.01284 | -0.01340 | -0.01432 |
| 0.55-0.65 | 42286 | -0.00032 | -0.00001 | -0.00168 | -0.00954 | -0.00974 | -0.01107 | -0.00700 | -0.00739 | -0.00799 |
| >=0.65 | 830 | +0.01727 | +0.01512 | +0.01357 | +0.00015 | -0.00114 | +0.00595 | +0.01600 | +0.01623 | +0.01643 |

## Choices where the plan was silent

- The keep rule uses all eight markets in the quote tables (the plan names no stat restriction); the four-bet-stat version is printed above as a secondary.
- A push (the stat equals the line) is dropped and a player who did not play has no outcome; both follow `props.models.market_bench.compare`.
- Coverage counts every consensus row of a variant (any line), before outcomes and before the common-line restriction.
- Log-loss probabilities are clipped to [1e-4, 1 - 1e-4], as `market_bench` does.
- Game date for the cluster is the ET game date of the `events` table.
