# Descriptive feature-group ablations (v2 pipeline)

One group dropped at a time, retrained with the frozen v2_base rate families; paired CRPS versus the full v2 model (positive = dropping the group hurts). 2023-24 is a development fold, 2024-25 is seen. Nothing was selected from this table and no group was removed. H4 (team-level opponent effects) is read from the `opponent` and `team_context` rows.

| season | dropped group | mean CRPS change over 8 stats (%) | points | points lo | points hi | pra | stats significantly worse (of 8) | stats significantly better (of 8) |
|---|---|---|---|---|---|---|---|---|
| 2023-24 | injury | 0.0541 | 0.0025 | 0.0009 | 0.0041 | 0.0028 | 5 | 0 |
| 2023-24 | teammates | 0.7786 | 0.0202 | 0.0153 | 0.0257 | 0.0442 | 8 | 0 |
| 2023-24 | opponent | 0.1303 | 0.0029 | -0.0004 | 0.006 | 0.005 | 5 | 0 |
| 2023-24 | team_context | -0.0931 | -0.0059 | -0.0085 | -0.0034 | -0.0058 | 0 | 4 |
| 2023-24 | rest | 0.2351 | 0.0056 | 0.0028 | 0.0084 | 0.0126 | 8 | 0 |
| 2023-24 | schedule | 0.1268 | 0.0036 | 0.0014 | 0.0057 | 0.0066 | 7 | 0 |
| 2023-24 | role | 0.1021 | 0.0029 | 0.0011 | 0.0049 | 0.0042 | 6 | 0 |
| 2023-24 | conversion | 0.0383 | 0.001 | -0.001 | 0.0028 | 0.0013 | 0 | 0 |
| 2024-25 | injury | 0.0045 | -0.0007 | -0.003 | 0.0014 | -0.0003 | 1 | 0 |
| 2024-25 | teammates | 0.9922 | 0.0303 | 0.0224 | 0.0383 | 0.0603 | 8 | 0 |
| 2024-25 | opponent | 0.0425 | 0.0023 | -0.0003 | 0.005 | 0.0029 | 2 | 0 |
| 2024-25 | team_context | -0.0849 | -0.0018 | -0.0043 | 0.0007 | -0.0061 | 1 | 4 |
| 2024-25 | rest | 0.3143 | 0.009 | 0.006 | 0.0119 | 0.018 | 8 | 0 |
| 2024-25 | schedule | 0.176 | 0.005 | 0.0023 | 0.0073 | 0.0094 | 8 | 0 |
| 2024-25 | role | 0.103 | 0.0024 | 0.0001 | 0.0046 | 0.0045 | 7 | 0 |
| 2024-25 | conversion | 0.0388 | 0.0021 | -0.0001 | 0.0041 | 0.0019 | 0 | 0 |
