# E08_fresh_chronic: step 1 - hypothesis test on discovery seasons ['2019-20', '2020-21', '2021-22'] (2558 games with reports for both teams)

## spread residual ~ fresh + chronic (registered)

| term               |    coef | 95% CI             |
|:-------------------|--------:|:-------------------|
| intercept          | -0.3561 | [-0.855, +0.135]   |
| x_inj_fresh_lost   | -0.1734 | [-0.2941, -0.0475] |
| x_inj_chronic_lost | -0.0161 | [-0.1643, +0.1240] |

## spread residual ~ fresh + chronic + mid (control)

| term               |    coef | 95% CI             |
|:-------------------|--------:|:-------------------|
| intercept          | -0.3567 | [-0.859, +0.135]   |
| x_inj_fresh_lost   | -0.1723 | [-0.2939, -0.0469] |
| x_inj_chronic_lost | -0.0157 | [-0.1631, +0.1239] |
| x_inj_mid_lost     |  0.0641 | [-0.1636, +0.2745] |

## total residual ~ sum of home+away fresh / chronic (descriptive)

| term      |    coef | 95% CI             |
|:----------|--------:|:-------------------|
| intercept |  0.0722 | [-0.950, +1.090]   |
| s_fresh   |  0.0398 | [-0.1442, +0.2234] |
| s_chronic | -0.0176 | [-0.1861, +0.1659] |

## Mean fresh / chronic / mid lost value per team-game (discovery seasons)

|      |   inj_fresh_lost_home |   inj_chronic_lost_home |   inj_chronic_lost_dec_home |   inj_mid_lost_home |
|:-----|----------------------:|------------------------:|----------------------------:|--------------------:|
| mean |                 1.56  |                   1.076 |                       0.292 |               0.498 |
| std  |                 2.784 |                   2.564 |                       0.78  |               1.685 |

## Result of the pre-registered test: hypothesis **SUPPORTED**
- fresh coef -0.1734, CI excludes zero & negative: True
- chronic coef -0.0161, CI includes zero or |chronic| < 0.5|fresh|: True
