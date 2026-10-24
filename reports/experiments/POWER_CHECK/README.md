# Power and size check of the registered rules (E14 slope rule, E17 interaction rule, E16 regression)

Run 2026-10-23 with `experiments/power_check.py` (300 replicates, 300 inner bootstrap draws each; results in `results.json`).

Method. Null world = the real 2019-20..2025-26 games with the estimated |spread| effect (+0.180) and product term removed from `r`. Each replicate resamples whole game days within the
discovery and within the confirmation seasons, adds a **known** effect, and applies the rule exactly as registered. "Registered" = the full rule; "confirmation only" = just the confirmation test
(no agreement with discovery); "pooled" = one test on both periods together (not allowed by the find/confirm design, shown for comparison). Monte Carlo error on a rate is about ±1.3 to ±2.9 points.

## Spread slope (E14 rule in the positive direction: discovery > 0, confirmation > 0, confirmation 95% lower bound > 0)

| injected slope (points of r per spread point) | registered | confirmation only | pooled |
|---:|---:|---:|---:|
| 0 (size) | 1.3% | 2.0% | 3.7% |
| 0.05 | 6.7% | 8.7% | 17% |
| 0.10 | 24% | 28% | 50% |
| 0.15 | 55% | 55% | 86% |
| 0.18 (observed) | 74% | 74% | 94% |
| 0.25 | 96% | 96% | 100% |

## Interaction (E17 rule: same sign in both periods and confirmation two-sided p < 0.05)

| injected product (points per spread point per total point) | change in spread slope across the interquartile range of the total (11 points) | registered | confirmation only | pooled |
|---:|---:|---:|---:|---:|
| 0 (size) | 0 | 2.7% | 7.7% | 5.0% |
| 0.005 | 0.055 | 6.7% | 8.7% | 18% |
| 0.010 | 0.11 | 19% | 19% | 39% |
| 0.015 | 0.165 | 40% | 41% | 78% |
| 0.020 | 0.22 | 68% | 68% | 94% |

## E16 regression of r on the overtime probability (analytic)

Standard error of the coefficient 197; the overtime probability differs by only 0.032 between the 0-3 and 12+ buckets. A real overtime-driven gap of 0.8 points between those buckets is detected 3% of the time, 2 points 5%, 5 points 12%.

## Reading

- The machinery is not blind: it detects a +0.18 slope 74% of the time with the registered rule (and it did detect it in E14).
- The combined rules are conservative: their false-positive rates are 1.3% and 2.7%, not 5%. The extra "agrees with discovery" condition costs almost nothing in power at large effects; most of the loss is sample size.
- Modest effects are mostly missed: a slope of 0.10 or an interaction changing the spread slope by 0.11 across the interquartile range is caught about 20-25% of the time.
- A Holm correction across 8 angles would cut the 74% to roughly 45% (z critical 1.96 -> 2.73, approximation from the confirmation-only column).
- The E16 regression has essentially no power; that is a design problem (the probability is a function of the controls), not a threshold problem.
