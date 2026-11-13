# Pre-registration: forward (paper) test of the frozen props rule, segment 1

**IN FORCE: signed by the owner on 2026-11-12, effective date E = 2026-11-13 (section 14). Every value marked OWNER below was accepted as proposed.** The text from here on is the draft exactly as signed. Nothing below binds until the owner signs it (section 14), commits it, and states the effective date. Every number marked **OWNER:** is a proposal the owner must confirm or change before signing; after signing none of them may change (a change means a new numbered segment, section 11). Prepared 2026-11-12 from the audit of the paper record; the figures quoted are from `data/props_live.db` as read on that day (1,359 bets, 1,328 settled) with `scripts/props_forward_monitor.py`, which implements every statistic below (`props/live/monitor.py`, tests in `tests/props/test_wp5_monitor.py`).

House style follows `phase5_revised_rule_preregistration.md`: plain, numbered, fixed thresholds, written before the data it will judge exists.

## 1. What is being tested

1. **The frozen rule, unchanged.** `config/props_frozen_rule.json` (sha256 `b597763ab3b13fd9cf295b1de802255ac87fe6e9bbf065103171db49eec2b6fb`): stats rebounds, assists, threes, ra; p* = p_market + k (p_model - p_market) with k per stat 0.2615 (rebounds), 0.2877 (assists), 0.3121 (threes), 0.2297 (ra); bet when EV at the best bettable price is at least 3%; earliest slot with positive EV; at least 3 books at the consensus line; one bet per player-game-stat; flat 1-unit stakes.
2. **The model**: `v3_live_2026-27r1` (bundle sha256 `7045a4ba34c4feb231ec3c7da1d6cfdd4cf6376a9503935667829525480ea2d1`, roster rule `v2`, `config/props_live.json`).
3. **Three decision slots**: T-240, T-60 and T-5 minutes before the NBA-schedule tip. The T-5 slot first ran on 2026-11-12 (evening games); before that only T-240 and T-60 bets exist.
4. **Closing reference** for CLV: the consensus snapshot of the T-5 slot, which lands about 10 minutes before tip (mean 9.9 minutes in the record). T-5 bets themselves have no CLV (the closing snapshot is the bet snapshot).
5. **The question**: does this rule, bought at the price an ordinary bettor gets, make money on games not yet played? It is a test of the rule as a whole (model, k, threshold, selection), not of any one stat.

What the test is not: it does not re-estimate k, does not tune anything, and does not test the best-price result as a trading strategy (that needs a separate fill study and pilot, sections 5 and 9).

## 2. What counts as data

1. **Included**: every row of `prop_bets` with `game_date >= E` (E = the effective date) and `model_version = v3_live_2026-27r1`, joined to its `prop_results` row. The bet record is immutable (database triggers); a bet enters the test when it is written, not when it settles.
2. **Stakes**: bets settled `win` or `loss` under the `stands` treatment (a bet stands once the player takes the court). Voids (player did not play) and pushes refund and are not stakes. A bet whose game has not been settled is not dropped; it joins when its result row is written. A game postponed past the horizon date leaves its bets out of the final test and the report lists them.
3. **Not included** (descriptive only): the record from 2026-10-22 to E minus one day. Reasons: (a) 814 of those bets are `v3_live_2026-27` rows restored from a dashboard export after the 2026-10-31 data loss and cannot be re-derived; (b) there was no T-5 slot before 2026-11-12; (c) that record has been inspected, sliced and discussed in the audit that proposed this test, so it is not an out-of-sample test of anything the audit looked at. It is shown in the monitor as "pre-registration record, descriptive only" and cannot rescue or sink the test.
4. **Prices**, as stored at the bet and never recomputed: *median* = `dec_median` (median decimal price across books quoting the consensus line at the decision snapshot), *best* = `dec_best`, *flat* = 1.909 (-110).
5. **Pooling**: T-240, T-60 and T-5 bets are pooled (they are one rule applied at three times). **OWNER:** confirm pooling; the slot split is reported but not decisive.

## 3. Primary metric

**ROI at the median bettable price** = sum of `profit_stands_median` over stakes divided by the number of stakes, with a 95% interval from a bootstrap over game dates (the cluster; 4,000 resamples, percentile interval, seed 20261112). The decision uses this metric only.

**OWNER:** price view confirmed on 2026-11-12: median is primary and best price is secondary. (The backtest picture was: best +4.63% [+2.96, +6.25], median -0.48% [-2.07, +1.06] on 2025-26; a no-model rule earned +7.0% at best price. The model's contribution is selection on top of price dispersion, which is why the median price is the honest primary.)

## 4. Record so far (descriptive, for calibrating the design)

| | Value |
|---|---|
| Stakes / game dates | 1,304 / 18 |
| ROI median price [95% date-clustered CI] | -1.0% [-6.1, +4.0] |
| ROI best price | +3.3% [-2.1, +8.6] |
| ROI flat -110 | -1.0% |
| Mean CLV (n = 1,155) | +0.32 points (clustered SE 0.06); T-240 +0.46, T-60 +0.02 |
| Win rate / mean p* / mean market fair p, bet side | 0.518 / 0.530 / 0.488; actual wins 676 against 691.1 expected under p* (z -0.8) |

## 5. Secondary metrics (reported, never decisive)

1. **ROI at the best price**, with the same interval. Interpretation is tied to the fill study: it is the value of the bets **if a limit order at that price fills**. The monitor reports the `prop_fill_checks` table (book still at or better on the next snapshot, best price still at or better, minutes the price was available, best price seen before tip) when the table exists, including best-price ROI split by whether the best price was still there. No fill proxy turns the best-price ROI into a tested result.
2. **CLV in points against the break-even**, T-240 and T-60 bets only. Break-even is derived from the record's own prices. A bet at decimal price d wins with probability p_close = p_bet + CLV if the closing consensus is right, so ROI = p_close d - 1 = 0 when CLV = 1/d - p_bet. Over the 1,154 stakes with a CLV value:

   | price | mean 1/d | mean fair p of bet side at bet time | break-even CLV | earned | ROI implied by CLV | realised ROI, same bets |
   |---|---|---|---|---|---|---|
   | median (mean d 1.942) | 0.5219 | 0.4877 | **+3.42 pts** | +0.32 | -5.8% | -1.0% |
   | best (mean d 2.028) | 0.5002 | 0.4877 | **+1.25 pts** | +0.32 | -1.7% | +3.4% |
   | flat -110 (d 1.909) | 0.5238 | 0.4877 | +3.61 pts | +0.32 | -6.3% | -1.0% |

   Reading: CLV of +0.32 points is far below what a median-price bettor needs (+3.4) and below what a best-price bettor needs (+1.25). The realised ROIs sit above the CLV-implied ones by 4.8 and 5.1 points, which at this sample size is inside the noise (half-width 5.2 points) and is exactly why the decision is made on realised ROI, not CLV. The monitor prints this table for the forward bets.
3. **Information slope per stat and pooled, with an intercept.** Regress (hit_over - p_market) on (p_model - p_market) on one row per market at its earliest slot, taken from `prop_signals` (every evaluated market, qualified or not, outcomes from the box scores) and on the bets. Estimated **with an intercept per stat** (pooled slope uses the per-stat intercepts) and also through the origin, which is how the frozen k was estimated, so the two can be compared. Date-clustered bootstrap intervals. Audit finding being checked: the market's fair P(over) ran about 2.5 points above the realised over rate on the Nov 4-11 signals (the monitor reproduces -0.025 pooled), which biases a through-origin slope (in simulation the origin estimate for a true slope 0.30 and a -0.03 bias is about 0.09, the intercept estimate recovers 0.30). Nothing is re-fitted inside this test; if the intercept version differs materially, it goes into a new pre-registration (section 11).
4. **Calibration of p***: expected wins under p* against actual wins with a clustered SE; Brier scores of p*, p_model and p_market for the bet side.
5. **Does claimed EV track realised ROI?** ROI by claimed-EV bucket (3-4, 4-5, 5-7, 7%+). Record so far: 7%+ bucket -1.1% at best price (-5.5% at median) with 304 stakes. Descriptive.

## 6. Sample size

Design from this record's own variance (18 game dates, 1,304 stakes, 72.4 stakes per date on average, residual SD per date 8.1 units, i.e. 11.2 units per 100 stakes). For G dates with the same mix, SE = sqrt(mean(resid_g^2)) / (mean stake per date x sqrt(G)) with resid_g = profit_g - ROI x stake_g, so the 95% half-width h needs G = (1.96 x 8.1 / (72.4 h))^2 dates.

| target 95% half-width on median-price ROI | game dates | settled stakes |
|---|---|---|
| 2.0 points | 121 | 8,770 |
| 1.5 points | 215 | 15,592 |

Cross-check treating bets as independent: per-stake SD 0.969, N = (1.96 x 0.969 / 0.02)^2 = 9,019 stakes, so date clustering adds almost nothing on this record (the per-date variance estimate comes from only 18 dates and is itself uncertain). A full NBA regular season has about 165 game dates, of which about 150 remain after E: **1.5 points is out of reach in 2026-27; 2.0 points is reachable by late March at the current pace** (121 dates is roughly mid-to-late March 2027).

**Proposed horizon** (fixed now, no extension): **8,800 settled stakes** (half-width about +/-2.0 points, SE about 1.02 points), or the end of the regular season **2027-04-12**, whichever comes first. If the date comes first the same final test applies to the stakes then available and the report states the half-width reached. **OWNER:** 8,800 stakes; 2027-04-12 (confirm the regular-season end date).

Power of the final test (PASS needs the lower bound above 0, i.e. an observed ROI of about +2.0 points at the horizon), by true median-price ROI: 0% -> 2.5% chance of a false PASS; +1% -> 16%; +2% -> 50%; **+3% -> 84%**; +4% -> 98%. The test is therefore built to find a median-price edge of 3 points or more; an edge of 1 point is not detectable in one season and the rule would be called FAIL, which means "not shown", not "shown to be zero" (section 8). **OWNER:** accept that design effect size of +3.0 points (this is also the alternative used by the futility rule).

## 7. Decision rule (the exact rule is `props.live.monitor.sequential_status`)

All statistics use the median-price ROI of the included bets, ROI units, date-clustered bootstrap as in section 3.

1. **Final test** (at the horizon, once): data through the first game date on which cumulative settled stakes reach 8,800, or through 2027-04-12 if that comes first. **PASS** if the lower bound of the 95% interval is above 0 **and** at least 30 game dates are included. Otherwise **FAIL**. There is no third outcome.
2. **Interim looks** (two, futility only): when cumulative settled stakes first reach 1/3 and 2/3 of the horizon (2,933 and 5,867 stakes; about late December and early February at the current pace). The data used are the settled stakes through the first game date on which the threshold is reached, not later data. At each look compute the conditional power of passing at the horizon, taking the observed ROI so far, its clustered SE and a true ROI of **+3.0%** for the remaining stakes (B-value form: Z_final ~ N(B_t + mu(1-t), 1-t), B_t = Z_t sqrt(t), mu = 0.03 / SE_final, SE_final = SE_now sqrt(t), pass when Z_final > 1.96). **STOP-FUTILITY** if conditional power is below **10%**. With a final SE of about 1.02 points that is an observed ROI below about -3.2% at the first look and below about +0.4% at the second.
3. **No stopping for success.** A strong interim result changes nothing; the test runs to the horizon. A futility-only rule cannot raise the false-PASS rate, which stays at or below 2.5% (one-sided); it costs power only when the true ROI is low (about 2 points of power at +3%).
4. **CONTINUE** otherwise. The monitor reports the status; no one acts on any other reading of the data.

**OWNER:** the 95% lower-bound-above-zero PASS criterion (an alternative is a hurdle above zero, for example lower bound > +1.0%, which needs a true ROI near +4% for 84% power); looks at 1/3 and 2/3; conditional power floor 10%; alternative +3.0%; minimum 30 dates; 4,000 resamples and seed 20261112.

## 8. What each outcome triggers (fixed now)

- **PASS**: the rule is shown to earn more than zero at ordinary prices over the segment. Triggers a separate, separately pre-registered **limit-order pilot study** on the exchanges (Polymarket, Novig) at the best sportsbook price, sized by the fill study. PASS does not authorise real money or any exchange order by itself.
- **FAIL**: the rule is not shown to earn at ordinary prices. It is **retired from the dashboard's "live" status** (shown as a paper record only), **no real-money or exchange deployment**, and no limit-order pilot is started on the strength of the best-price ROI. The report prints the point estimate and the interval next to the word FAIL so it cannot be read as proof the rule is negative.
- **STOP-FUTILITY**: the same consequences as FAIL, dated at the look; paper bets already placed settle and are reported, and the signal job may continue as a record.
- In every outcome the result, the monitor report and this document are committed together. Nothing in the secondary list (sections 5, 10) changes the outcome.

**OWNER:** confirm these consequences, in particular that FAIL retires the dashboard's "live" label.

## 9. Fill study and the best price

The owner intends to work limit orders at the best sportsbook price on exchanges. The best-price ROI (+3.3% so far, a median-to-best gap of 2.2 points of probability in the record: mean 1/d at the median 0.5219 against 0.5002 at the best) is price dispersion across books, and the backtest's no-model rule earned more than the model at the best price. It is therefore reported as "value if a limit order at that price fills" and is **not** evidence about the model. The fill study (`prop_fill_checks`, added by another work package) measures how often the price is still there; a pilot is considered only after PASS (section 8) and under its own pre-registration. **OWNER:** nothing to confirm beyond section 8; listed so the link between the two studies is on record.

## 10. One secondary hypothesis, and splits that are never decisive

**Secondary hypothesis H2 (exploratory, post hoc).** Unders on players with `expected_minutes` < 20 earn a positive median-price ROI. Origin: found by slicing the paper record, so it is a candidate and the in-sample size is optimistic (winner's curse). In the record to date: 218 stakes (about 17% of stakes, 12.8 per game date), median-price ROI +11.8% [-1.7, +23.0], best price +15.2% [+1.3, +26.8], present in both samples (v3_live_2026-27 +10.7% median on 142 stakes, r1 +13.8% on 76) and CLV +0.66 points. It is evaluated **only on bets placed on or after E**, once, at the primary horizon, with its own threshold: **at least 300 stakes, median-price ROI point estimate of at least +5.0%, and lower bound of the 95% date-clustered interval above 0.** At the current rate about 1,500 stakes are expected by the horizon, so the lower bound needs an observed ROI of roughly +5% to +6%; the point-estimate condition is then redundant but is kept to rule out a lucky lower bound. H2 cannot change the primary outcome. If it passes, the only consequence is that it may be written up as a new pre-registered rule (segment 2) for a further forward test; it never goes live on this evidence. **OWNER:** +5.0% and 300 stakes.

**Splits reported but never used for a decision** (ex post, descriptive; each with its sample size and a flag below 50 stakes): blowout margin of the game (20+ points; record so far: unders in games decided by 20+ earned about +53 units at the best price and other unders -38, an ex-post split to be reported, not tested), book of the best price, home and away, days of rest, side, stat, stat x side, slot, week, claimed-EV bucket, expected-minutes bucket, model version. The monitor produces margin, book and the others from the record; home/away and rest need a schedule join and will be added when first reported.

## 11. What voids the test; how segments are numbered

The test is valid only while one configuration produces the bets. The monitor prints the distinct (model_version, model sha256, rule sha256, roster rule) of the forward bets; a second row ends the segment. These void segment 1 (it is closed and reported as it stands, descriptive from that date):

1. a model retrain or replacement (new `model_version` or bundle sha256), including a weekly retrain if one is ever turned on for props;
2. any change to the rule file (stats, k, EV threshold, minimum books, earliest-slot logic, stake size) or its sha256;
3. a change of slot timing (T-240, T-60, T-5) or of the closing snapshot definition;
4. a change of the roster rule (`v2`) or of the price definitions (set of books, median or best computation, staleness limit).

A model or rule change starts **segment 2**, which needs its own signed pre-registration with its own effective date, horizon and thresholds; the bets before it never count towards it. Bug fixes that do not change any stored bet field (health checks, dashboard, settlement of a bet's result, CLV back-fill) do not void the test but are listed in the monitor report. A settlement bug that changes a result is corrected in `prop_results` with the correction noted in the report; the bet rows themselves are never altered. **OWNER:** segment numbering and the void list.

## 12. Multiple comparisons

One confirmatory test: the primary metric at the horizon (two interim futility looks that can only reduce passes). One exploratory hypothesis, H2, with its own threshold, which cannot affect the primary outcome and is not corrected against it. Everything else (best-price ROI, CLV, slopes per stat, calibration, every segment table and split) is descriptive: the monitor prints well over 40 intervals per report, so a couple will exclude zero by chance and none of them is a finding. The number of rules tried is not increased by this test (frozen rule, 37 tried before the 2025-26 scoring; H2 is one more candidate, 38, noted for any family-wise statement).

## 13. Reporting

The monitor (`scripts/props_forward_monitor.py --db data/props_live.db --since E`) is read-only, opens every database in `mode=ro` and writes only where `--out` points. It is run at least weekly and at each look; its output is saved dated. The report separates the pre-registration record from the forward record. The final report quotes the final test and the secondaries side by side in this document's order, whatever the result.

## 14. Signature

Effective date E: **OWNER:** 2026-11-13 proposed; (the 2026-11-12 bets, including the first five T-5 bets, are pre-registration record). Signed: Jordan Nishimura  Date: 2026-11-12

## Owner checklist (everything marked OWNER)

1. Effective date 2026-11-13.
2. Median price primary, best price secondary (decided 2026-11-12; confirm wording).
3. Pool T-240, T-60 and T-5 bets.
4. Horizon: 8,800 settled stakes or 2027-04-12 (confirm the season end), whichever first.
5. Design effect size +3.0 points (84% power; also the futility alternative).
6. PASS criterion: 95% lower bound above 0 with at least 30 game dates (or a higher hurdle).
7. Interim looks at 1/3 and 2/3 of the horizon, futility only, conditional power floor 10%.
8. Bootstrap: 4,000 resamples, seed 20261112, percentile 95% interval.
9. Consequences: PASS leads to a separately pre-registered limit-order pilot only; FAIL and STOP-FUTILITY retire the dashboard "live" status and block real-money or exchange use.
10. H2 threshold: at least 300 stakes, median-price ROI at least +5.0%, lower bound above 0.
11. The void list and segment numbering of section 11.
