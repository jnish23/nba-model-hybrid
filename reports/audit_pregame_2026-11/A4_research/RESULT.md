# A4: outside research on beating NBA pregame markets (read 2026-11-11)

Strength tags: **[S]** strong evidence (peer-reviewed or primary documentation, consistent), **[M]** some evidence (one study, a vendor's own numbers, an old sample, or a working paper), **[L]** practitioner lore (no data shown, or a source with a commercial interest). "(read)" in the source list means I fetched the page; "(snippet)" means I only saw a search-engine summary of it and the claim should be re-checked before it is relied on.

## Bottom line

1. Closing lines in NBA sides and totals are close to efficient; no peer-reviewed study I found shows a bet that stays profitable after vig on closing lines. Documented inefficiencies are small, narrow (early-season totals, big home underdogs, fatigue spots, absences in the opener) and mostly appear in 1995-2014 samples; later replications did not find them. [S for "close to efficient", M for each anomaly]
2. People who do win long-term mostly do it through price, not forecasts: better numbers early at low limits, speed on news, taking stale prices against a sharp book, and promotions. Each is capacity-limited and bookmakers restrict accounts that win. The evidence for how (and how much) is mostly practitioner lore plus one football paper (Kaunitz et al.). [M/L]
3. For a solo bettor with public box-score data, the realistic sources are line shopping/top-down use of sharp prices (as a measurement and selection tool), promotions (finite), and low-fee exchange/prediction-market pricing. Originating a better number than the market, or beating it on news speed, are not realistic with this input set. [M]
4. No public evidence was found that any named data source (RAPM/EPM/DARKO/LEBRON, tracking, referees, rest/travel) adds value beyond the closing line. The only public numbers are vendor claims against a team-rating baseline, not against the market. [gap]
5. CLV is the right yardstick: per-bet CLV is roughly ten times less noisy than win/loss (Buchdahl, who sells a tool built on it), so tens to low hundreds of bets, not thousands. Its limits: it assumes the close is efficient, and an originator can show no CLV at first. [M]
6. Pregame prediction markets: Polymarket US's own fee page (read today) shows a 0.095 coefficient for small accounts (2.4 cents per contract at 50c, about the same as -110 vig), 0.0695 for volume above $250k a month and for an account's first 7 days; Novig charges no pregame fee to either side. I could not read Kalshi's fee PDF (HTTP 429). No published study benchmarks pregame NBA prices on these venues against Pinnacle. [S for fees, gap for lead/lag]
7. Best-evidenced changes for the owner's model: (i) bet against the same-time market the model sees, i.e. put the latest line into the model or the bet rule (Dare et al. 2015, Moskowitz 2021); (ii) pick and tune models by calibration/log loss and betting-ROI on out-of-sample lines, not accuracy (Walsh and Joshi); (iii) shrink stakes for estimation error (Baker and McHale); (iv) player-availability-aware inputs (expected minutes, injuries) have a vendor-reported 3% MAE gain over a team baseline. Evidence is thin; none is a replicated claim of beating the close.
8. Not supported: public-fade/contrarian angles, streak or "hot hand" betting, referee-crew angles, steam chasing, and accuracy-first model selection.

---

## 1. Can modellers beat the market? Efficiency of NBA and US-sport closing lines

**Academic consensus.** Sauer's survey concludes that wagering prices are, to a first approximation, efficient forecasts, with important departures (favourite-longshot bias, heterogeneous bettors, transaction costs) [S] (Sauer 1998, JEL, abstract via EconPapers). Levitt (2004) adds that bookmakers do not just clear the market: they shade prices to exploit bettor biases, and they appear to be better forecasters than their customers; the NFL contest data show no sign of skilled bettors [M] (Levitt 2004, Economic Journal, snippet).

**NBA sides and totals, closing lines.**
- Compton and Sigler: 10,325 games (2000-01 to 2007-08), Gold Sheet lines. Overs won 49.87%, underdogs 49.86%; no subgroup beat the 52.38% break-even except 25-game home-underdog cells that faded in the second half of the sample. Conclusion: efficient, nothing profitable after vig. [M: one 8-season sample, their tables contain typos, many subgroup tests] (read)
- Paul and Weinbach (2005, 1995-96 to 2001-02) found big home underdogs and fading streaks profitable and overbetting of favourites, i.e. a rejected fair-bet null; the later replication above found the underdog edge did not persist [M] (snippet). Paul and Weinbach (2008, IJSF) found that betting against public sentiment did not win more than efficiency implies, and Paul, Humphreys and Weinbach (2011) found the same for "hot hand" fades [M] (snippet).
- Totals: Baryla et al. (2007) found early-season NBA totals biased (sides were not); Girdner et al. (2013) found the totals market efficient except week 1 of 2009-2012, with unders hitting 58.2% of first-week games and a reported 11.1% return per game for betting under in week 1 [M: small first-week samples, and neither paper says whether the bias survives past 2012] (snippet of abstracts).
- Fatigue: Ashman, Bowman and Lambrinos (2010, 19 years) found home teams in the second game of a back-to-back, facing a rested visitor, did poorly against the spread, more so after eastward travel [M] (read). I found no post-2010 NBA test of it; practitioner pages say it is now priced (L).
- Player absence: Dare, Dennis and Paul (2015) found the opening line often misprices games with absences but that the closing line leaves no profitable strategy [M] (read). This is directly relevant to a model that reads the opener.
- Cross-sport: Moskowitz (2021, Journal of Finance) finds momentum and (weaker) value effects in the price move from open to close across NBA, NFL, MLB and NHL contracts, fully reversed by the outcome, but returns are a fraction of those in equities and do not cover transaction costs [S] (read abstract).
- Anomaly episode: an unreviewed arXiv paper reports NBA moneylines were markedly inefficient in the 2020 no-fan period (underdog bets +16.7%); no other sport showed it [L/M: not peer-reviewed; explanation untested] (read abstract).
- Season-wins totals (a different market) showed profitable over/under strategies from bettor bias toward last season's record [M] (read).

**Have inefficiencies shrunk?** The pattern across sources is that edges published in 1995-2008 samples did not reproduce in later samples, and the few post-2010 findings (Ashman 2010, Dare 2015) are either about the opener or untested on recent data. I found no study that estimates a time trend directly, and no peer-reviewed NBA closing-line efficiency study covering 2015-2025. That absence is itself a gap in the literature, not evidence that the market got more efficient.

**Counter-evidence that models can win.** Hubáček, Šourek and Železný (2019, Int. J. Forecasting) report positive cumulative profit on NBA 2007-2014 using a model trained to be decorrelated from the bookmaker's odds, with CNNs and portfolio-style staking [M: single sample, older seasons, authors' own pipeline] (read). Walsh and Joshi (2024) report ROI +34.7% vs -35.2% for calibration- vs accuracy-selected NBA models on one held-out season [M: single season, ROI figures changed between arXiv versions, so treat the size with suspicion] (read). Kaunitz, Zhong and Kreiner (2017) beat football books using the consensus of other books' odds and were then restricted [M] (read).

**Disagreement.** Academic closing-line studies mostly say "efficient"; Pinnacle-linked and tout-adjacent sources say winners exist but are limited. Both are consistent with: closing lines are hard to beat, but opening and soft-book prices can be.

## 2. How the winners win: distinct sources of edge

| Source | How it works | Needs | Durability | Strength | Solo/public-data 2026? |
|---|---|---|---|---|---|
| (a) Originate a better number, bet early at low limits | Bet the opener or an early line where you have a better view; the market then moves toward you. Dare et al. show openers misprice absence games. Pinnacle's help page (snippet) says NBA lines post about 2pm PT the day before with $5,000 spread and $2,000 total opening limits, rising to $20,000/$10,000/$5,000 about 07:00 PT game day (page returned 404 on fetch, so unverified) | A model that already beats the opener, an account with access, a stake size that fits low limits | Fragile: early limits are small, and the opener is exactly what sharp books use as a baseline | M | No for the owner's setup: model trails the close; a US resident also generally cannot use Pinnacle (my background knowledge, not verified here) |
| (b) Information speed (injuries, lineups) | Act on news before lines move. Official NBA injury report is published every 15 minutes, with a game-day window of 11am-1pm local (8-10am for early tips) and lineups 30 min before tip (rule from 2021) (read). Dare et al. say the close removes the bias | Low-latency feeds, bet execution in seconds, accounts that are not limited | Eroding; books and sharp bettors share the same public sources; Westgate's risk VP said lineup timing changes little (2021 article) (read) | M/L (no timing study found) | No: a noon batch run cannot compete |
| (c) Top-down/line shopping/steam | Treat a sharp book's no-vig price as fair value and bet softer books' stale prices or better numbers. Kaunitz et al. did this with consensus odds in football and were limited to as little as $1.25 stakes (the limit figure is from press coverage) | Several books/accounts, a sharp reference price, fast scraping | Short-lived per account because softer books restrict; steam-chasing after the move is lore: "roughly break-even before vig" (one site, unverified) | M for top-down (one paper), L for steam | Realistic as a measurement tool and for small stakes; not as a business |
| (d) Promotions and bonuses | Sign-up offers and boosts have positive EV when used correctly; bonus bets are worth well under face value; rollover requirements remove much of the deposit-match value | Many accounts, careful hedging, capital | Finite per account; books exist to recoup via lifetime losses | L (affiliate-heavy sources; no study found of net profit) (snippet) | Yes but one-off and capped |
| (e) Market making / liquidity provision | Post limit orders on exchanges. On Kalshi, makers do better than takers (maker average loss about a third of the taker loss in one summary), but both groups lose on low-priced contracts in the Burgi/Deng/Whelan data: still a negative average return before skill. Pregame Novig has zero fee to both sides; Polymarket US pays small rebates only above $10M maker volume | Capital, inventory risk management, a fair-value model, patience | Depends on fee schedule and on being not adversely selected; takers who are better informed than you eat makers | M (read abstract: makers earn more, both show favourite-longshot) | Plausible, small. A maker with no information edge is paid the spread but adversely selected |
| (f) Derivative and niche markets | Props, alternates, quarters/halves, small leagues: less sharp attention and lower limits | Models for those markets, many accounts | Books drop or cap the soft ones; NBA ordered certain prop types removed for 2026-27 (low-stat unders, opening tip, some bench props) (read) | L: no peer-reviewed NBA prop efficiency study found; the only prop data I saw is an undergrad thesis and an NFL blog | Most plausible for a public-data modeller but unproven; limits are the binding constraint |
| (g) Syndicate scale | Many models, many accounts, runners, execution infrastructure. Voulgaris (ESPN/Guardian via snippets) says a model took years of data and testing | Capital, staff, data licences | Works only while there is access; Pinnacle claims not to limit winners (company's own marketing) | L | No |

**Which are realistic for a solo bettor with a public-data model:** (c) for validation and small stakes, (d) as one-off, and (e)/(f) as experiments on low- or zero-fee venues. (a), (b), (g) are not.

**Caveat.** I found no audited track record of an outside bettor beating NBA closing lines over many years. Everything for "who wins and how" rests on interviews, vendor claims and one football paper.

## 3. Data used by winning NBA modellers (public? cost? evidence beyond the close?)

| Data | Public / cost | Published evidence it adds beyond closing line or a box-score model |
|---|---|---|
| Player-impact metrics (RAPM, EPM, DARKO, LEBRON, BPM) | EPM (Dunks and Threes): ratings page with subscribe/login; API requires a subscription; price not listed (read). DARKO: leaderboard viewable today; pricing not stated on the page I read; methodology not on that page (read). LEBRON/BPM: not checked | EPM's own page: game predictions with expected minutes, injuries, rest and opponent strength averaged 9.6-point miss in a 24,425-game backtest vs 9.9 for opponent-adjusted net rating; live 2024-26 11.0 vs 11.4 on 2,643 games; the comparison is against a team baseline, not the closing line, and the model changed this season (read). I found no study comparing any of these to the market. Evidence level M (vendor) |
| Projected minutes / rotations | Mostly inside paid projection products; EPM reports a 14.5% improvement in minutes forecast over season-to-date average (vendor) | None against the market; but absence-driven line errors (Dare 2015) suggest minutes/availability is the component that matters |
| Lineup data | pbpstats and NBA.com possession data are public but terms unclear | None found |
| Injury-news feeds | Official NBA report is free and 15-minute granular; teams must file by 5pm local the day before (1pm same day for second night of a back-to-back) and a game-day report between 11am and 1pm local (read) | Dare et al. 2015 (read): openers err in absence games, closes do not. Speed claims in practitioner pages are unsupported |
| Tracking data | NBA.com tracking endpoints (since 2013-14) are free but unofficial and rate-limited; Second Spectrum is licensed by the league (low-authority sources) (snippet) | EPM uses tracking to model defence; no evidence found that tracking adds value against the close |
| Rest and travel | Free (schedules) | Ashman et al. 2010 (read); later practitioner pages say it is priced; no post-2010 academic test found |
| Referee assignments | NBA posts assignments at about 9:00am ET each game day (read) | No rigorous test found; pages claiming a crew edge come from tool vendors with no line control (L) |
| Market data (cross-book prices, movement, limits, handle) | Free to see at a few books; paid APIs for history (The Odds API costs credits; not used by this audit) | Strongest candidate: Kaunitz consensus approach, Hubáček decorrelation, Moskowitz open-to-close drift. Handle/limit data is not public in any form I found |

## 4. Selecting and sizing bets

**Closing-line value.** Buchdahl (via Pinnacle Odds Dropper, which sells a CLV-based service) puts the per-bet standard deviation of CLV at about 0.1 against about 1.0 for profit and loss at even money, so a 5% CLV edge shows in about 50 bets where results need several thousand (read). My own check of the arithmetic (2 standard errors): with sd 1.0 and a 5% true edge you need about (2x1.0/0.05)^2 = 1,600 bets; with sd 0.1 about 16, so his 50-bet figure is conservative. Football-data.co.uk's test of tipsters uses the same logic (read). Pinnacle's own head of trading is quoted (secondary) as saying beating the close by 10% means about 10% long-run profit [L: company marketing].

**Limits of CLV.** (1) The close must be an efficient, correctly de-vigged price; Buchdahl calls the equal-margin de-vig crude and says the premise holds best for popular markets (read). (2) An originator whose information is not yet in the sharp book's price shows no CLV at first (read). (3) Arbitrage flow can push Pinnacle's close the wrong way (snippet of a Buchdahl post). (4) A sample of 20,000 bets gave 3.4% actual profit vs 4.0% expected from CLV (his own data) (read). A separate "Sharpery" study quoting correlations of 0.42 at 200 bets and 0.81 at 1,200 bets has no identified authors and I did not rely on it (snippet).

**Edge thresholds vs vig.** A -110 bet needs 52.38% (4.5% of stake). Under the rule "bet when the probability edge exceeds 5% over break-even", the threshold is meaningful only if the model's probabilities are calibrated; Walsh and Joshi find calibration, not accuracy, predicts betting returns on NBA data (read). Compton and Sigler show subgroup hit rates reaching 54-59% in small cells that did not hold up out of sample (read): thresholds chosen on subgroups overfit.

**Kelly.** Baker and McHale (2013, Decision Analysis): when the win probability is itself estimated, stakes should be shrunk below raw Kelly, and their simulations and tennis data show shrunk Kelly beats raw Kelly out of sample [M] (snippet). Common practice (quarter to half Kelly) is practitioner lore without a source I could verify. Hubáček et al. use modern-portfolio-style staking over simultaneous bets (read).

**Correlation.** Same-game spread and moneyline bets are driven by the same margin, so treat them as one position. Totals are only weakly related to sides (my reasoning, not from a source). The owner's own saved predictions can measure this directly.

**Account limits.** Kaunitz et al. report discriminatory stake limits after a few wins (read abstract). Pinnacle's stated policy is that it does not limit winners, to attract sharp volume; this is company marketing and unverified (snippet).

**Timing.** Evidence for "which time" is thin: openers err in absence games and closes do not (Dare), prices drift open to close in the direction of momentum and reverse on the result (Moskowitz). I found no NBA study that compares bet timing (opener, morning, post-injury report, close) on profit.

## 5. Prediction markets and no-vig exchanges (fees quoted only from venue pages; read 2026-11-11)

**Polymarket US.** Fee page (https://docs.polymarket.us/fees, read 2026-11-11; the page has no last-updated date, the exchange-wide effective date it gives is 10:00 pm ET, 2026-10-28): fee = coefficient x contracts x p x (1-p). Taker coefficient by trailing-30-day taker volume: 0.095 below $10,000; 0.085 from $10,000 to $50,000; 0.075 from $50,000 to $250,000; 0.0695 above $250,000. New accounts: 0.0695 in days 1-7, then the higher of 30/7 x first-week volume and the trailing volume. Maker rebate: none below $10M of qualifying maker volume (then -0.0035/-0.0070/-0.0104). Taker volume rebates start at $250k (3%). Fees round to the nearest cent (banker's rounding). No sports-specific rate is listed on this page. At 50c the cost is 2.4 cents per contract at 0.095 (1.7 at 0.0695), about the same as the 2.38 points of break-even over 50% on a -110 bet. Changelog: https://docs.polymarket.us/changelog (read) records the rebate tiering from 2026-10-19 and a combo-fee change on 2026-11-11 (combo markets, not game markets).

**Polymarket international.** https://docs.polymarket.com/trading/fees (read 2026-11-11, no date on the page and no mention of the US exchange): sports taker coefficient 0.05, maker 0, maker rebate 15%; peak fee $1.25 per 100 shares at 50c. This differs from the US page, so do not apply one to the other. A February-March 2026 study states Polymarket charged zero fees on NBA markets in its sample window (read), so earlier price-quality studies are not fee-comparable. Third-party pages giving 0.30%/0.20% or 5% US rates conflict with the primary page and are ignored.

**Novig.** Fees article (https://support.novig.com/en/articles/16195057-fees-on-novig, read 2026-11-11, "updated over 3 weeks ago"): pregame straight trades, app/web and API: no fee for maker or taker. Live straight and futures takers pay 0.06 x P x (1-P) per contract; parlays 0.10; maker fee $0 across trade types. The page's own FAQ section contradicts the live coefficient (0.03 vs 0.06). Pregame fee-free status is stated as current, not guaranteed. Eligibility (https://support.novig.com/en/articles/16194031-eligibility-legality, read; dated September 2, 2026): available nationwide except Nevada, Arizona and Michigan, based on physical location at the time of trading, age 21+, one account per person. Another Novig blog page (state list, snippet only) names Ohio as well, and Novig blog posts (snippet only) describe a September 2026 Connecticut cease-and-desist and a New York suit filed August 5, 2026; the support page does not mention either. Stated neutrally: the excluded-state list differs between Novig's own pages, and legal status is disputed in at least some states, so check the in-app eligibility check before relying on it.

**Kalshi.** The help article (https://help.kalshi.com/en/articles/13823805-fees, read) confirms some markets have maker fees (charged only on execution) and points to https://kalshi.com/docs/kalshi-fee-schedule.pdf. I could not read that PDF (HTTP 429 twice), so I cannot confirm the NBA coefficient. A search-engine summary of it gave taker 0.07 x P x (1-P) with a per-series multiplier and maker 0.0175, with NBA game series possibly among those that charge makers; treat this as unconfirmed.

**ProphetX and Sporttrade.** Only affiliate or review pages found (2% on net winnings; ProphetX 1% on some NBA props; Sporttrade in five states and applying for CFTC status). No venue documentation read; fee and eligibility numbers unconfirmed.

**Spreads and depth.** Cheng et al. (arXiv 2605.00864, read): 173 Polymarket NBA games, 75 million top-of-book snapshots, median bid-ask spread of 392 bps pregame, 1,031 bps in-game (units as the paper reports them), average executable size about 14.8 shares in the arbitrage episodes, and 76.9% of combinatorial episodes could not deploy a $100 budget: depth is the constraint, "nearly efficient in aggregate". This is Polymarket's global book in Feb-Mar 2026; I found no equivalent measurement for Polymarket US, Novig or Kalshi NBA pregame.

**Lead or lag against sharp books.** I found no published study that benchmarks pregame NBA prices on Polymarket/Kalshi/Novig against Pinnacle or the sportsbook close. Related evidence: Angelini and De Angelis (arXiv 2606.07811, abstract read) find real-time prediction-market prices underreact to public signals (about 0.64 of a benchmark probability change in a minute), worse in thin markets; Moshrefi (arXiv 2607.14430, read) finds Kalshi sports moneyline prices well calibrated mid-contract but departing sharply in the last ten minutes, and cross-game parlays overpriced; Burgi, Deng and Whelan find favourite-longshot bias and better results for makers on Kalshi (read abstract); Aktuğ and Torul (2026) report exchange venues carrying most information share in 2024 election markets (snippet; politics, not sports). None answers the question for NBA games.

## 6. Which changes have the best evidence for a model like this one?

Context from BRIEF.md: inputs are rolling box-score averages, the noon injury report and the opening line, and the model trails the close.

**Best-evidenced changes** (all with the caveat that no public study shows any of them beating the close for the NBA after 2015):
1. **Use the latest line when betting, and treat the market as the prior.** Dare et al. found opener errors that the closing line removes (read); Moskowitz finds the open-to-close move carries information; Kaunitz et al. and Hubáček et al. both build from market prices. This is consistent with preliminary finding P1/P2 in BRIEF.md (edges measured against a later line than the one the model read). [M]
2. **Select and tune by calibration/log loss against market-priced out-of-sample bets, not accuracy.** Walsh and Joshi (read). [M: one season]
3. **Shrink stakes for estimation error.** Baker and McHale (snippet). [M]
4. **Availability-aware player inputs (expected minutes, injuries).** Only vendor evidence: 3-4% MAE improvement over a team-rating baseline (EPM page, read) and the Dare et al. absence-error result. Neither compares against the closing line. [M/L]
5. Early-season totals, if tested out of sample on recent years: the only recurring seasonal bias in the totals literature (Baryla 2007; Girdner 2013), but the data end in 2012 and neither study reports a recent replication. [M, possibly dead]

**Commonly tried changes with evidence of not helping (or no evidence):**
- Fading the public / sentiment or hot-hand contrarian angles: Paul and Weinbach (2008) and Paul, Humphreys and Weinbach (2011) find no win rate above efficiency (snippet). [M]
- Betting momentum or streaks: Moskowitz finds the price already moves with momentum and the effect does not cover costs (read). [S]
- Optimising for accuracy: lower returns than calibration-selected models (Walsh and Joshi). [M]
- Rest/back-to-back rules of thumb: one academic finding from 2010, no recent replication; commercial and practitioner pages say it is priced. [M/L]
- Referee-crew and steam-following angles: no rigorous tests found; claims come from vendors or sites with no line control. [L]
- Subgroup rules (big home dogs etc.): small-cell wins that did not persist (Compton and Sigler). [M]

## Not verified / limits

- WebSearch returns model-written summaries, so claims marked "snippet" were not checked against the page and may be wrong in detail.
- The PDF of Kalshi's fee schedule, the Pinnacle NBA limits page, the Moskowitz paper body, the Brill 2023 DARKO paper, the Whelan Kalshi paper body and Novig's state blog page could not be read (HTTP 429/404/405 or PDF binary). Several fetches saved a binary PDF automatically into the user's Claude tool-results folder under `C:\Users\jorda\.claude\projects\...\tool-results\`; I did not choose or open them. If "do not download" is meant strictly, delete them.
- I found no NBA-specific, peer-reviewed study of closing-line efficiency for 2015-2025, none on pregame NBA prediction-market lead/lag, and none on NBA props efficiency beyond an undergraduate thesis I only saw listed.
- I did not check Polymarket US order-book depth for NBA games; the owner's own feed would answer that.
- I did not verify that Pinnacle is unavailable to US residents (background knowledge only).

## Source list

Primary documentation (venues), all read 2026-11-11:
1. Polymarket US fee schedule: https://docs.polymarket.us/fees (read)
2. Polymarket US changelog: https://docs.polymarket.us/changelog (read)
3. Polymarket international fees: https://docs.polymarket.com/trading/fees (read)
4. Novig fees: https://support.novig.com/en/articles/16195057-fees-on-novig (read)
5. Novig eligibility: https://support.novig.com/en/articles/16194031-eligibility-legality (read)
6. Kalshi fees help: https://help.kalshi.com/en/articles/13823805-fees (read); PDF https://kalshi.com/docs/kalshi-fee-schedule.pdf (not read, HTTP 429)
7. NBA injury report rules: https://official.nba.com/nba-injury-report-2025-26-season/ (read)
8. NBA referee assignments: https://official.nba.com/referee-assignments/ (read)

Academic and working papers:
9. Compton and Sigler, NBA Gambling Inefficiencies: A Second Look: https://thesportjournal.org/article/nba-gambling-inefficiencies-a-second-look/ (read)
10. Dare, Dennis and Paul 2015, Player absence and betting lines in the NBA: https://ideas.repec.org/a/eee/finlet/v13y2015icp130-136.html (read)
11. Ashman, Bowman and Lambrinos 2010, fatigue: https://econpapers.repec.org/RePEc:sae:jospec:v:11:y:2010:i:6:p:602-613 (read)
12. Moskowitz 2021, Asset Pricing and Sports Betting: https://econpapers.repec.org/RePEc:bla:jfinan:v:76:y:2021:i:6:p:3153-3209 (read abstract)
13. Hubáček, Šourek and Železný 2019: https://ideas.repec.org/a/eee/intfor/v35y2019i2p783-796.html (read)
14. Walsh and Joshi: https://arxiv.org/abs/2303.06021 (read)
15. Kaunitz, Zhong and Kreiner 2017: https://arxiv.org/abs/1710.02824 (read)
16. Burgi, Deng and Whelan, Makers and Takers: https://econpapers.repec.org/RePEc:cpr:ceprdp:20631 (read abstract)
17. Cheng, Yang and Zou, Arbitrage Analysis in Polymarket NBA Markets: https://arxiv.org/html/2605.00864v1 (read)
18. Moshrefi, Systematic Bias in Sports Prediction Markets: https://arxiv.org/abs/2607.14430 (read)
19. Angelini and De Angelis: https://arxiv.org/abs/2606.07811 (read)
20. Qureshi and Zaman, COVID and sports betting: https://arxiv.org/abs/2109.07581 (read, not peer-reviewed)
21. Woodland and Woodland 2015, NBA season wins: https://econpapers.repec.org/RePEc:jsf:intjsf:v:10:y:2015:i:2:p:160-174 (read)
22. Sauer 1998: https://econpapers.repec.org/RePEc:aea:jeclit:v:36:y:1998:i:4:p:2021-2064 (snippet)
23. Levitt 2004: https://econpapers.repec.org/RePEc:ecj:econjl:v:114:y:2004:i:495:p:223-246 (snippet)
24. Baker and McHale 2013: https://econpapers.repec.org/RePEc:inm:ordeca:v:10:y:2013:i:3:p:189-199 (snippet)
25. Baryla et al. 2007 and Girdner et al. 2013 (early-season totals): https://dc.etsu.edu/etsu-works/17780 and https://ideas.repec.org/a/buc/jpredm/v7y2013i2p1-9.html (snippet)
26. Paul and Weinbach 2005/2008 and Paul, Humphreys and Weinbach 2011: https://econpapers.repec.org/RePEc:sae:jospec:v:6:y:2005:i:4:p:390-400 and https://econpapers.repec.org/RePEc:jsf:intjsf:v:3:y:2008:i:3:p:137-145 (snippet)

Practitioners and vendors (commercial interest flagged):
27. Buchdahl on CLV, Pinnacle Odds Dropper (sells a subscription service built on this method; says it is not affiliated with Pinnacle): https://www.pinnacleoddsdropper.com/blog/closing-line-value--clv-demystified-by-expert-joseph-buchdahl (read)
28. Buchdahl, closing odds as a skill test: https://football-data.co.uk/blog/closing_odds.php (read; the site runs on betting-data and tipster-testing)
29. Dunks and Threes EPM methodology and results (vendor, paid API): https://dunksandthrees.com/about (read)
30. DARKO: https://darko.app/ (read; no methodology on the page)
31. Pinnacle NBA limits article (snippet; 404 on fetch): https://pinnaclesports.freshdesk.com/support/solutions/articles/1000115292-when-are-nba-limits-and-odds-available-
32. Action Network on the 30-minute starting-lineup rule (Sept 2021): https://www.actionnetwork.com/nba/nba-rule-change-starting-lineups-sports-betting (read; sportsbook-affiliated media)
33. NBA prop removals, Oct 2026: https://www.wafb.com/2026/10/19/nba-orders-sportsbooks-drop-certain-bets-this-season/ (read; cites ESPN)
