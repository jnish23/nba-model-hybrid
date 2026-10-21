# Void and settlement rules for NBA player props, by book

Compiled 2026-10-19. **Status: partly verified.** Only one book's own rule text could be read (Fanatics, Ohio house rules). The other books' rules below come from
news and explainer articles and must be checked against each book's current house rules (state-specific) before any live use or paper-trading settlement.
The model's target is always "the outcome given the player plays", which is correct for every book below; what differs is the edge cases.

## 1. The common rule (all sources agree)

- A player who is **inactive or does not play at all**: the prop is **void** and the stake refunded, over and under alike.
- A player who **takes the court** (even briefly) **and is then injured**: the bet **stands**. Overtime counts for full-game props.
- A void leg inside a parlay is dropped and the parlay is repriced on the remaining legs.

## 2. By book

| Book | What I found | Source type | Confidence |
|---|---|---|---|
| **Fanatics** | Basketball > General Player Prop Rules: a player must "enter the court and receive at least 1 second of playing time for bets to have action"; inactive players void the bet; Fanatics "may void selections" if the player "leaves the game during the 1st/2nd Quarter (WNBA + NBA), without returning to the game due to injury"; such voids are "entirely at the discretion" of Fanatics; markets settle on the box score. Separately, its "Fair Play" policy is reported to refund overs after early-game injuries (first quarter in NBA regular season, first half in the playoffs); unders stand | **Primary** (Ohio house rules, https://sportsbook.fanatics.com/legal/oh/house-rules/); Fair Play from Action Network | High for the quoted text; Fair Play window unverified (reported, may have changed) |
| **FanDuel** | For NBA, WNBA and NCAA basketball props, a bet stands once the player takes the court regardless of minutes; no participation at all voids it; totals include overtime. One report says FanDuel voided props on a player who played about eight minutes while leaving others standing; FanDuel denied breaking its policy | Secondary (quoted FAQ in explainers: https://www.actionnetwork.com/education/does-fanduel-void-injured-player-props ); FanDuel's house-rules page returned HTTP 403 | Medium |
| **DraftKings** | No published NBA prop wording found. Mid-game injuries generally stand. After the 2025 Game 7 Haliburton injury DraftKings refunded losing bets on markets still undetermined when he left (a discretionary refund, not a standing rule); in the Draymond Green seven-second case it held bets under review, then paid out after a delay | Secondary (news: https://www.covers.com/industry/multiple-sportsbooks-void-tyrese-haliburton-bets-after-early-game-7-injury ); rules page returned HTTP 403 | Low to medium |
| **BetMGM** | Described as the traditional "Full Action" rule: the prop is void if the player records no minutes and stands once he plays. In the Draymond Green case it paid winners and graded overs as losses | Secondary (https://rotogrinders.com/sports-betting/does-betmgm-void-prop-bets ) | Medium |
| **Caesars** | House rule listing "Basketball: player must play" for a player proposition to have action; same Full Action approach | Secondary (https://www.actionnetwork.com/education/prop-betting-rules-what-happens-if-player-doesnt-play, updated 2026-05-05) | Medium |
| **BetRivers** | Nothing specific found | None | Unknown: assume the common rule |
| **Bovada** | Nothing specific found | None | Unknown: assume the common rule |
| **BetOnline** | Nothing specific found | None | Unknown: assume the common rule |

## 3. Consequences for the modelling and backtests

1. **Early exits are where books differ.** Fanatics (and, in practice, sometimes DraftKings) can void or refund an **over** after an early injury, while the under stands. For those books the over's true win probability conditional on settlement is higher than the raw P(over): the voided early-injury cases leave the sample. Our "limited minutes" regime (8.9% of played games) includes early injuries but also coach decisions, foul trouble and blowouts, and **the box score cannot tell them apart**, so no historical settlement can be exact.
2. **Phase 5 will therefore settle every bet under two bounding treatments** and report both: (a) the bet stands whenever the player played at all; (b) overs are voided when the player's minutes fall below a stated fraction of expectation (a proxy for early injury exits) at the books with an early-exit rule. The conclusions must hold under both, or the difference is reported as a limit.
3. **Per-book settlement is data, not code:** a table `book -> rule` (stands once on court / may void early injury overs / unknown) used by the simulated betting, defaulting to the common rule for books with unknown wording.
4. **Open action for you before paper trading:** read each book's current NBA player-prop house rules in its app for your state (the in-app wording is the contract) and send me any that differ from this table; I will update the table and re-run the settlement sensitivity.
