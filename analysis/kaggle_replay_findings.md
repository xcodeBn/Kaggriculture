# Kaggle replay: first loss diagnosis

Source: local copy `replays/kaggle_replays/kaggle_game.json` (the raw replay is
ignored by Git; original Kaggle replays belong in that folder).

## Result

- Our agent is player 0: final reward / money **8,111**.
- Opponent is player 1: final reward / money **144,750**.
- This was a loss by **136,639**.

Player 0's action and market-order pattern matches this project's current crop agent, so the player assignment is clear.

## What the opponent did

The opponent built a livestock business. It bought 2 cows and 16 sheep, built 18 pastures, and used 205 hire orders. The replay shows the flock growing from 3 animals around turn 120 to 9 at turn 240, 13 at turn 360, and 18 by turn 600. Its market actions requested 320 wool and 305 fertilizer sales. The replay records orders, not confirmed sale revenue.

Our agent hired 90 times (three hands per day), bought land once, and planted wheat, carrots, and strawberries. It built no animal structures and had no animal actions. The replay analyzer counts movement as about 58% of our unit actions and `PASS` as about 14%; these are diagnostic counts, not proof that every movement or pass was wasted.

## First local livestock experiment

A prototype policy was added behind JSON settings so the saved manual baseline stays unchanged. On 5 fixed local seeds against Kaggle Environments' `starter` agent, it averaged **31,756** reward with 8 sheep, 8 hands, and no hire cash threshold. On the project's fixed seed splits, the same trial averaged:

| Split | Baseline mean | Livestock trial mean | Games |
| --- | ---: | ---: | ---: |
| Training | 8,408 | 40,831 | 10 |
| Validation | 8,524 | 31,614 | 5 |
| Holdout | 8,128 | 27,814 | 5 |

This improves the measured result against `starter`, but `starter` is much weaker than the opponent in the uploaded replay. It does not establish that the trial can beat that opponent. The trial config is `experiments/animal_strategy_trial.json`; its default is not enabled in the competition entry.

## Replay-policy stress tests

The optimizer's first livestock search selected an 8-cow policy. Against the
fixed actions from the uploaded replay, it raised our mean money from **7,429**
to **48,481** over five untouched holdout seeds. It still lost all five games;
the mean margin improved from **-63,392** to **-28,566**. A separate three-seed
check gave the same direction (baseline **9,213**, candidate **27,582**), with
zero wins. The candidate and report are in
`experiments/results/run_20260925_043423_453081/`.

This is a meaningful response to the recorded livestock strategy, but not a
winning policy. The replay opponent reuses fixed actions and cannot react to
our new play, so it is a stress test rather than a substitute for live
opponents. The earlier search ranked candidates by our cash alone; subsequent
searches now rank by money margin, which matches the competition's win
condition.

The later margin-based search kept the same 8-cow configuration. On fresh
replay-policy seeds its baseline/candidate mean margins were:

| Split | Baseline | Candidate | Candidate wins |
| --- | ---: | ---: | ---: |
| Training, 5 games | -61,796 | -16,589 | 1/5 |
| Validation, 5 games | -60,772 | -14,737 | 1/5 |
| Holdout, 5 games | -67,694 | -12,636 | 1/5 |

The candidate also beat `starter` on 5/5 fresh seeds (mean money 46,135 vs
9,721 baseline) and the seeded random agent on 5/5 (53,828 vs 7,950). Those
opponents are much weaker than the uploaded replay strategy. Against the
replay it still loses 4/5 on holdout, so this is a stronger candidate, not a
solution to the high-performing player strategy. Full reports are in
`experiments/results/run_20260925_045434_453999/`.

A targeted sweep then compared 8, 12, and 16 animals for cows and sheep, both
with and without land expansion. Eight cows without expansion was best on
training and validation; more animals and buying land reduced the mean margin
on these seeds. Its fresh holdout mean margin was **-25,800**, compared with
**-63,878** for the manual baseline, but it still lost all five games. This
supports the 8-cow setting for now; it does not show that a larger herd is
always worse. Report: `experiments/results/animal_sweep_20260925_051618_401412/report.json`.

A follow-up sale-threshold sweep tested ratios 0.0, 0.25, 0.5, and 0.8 for
this cow policy. The 0.5 ratio ranked best on the fixed validation seeds
(-18,548 mean margin); the original 0.8 ratio scored -23,402. Two threshold
settings were carried to holdout: 0.5 scored -10,594 and 0.25 scored -9,886,
with one win each. That small holdout difference is not enough to prefer 0.25;
the config selected by validation is 0.5. Full report:
`experiments/results/sell_sweep_20260925_053640_957076/report.json`.

On another fresh comparison using the same seeds for both policies, the 0.5
ratio improved mean margin over 0.8 against the replay opponent (-15,095 vs
-21,555), `starter` (43,673 vs 42,679), and random (41,375 vs 39,872). It won
all five games against each simple opponent and one of five against the replay
actions. Results:
`experiments/results/sell_sweep_20260925_053640_957076/generalization.json`.

A further evolutionary pass found a training-stronger variant (land expansion,
12-unit market batches, and small crop-weight changes). Its training margin
was -12,563 vs -15,999 for the current config, and validation was -27,188 vs
-28,039. On the untouched holdout it fell to -31,745, while the current config
scored -28,381 and won one game. That is a possible overfit; I kept the current
8-cow, 0.5-ratio policy. The comparison is saved in
`experiments/results/run_20260925_054503_880006/current_vs_challenger.json`.

On the original replay seed, the current policy finished with 22,845 against
the replay actions' 54,437; the manual baseline finished with 8,006 against
85,296. The policy cut the gap substantially but still lost. This is a direct
same-seed stress test, not an adaptive rematch.

## Final-turn policy

The local interpreter calls the agent with observation `step=718` for the last
action in its 720-step episode (verified from the actual `env.steps` trace).
On that action the agent no longer buys seeds, land, animals, or hires. It
deposits saleable carried goods from units already beside the shed, then queues
sales for all saleable shed stock, including low-priced wheat. The documented
10-order limit is enough for the nine sale products. Units away from the shed
cannot reach it and deposit goods in a single action.

This final-turn adjustment is covered by unit tests and a full local episode.
The policy in `optimized_config.json` is now the competition-facing default;
it includes 8 cows and a 0.5 sell-price ratio. The manual baseline remains
unchanged in `baseline_config.json`. The optimized policy still loses most
replay-policy games, so this promotion means it is the best measured candidate
so far, not that it has solved the strong-opponent matchup.
