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

## Replay-policy stress tests: prior results withdrawn

The original Kaggle outcomes above are valid. The offline experiments that
played against fixed Kaggle replay actions were not: replay actions were fed
one turn late. A direct same-seed replay in the local simulator established
that an action for observation step `t` belongs to replay frame `t + 1`. The
replay opponent has been corrected to use that offset.

All previous replay-opponent optimization scores and conclusions in these
reports are withdrawn, including the livestock, sale-threshold, and
current-vs-challenger comparisons. Their saved output folders are historical
artifacts, not valid evidence. Rerun them with the corrected opponent before
using their scores to select a policy. Results against `random` or `starter`
are not affected by this replay alignment bug.

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
unchanged in `baseline_config.json`. Because the replay-policy experiments used
the wrong action-frame offset, they do not establish that this policy is
stronger against the uploaded replay opponents. Keep this as the current config,
but rerun those comparisons before claiming a replay-matchup improvement.
