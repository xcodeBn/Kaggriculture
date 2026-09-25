# Kaggle replay: first loss diagnosis

Source: local copy `replays/kaggle_game.json` (the raw replay is ignored by Git).

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

## Next test

The evolutionary search is now being run against `starter` with livestock settings available to the optimizer. Next, compare its selected policy against the observed sheep-and-cow strategy, then use additional Kaggle replays to check that it is not tailored to this single game.
