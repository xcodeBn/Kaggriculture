# Kaggle replays

Put every original replay JSON downloaded from Kaggle in this folder. Keep its
timestamped filename unchanged; each file can be analyzed or tested separately.
Keep locally simulated runs in timestamped folders directly under `replays/`
so source games and generated experiments stay distinguishable.

Replay data is ignored by Git; this directory's instructions are tracked. To
analyze an original replay:

```bash
python analysis/analyze_replay.py replays/kaggle_replays/<replay-file>.json
```

To test the current policy against the recorded actions from player 1:

```bash
python experiments/run_games.py --seeds 1000 \
  --opponent replay:replays/kaggle_replays/<replay-file>.json:1
```

The replay opponent repeats the recorded actions and does not react to the new
game state. The runner and analyzer accept explicit paths, so this folder can
move or the project layout can change without changing replay format or logic.
