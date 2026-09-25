# Latest Kaggle match review

Replay files checked: 16. Unique episodes: 15. Self-play episodes (excluded from opponent training): 2. Usable games with one Hassan Bazzoun-dev seat: 13.

Against named opponents: **7 wins, 6 losses**; mean cash $41,096, median $44,824. This is a descriptive set of competition episodes, not a controlled comparison: the seasons have different seeds and opponents.

| Episode | Opponent | Our seat | Our cash | Opponent cash | Result | Opponent animals | Opponent crop actions |
|---|---|---:|---:|---:|---|---|---:|
| 113076092 | GusNicho | 0 | $8,111 | $144,750 | loss | {'SHEEP': 16, 'COW': 2} | 70 |
| 113234468 | Alizen James | 0 | $7,694 | $51,886 | loss | {'SHEEP': 4, 'COW': 4} | 93 |
| 113235655 | BSCode | 0 | $44,824 | $60,712 | loss | {'GOOSE': 8} | 350 |
| 113236825 | Dharmeswar Basumatary | 1 | $18,843 | $25,488 | loss | {'GOOSE': 10, 'COW': 3, 'SHEEP': 5} | 96 |
| 113238087 | Odilon Yehouenou | 1 | $70,525 | $6,722 | win | {'GOOSE': 1} | 65 |
| 113239264 | Yudha Eka Saputra | 1 | $62,548 | $27,987 | win | none | 86 |
| 113240422 | Jenny Luo | 1 | $73,765 | $58,185 | win | {'COW': 8, 'GOOSE': 2, 'SHEEP': 4} | 79 |
| 113241592 | Kohei | 0 | $3,306 | $23,929 | loss | {'COW': 11, 'SHEEP': 4} | 53 |
| 113242748 | Kawa_pt | 1 | $72,953 | $51,257 | win | {'GOOSE': 1, 'SHEEP': 1, 'COW': 1} | 144 |
| 113243959 | Ivan Berrutto | 0 | $81,577 | $30,036 | win | none | 126 |
| 113245140 | Aleksander Sachuk | 0 | $9,346 | $130,044 | loss | {'SHEEP': 12, 'COW': 3} | 88 |
| 113246321 | CarlitrosParra | 0 | $33,502 | $31,824 | win | {'SHEEP': 2, 'COW': 4} | 134 |
| 113247477 | Comet19 | 0 | $47,253 | $16,968 | win | {'COW': 14, 'GOOSE': 2} | 42 |

## Patterns

- Our side repeatedly shows the same 8-cow production footprint and typically 240 hires (eight hires per day). That makes the current policy behavior identifiable across seats.
- Outcomes range from $3,306 to $81,577 for our side. We beat several crop-focused or small-herd agents, but lost to multiple mixed-herd and high-throughput opponents.
- The largest losses include a sheep/cow operation ($144,750), a mixed sheep/cow crop farm ($51,886), a high-volume goose/crop farm ($60,712), a larger multi-species farm ($23,929 vs our $3,306), and a sheep/cow-heavy farm ($130,044). These games suggest that animal count, workforce, and sales capacity deserve more testing; they do not prove which parameter caused each loss.
- Two downloaded files are self-play episodes. The duplicate `kaggle_game.json` and `113076092.json` have the same episode ID, so the duplicate was counted once.

## Training opponent selection

The next small evolutionary run uses six distinct recorded opponents plus the local `random` and `starter` agents: GusNicho, Alizen James, BSCode, Jenny Luo, Kohei, and Aleksander Sachuk. Recorded replay policies repeat fixed actions in a fresh game and do not adapt. This pool broadens stress coverage but is not adaptive self-play.

Full machine-readable per-game summary: `analysis/latest_kaggle_games.json`.

## Offline evolutionary check against replay opponents

Run: `experiments/results/run_20260925_135358_681401/`. The pool included
`random`, `starter`, and recorded actions from GusNicho, Alizen James, BSCode,
Kohei, and Aleksander Sachuk. Each candidate used the same two seeds per
split: training 11000–11001, validation 12000–12001, and holdout 13000–13001.
The short seed sets make this an exploratory check, not a stable estimate.

| Policy | Training mean margin (wins/games) | Validation mean margin (wins/games) | Holdout mean margin (wins/games) |
|---|---:|---:|---:|
| Manual baseline | -$32,303 (8/14) | -$27,174 (8/14) | -$30,067 (8/14) |
| Existing optimized config | +$26,524 (10/14) | +$6,109 (8/14) | +$12,184 (8/14) |

The optimizer selected the existing `optimized_config.json`; it did not find a
new config to promote. That config was identical to the starting policy and
remains unchanged. On holdout its mean margins by recorded opponent were
GusNicho -$21,167, Alizen James -$50,032, BSCode +$43,651, Kohei +$65,229,
and Aleksander Sachuk -$27,358. It also beat `random` and `starter` in both
holdout seeds. This average hides three replay-opponent losses.

The fixed replay tests are counterfactual: they repeat one player's recorded
actions under new seeds, where market and town conditions differ and the
replayed policy cannot react. For example, the actual BSCode match was a loss
for us, while the replay-tape holdout scored positive margin. Use those tests
as stress checks, not predictions of live rematches. Full per-opponent metrics
are in the run's `validation.json`; no improvement over the existing optimized
config was measured in this search.
