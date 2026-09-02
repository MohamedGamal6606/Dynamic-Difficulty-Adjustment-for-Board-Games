# Dynamic Difficulty Adjustment for Board Games

A chess application that adjusts the engine's playing strength in real time to match the player's skill, using Stockfish evaluations combined with a machine learning win-probability model.

Bachelor thesis project — Media Engineering and Technology Faculty, German University in Cairo.

- **Author:** Mohamed Gamal
- **Supervisor:** Dr. Yomna M.I. Hassan
- **Submitted:** 29 May, 2025

---

## Motivation

Traditional chess programs ship with fixed difficulty levels. Beginners get crushed and quit; stronger players get bored. Dynamic Difficulty Adjustment (DDA) solves this by watching how the player is actually doing — move accuracy, centipawn loss, position evaluation — and continuously retuning the engine so the game stays winnable but not trivial.

This project extends the rule-based DDA work of Bontchev and Ivanov (2024) by replacing predefined heuristics with a model trained on ~100,000 real games.

## How it works

Every move the player makes goes through the same loop:

1. **Evaluate.** Stockfish scores the position before and after the move (`{'time': 0.01, 'depth': 20}`).
2. **Classify.** The evaluation delta (ΔE, in centipawns) sorts the move into a quality bucket:

   | Classification | Evaluation Delta |
   |---|---|
   | Brilliant | ΔE ≥ 200 |
   | Best | 70 < ΔE < 200 |
   | Good | −70 ≤ ΔE ≤ 70 |
   | Inaccuracy | −200 < ΔE < −70 |
   | Blunder | ΔE ≤ −200 |

3. **Predict.** A Random Forest classifier estimates the player's win probability from the position and their behavioural metrics.
4. **Adjust.** The skill algorithm nudges Stockfish's skill level (bounded to `[1, 20]`) toward a target win probability of **70%**, with damping to stop the difficulty oscillating between moves.

### Win probability

```
P(win) = P_base + A_score + A_accuracy + A_target
```

Where `P_base` comes from the model, `A_score` is a sigmoid mapping of the centipawn evaluation, `A_accuracy` reflects recent move quality, and `A_target` biases toward the 70% target. (50% was tried first and tested poorly with users — 70% is what shipped.)

### Skill adjustment

The total adjustment combines win-probability gap, recent performance trend, move-quality ratio, average accuracy, and current board score, then scales by the current skill level before being applied with stability controls.

## Features

- Full chess rules via `python-chess` — legal move generation, FEN/PGN parsing, rule enforcement
- Tkinter GUI with click-to-move, highlighted previous moves (orange) and legal destinations (green)
- Live score bar, move timer, accuracy percentage, and win-probability readout
- Move classification panel with trend indicators (↑, →, ↓)
- Contextual audio for moves, captures, checks, and castling
- Restart and help controls

## Dataset and model

Training data came from the public Lichess database (January 2025), partitioned into 100,000-game chunks for processing. Each move was re-evaluated with Stockfish 16 to derive quality labels, producing two records per game (one per player) for 200,000 entries total.

| Metric | Value |
|---|---|
| Games processed | 100,000 |
| Average rating (White / Black) | 1668.89 / 1668.71 |
| White wins / Black wins / Draws | 49.87% / 46.24% / 3.89% |
| Average moves per game | 66.29 |

Time controls skew heavily toward fast formats — blitz (58,462) and bullet (40,202) make up over 98% of the set, with classical (1,243) and correspondence (93) making up the rest. The model is consequently stronger at tactical recognition than deep positional judgment.

Trained with an 80–20 split:

| Model | Accuracy | Precision | Recall | F1 |
|---|---|---|---|---|
| This project | 0.70 | 0.69 | 0.69 | 0.69 |
| Baseline (Dreżewski & Wątor, 2021) | 0.689 | — | — | — |

## Evaluation

Fifteen participants across a range of skill levels (40% unrated, 20% beginner, 40% amateur) played the system and completed a survey adapted from Bontchev and Ivanov (2024).

- Every participant eventually won, averaging **2.29 games** to a first win
- **66.7%** rated the difficulty level satisfactory; 6.7% found it too hard, 26.7% too easy
- **86.7%** liked the game overall
- **66.7%** reported better focus, more fun, and improved chess understanding
- **46.7%** explicitly noticed the engine adapting its strength to theirs

Mean win-game accuracy by self-reported level: beginners 46.20%, amateurs 68.73%, unrated 57.96%.

## Known limitations

- Skewed toward blitz and bullet, so positional depth suffers
- Accuracy drops at the extreme ends of the rating spectrum
- Initial difficulty can start too high before the system calibrates
- Desktop only — no mobile or web build

## Future work

Broader dataset coverage of classical time controls; player-specific models that learn individual patterns; coaching features that target detected weaknesses; voice interaction; cross-platform builds; and longitudinal studies of skill development over time.

## References

Key sources — the full bibliography is in the thesis.

1. B. Bontchev and H. Ivanov, "Dynamic adaptation of difficulty in chess," *2024 IEEE 12th International Conference on Intelligent Systems (IS)*, 2024.
2. L. Ilici, J. Wang, O. Missura, and T. Gärtner, "Dynamic difficulty for checkers and Chinese chess," *IEEE CIG*, pp. 55–62, 2012.
3. R. Hunicke and V. Chapman, "AI for dynamic difficulty adjustment in games," *AAAI Workshop on Challenges in Game AI*, 2004.
4. M. Guid and I. Bratko, "Search-based estimation of problem difficulty for humans," *AIED 2013*.
5. R. Dreżewski and G. Wątor, "Chess as sequential data in a chess match outcome prediction using deep learning with various chessboard representations," *Procedia Computer Science*, vol. 192, pp. 1760–1769, 2021.

## Acknowledgments

Thanks to Dr. Yomna Hassan for her supervision and support throughout this project, and to the fifteen participants who tested the system.
