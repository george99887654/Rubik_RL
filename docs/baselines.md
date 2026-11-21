# Phase 2 baselines

All numbers come from `uv run python -m rubik_rl.eval.evaluate --compare`, run on the fixed
test sets in `testsets/`. Lengths are in HTM. "% optimal" and "gap" are measured against
exact distances from `OptimalSolver`, which is exact up to 12 moves.

## Headline numbers (`random_1000`, effectively random cubes)

| Solver | Cubes | Solved | Mean length | Time per cube |
|---|---|---|---|---|
| two-phase, stop at the first solution of 20 moves or fewer | 1000 | 100% | **19.77** | 0.16 s |
| two-phase, best found in 5 s | 100 | 100% | **18.75** | 5 s |
| random moves (2000 tries × 30 moves) | – | 0% beyond depth 4 | – | – |

Superflip (exactly 20 moves): two-phase returns 21 moves, both with the fast setting and
with 5 s.

## By scramble depth (two-phase, fast setting)

| Depth | 1 | 2 | 4 | 6 | 8 | 10 | 12 | 14 | 15 |
|---|---|---|---|---|---|---|---|---|---|
| Mean length | 2.79 | 4.43 | 7.61 | 10.28 | 13.07 | 15.43 | 18.16 | 19.17 | 19.37 |
| % optimal | 74.5 | 64.5 | 43.5 | 33.0 | 23.5 | 22.0 | 9.0 | 0 | 0 |

The fast setting stops at *any* solution of 20 moves or fewer, so even very easy cubes get
long solutions. With 5 s it was optimal on every sampled cube at depths 8, 10 and 12.

## Label study (`runs/label_study.json`): 200 random walks, exact distances

| Step t | 1–8 | 9 | 10 | 11 | 12 |
|---|---|---|---|---|---|
| Fraction where true distance equals t | 100% | 98.5% | 98.0% | 97.0% | 97.0% |
| "Undo the last move" lies on a shortest path | 100% | 98.5% | 99.5% | 99.0% | 100% |
| Mean number of optimal moves from the state | 1.0–1.15 | 1.15 | – | – | – |

## What this tells us

1. **The bar.** A learned solver is competitive if it beats about 19.8 moves quickly or about
   18.75 with a few seconds of search, on `random_1000`. Optimal solutions for random cubes
   average roughly 18, so the remaining room is about 1 move per cube on average.
2. **Up to 12 moves, random-walk data is almost perfectly clean.** With non-redundant moves,
   a walk of t ≤ 12 steps is nearly always a shortest path. "Undo the last move" is optimal
   99–100% of the time, and there is usually exactly one optimal move.
3. **So any label noise must live deeper, between 13 and 20 moves.** The cube's maximum
   distance is 20, so a walk longer than that *must* double back, and labels there can't
   always be optimal. That depth range, which is where random cubes sit, is exactly what
   our exact solver can't see. Measuring it needs a different tool, for example long
   two-phase searches as upper bounds, or a learned estimate checked against them.
4. **Speed matters.** Two-phase solves a cube in about 0.16 s. A learned solver should
   report time per cube as well as length.
