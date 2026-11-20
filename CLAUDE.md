# CLAUDE.md

RL + search system that solves the 3×3 Rubik's cube in as few moves as possible, measured
in the **half-turn metric (HTM)**, where `R`, `R'` and `R2` each count as 1 move.
The bar to beat is the two-phase solver (about 20–22 moves). Optimal solutions average
about 18 moves, and no position needs more than 20.

## Commands

```bash
uv sync                                  # install deps into .venv (Python 3.11, torch cu128)
uv run pytest -q                         # all tests (~3 s once solver tables exist)
uv run python -m rubik_rl.cube.bench     # env throughput benchmark
```

- Windows machine with an RTX 3080 (10 GB). Use `uv run ...`; there is no system Python.
- Use `RubikTwoPhase` (`import twophase.solver`), not `kociemba`. `kociemba` needs MSVC to
  build, and it isn't installed.
- The first `twophase.solver` import builds about 70 MB of tables into `./twophase/` in the
  current directory. That takes about 15 minutes, and the folder is git-ignored. Run from the
  repo root so the tables are reused.
- Light imports `twophase.face` and `twophase.cubie` don't load the tables. `from_string` and
  `solvability_error` use only those.

## Layout

- `rubik_rl/cube/cube.py`: numpy core. It holds the state, moves, pruning mask, scrambles,
  random walks, one-hot encoding and string conversion.
- `rubik_rl/cube/torch_cube.py`: `TorchCube`, the same operations on GPU. Use it for
  training data.
- `rubik_rl/cube/env.py`: `CubeEnv`, gym-style, with reward -1 per move.
- `rubik_rl/viz.py`: `show(state, moves)` opens the viewer through a `file://` URL.
- `viz/cube_viewer.html`: standalone three.js viewer, loaded from cdnjs. Open the file
  directly. **Don't start a local web server for it.**
- `rubik_rl/{models,train,search,eval}`: empty so far. They're for Phases 2–5.
- `tests/`: on the pytest `pythonpath`, so tests can import each other, e.g.
  `from test_viewer_conventions import run`.

## Conventions (shared by the env, the viewer and the two-phase solver; don't change them)

- Axes: x → R, y → U, z → F.
- State: `int8[54]` or `[N, 54]`. Entry i is the colour 0..5 (U R F D L B) of facelet i,
  in Kociemba facelet order: faces `U R F D L B`, 9 stickers each, row-major as seen looking
  at that face.
- Move index `m = 3*face + k`, with k = 0 for `X`, 1 for `X'`, 2 for `X2`. There are
  18 moves.
- Applying a move is a gather: `new = state[..., PERMS[m]]`. `PERMS` is generated from the
  geometric sticker model in `_build_perms`, which is the same model the viewer uses. Never
  hand-edit the permutation tables.
- `ALLOWED[prev, m]` is the pruning mask. It forbids the same face twice in a row and
  opposite faces in non-canonical order (`D U` is forbidden; `U D` is kept). Row `NO_PREV`
  (18) means no previous move. Scrambles use it, and search and policy masking should too.
- Network input is `one_hot(states)`: shape (N, 288), centres dropped.

If you touch move or facelet logic, keep `tests/test_viewer_conventions.py` and
`tests/test_cube.py` passing. They cross-check the env, the viewer's math and the
two-phase solver.

## Plan / status

1. ✅ Phase 0: setup. ✅ Phase 1: batched environment plus 3D viewer (tag `v1.0.0`).
2. Phase 2: eval module. Fixed test sets (random-state cubes plus per-depth sets), BFS
   optimal lengths for short scrambles, two-phase baseline. Report solve rate, average
   HTM length, % optimal, time and nodes expanded.
3. Phase 3: EfficientCube-style policy. Residual MLP trunk with an 18-way softmax, trained
   with cross-entropy on GPU random walks (label = `INVERSE_MOVE` of the last move), plus
   beam search.
4. Phase 4: add a value head (cost-to-go, DeepCubeA-style value iteration) and value-guided
   or weighted A* search.
5. Phase 5: 48-symmetry augmentation, bigger models and beams, policy+value combined search.

Design decision: a policy with greedy rollout isn't enough to get the fewest moves. Always
pair the network with search, and measure every change on the Phase 2 test sets.

## Git

- Remote: `github.com/george99887654/Rubik_RL`, branch `main`.
- Commit author is `george99887654 <george99887654@users.noreply.github.com>`, set in the
  repo config. Keep the noreply email.
- Commit or push only when asked.
