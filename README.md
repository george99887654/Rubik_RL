# Rubik RL

Train a neural network + search to solve the 3×3 Rubik's cube in as few moves as possible
(half-turn metric, HTM: `R`, `R'`, `R2` each count as 1 move).

## Setup

Requires [uv](https://docs.astral.sh/uv/). PyTorch is installed from the CUDA 12.8 wheel index.

```bash
uv sync          # creates .venv with Python 3.11, torch (CUDA), numpy, RubikTwoPhase (pure-Python Kociemba solver), pytest
uv run pytest    # tests (first run builds two-phase solver tables, ~15 min, one time)
```

## Layout

```
rubik_rl/
  cube/     cube environment: state representation, moves, scrambles   (Phase 1)
  models/   neural networks (policy / cost-to-go)                      (Phase 3–4)
  train/    training loops                                             (Phase 3–4)
  search/   beam search, batch weighted A*                             (Phase 3–4)
  eval/     baselines (two-phase solver, BFS) and benchmark scripts            (Phase 2)
tests/      pytest tests
viz/        cube_viewer.html — interactive 3D state viewer
configs/    experiment configs (YAML)
checkpoints/, data/   generated artifacts (git-ignored)
```

## Conventions

- Axes: x → R, y → U, z → F.
- **State string**: the 54-character Kociemba facelet string. Faces are in the order
  `U R F D L B`, with 9 stickers per face, row-major as seen looking straight at that face.
  The solved state is `UUUUUUUUURRRRRRRRRFFFFFFFFFDDDDDDDDDLLLLLLLLLBBBBBBBBB`.
- Colours: U white, R red, F green, D yellow, L orange, B blue.
- **State array**: `int8[54]` (or `[N, 54]` for a batch), where entry i is the colour 0..5 of
  facelet i, in the same order as the string.
- **Moves** (HTM, 18): index `m = 3*face + k`, where face is in the order `U R F D L B` and
  k is 0 = `X`, 1 = `X'`, 2 = `X2`.

## Cube environment (`rubik_rl.cube`)

```python
from rubik_rl import cube as C

s = C.apply_moves(C.solved(), C.parse_moves("R U R' U'"))
C.is_solved(s); C.to_string(s); C.from_string("...")    # from_string checks the state is legal

states, moves = C.scramble(1000, 20)       # batch of 1000 scrambled cubes (redundant moves pruned)
walks, wmoves = C.random_walks(1000, 30)   # every state along each walk, for training data
kids = C.expand(states)                    # (N, 18, 54): all children of each state
x = C.one_hot(states)                      # (N, 288) network input (centres dropped)

from rubik_rl.cube.torch_cube import TorchCube   # the same operations on the GPU
from rubik_rl.cube.env import CubeEnv             # gym-style reset/step, reward -1 per move
from rubik_rl.viz import show; show(s, "U R U' R'")   # open a state in the 3D viewer
```

`C.ALLOWED[prev, m]` is the move-pruning mask. It forbids turning the same face twice in a row
and forbids opposite faces in non-canonical order (`D U`; only `U D` is kept). Scrambles and
search both use it.

Throughput on an RTX 3080 (`uv run python -m rubik_rl.cube.bench`):

| op | numpy | torch (CUDA) |
|---|---|---|
| apply one move | 3.6 M/s | 209 M/s |
| expand to all 18 children | 26 M/s | 1,130 M/s |
| random-walk training states | 3.2 M/s | 170 M/s |

## Visualizer

Open `viz/cube_viewer.html` in a browser (it loads three.js from a CDN). You can:
- turn faces with the buttons or the keyboard (`Shift` gives the prime move);
- run random scrambles;
- play an algorithm or its inverse;
- load or copy a facelet string;
- see the cube as a 3D model and as a 2D net.

It also reads a state and moves from the URL hash, so Python code can open it directly:

```
viz/cube_viewer.html#state=<54 chars>&moves=R_U_R'_U'
```
