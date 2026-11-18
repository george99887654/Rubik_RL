"""Throughput benchmark for the cube environment.

    uv run python -m rubik_rl.cube.bench
"""
import time

import numpy as np
import torch

from . import cube as C
from .torch_cube import TorchCube


def timeit(fn, reps=5):
    fn()  # warm-up
    t = time.perf_counter()
    for _ in range(reps):
        fn()
    return (time.perf_counter() - t) / reps


def main():
    n, depth = 10_000, 30
    rng = np.random.default_rng(0)
    print(f"batch {n:,} cubes x {depth} moves\n")

    states, _ = C.scramble(n, 20, rng)
    moves = rng.integers(0, 18, size=n)
    dt = timeit(lambda: C.apply_move(states, moves), 20)
    print(f"numpy  apply_move     {n / dt / 1e6:8.2f} M moves/s")
    dt = timeit(lambda: C.expand(states), 20)
    print(f"numpy  expand (x18)   {18 * n / dt / 1e6:8.2f} M children/s")
    dt = timeit(lambda: C.random_walks(n, depth, rng))
    print(f"numpy  random_walks   {n * depth / dt / 1e6:8.2f} M states/s")

    devices = ["cpu"] + (["cuda"] if torch.cuda.is_available() else [])
    for dev in devices:
        tc = TorchCube(dev)
        big = 100_000 if dev == "cuda" else n
        s, _ = tc.random_walks(big, 20)
        s = s[:, -1]
        m = torch.randint(0, 18, (big,), device=dev)

        def sync():
            if dev == "cuda":
                torch.cuda.synchronize()

        dt = timeit(lambda: (tc.apply_move(s, m), sync()), 20)
        print(f"torch/{dev:4s} apply_move  {big / dt / 1e6:8.2f} M moves/s   (batch {big:,})")
        dt = timeit(lambda: (tc.expand(s), sync()), 20)
        print(f"torch/{dev:4s} expand      {18 * big / dt / 1e6:8.2f} M children/s")
        dt = timeit(lambda: (tc.random_walks(big, depth), sync()))
        print(f"torch/{dev:4s} random_walks{big * depth / dt / 1e6:8.2f} M states/s")


if __name__ == "__main__":
    main()
