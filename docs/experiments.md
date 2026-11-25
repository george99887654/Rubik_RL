# Experiments

Version labels use **rMAJOR.MINOR.PATCH**: MAJOR is a big performance change, MINOR a
training-scheme change, PATCH a bug fix. Overview page: `viz/versions.html`.

| Label | Run | Change |
|---|---|---|
| r1.0.0 | exact_v1 | first model (exact labels) |
| r2.0.0 | exact_v2_sym | 48-symmetry augmentation |
| r2.1.0 | exact_v3_sym40k | 4× longer training |
| r3.0.0 | self_v1 | user's self-training method (value beam, curriculum, self-consistency) |
| r3.1.0 | self_v2_w15 | width 15, curriculum allowed up to 15 |
| r3.2.0 | self_v3 | halving length schedule; promotion at 90% solved + value rounds to L |
| r3.2.1 | self_v4 | fix: BatchNorm two-pass bug |
| r3.3.0 | self_v5 | value criterion within 10% of L |
| r3.4.0 | self_v6 | per-depth value closeness thresholds |
| r3.5.0 | self_v7 | all scrambles at length n; groups by moves remaining |
| r3.6.0 | self_v8 | beam width 64 |
| r3.7.0 | self_v9 | width 32, ±10% proportional ranking noise |
| r3.8.0 | self_v10 | width 32, fixed ±0.3 ranking noise |
| r3.9.0 | self_v11 | width 128 |
| r3.10.0 | self_v12 | halving solve rule (n 50%, n−1 75%, n−2 87.5%, …) |
| r3.11.0 | self_v13 | value on solved-path cubes only (no self-consistency), from scratch |

All models were trained on `data/exact_L12`: 2 M cubes within 12 moves, exactly labelled,
about 20.8 k per distance. Settings: "any correct move counts" policy loss, natural batches,
value head trained but unused. The network is `CubeNet` (13.8 M parameters), trained for
10,000 steps × 8,192 cubes (about 4 minutes on an RTX 3080). Held-out set: 50 k cubes from
the same distribution. Search uses the policy only.

## exact_v1: baseline

- Training accuracy 100%, held-out 70.3%. Held-out accuracy peaked at step 1000 and then
  declined: **memorisation**.
- Held-out accuracy by distance: 1–4: 100%, 8: 57%, 12: 20%.

## exact_v2_sym: 48-symmetry augmentation (option B)

Same setup, but every training cube gets a random one of the 48 cube symmetries each time
it is used. The labels stay exact (`tests/test_symmetry.py`).

- Training accuracy 80.3%, held-out 79.6%: **no memorisation**. Held-out accuracy was still
  rising at the end of training, so the run was too short.
- Held-out accuracy by distance: 8: 78%, 10: 51%, 12: 31%.

## exact_v3_sym40k: symmetry augmentation, 4× longer (40,000 steps, about 18 min)

- Training accuracy 84.6%, held-out 83.0%. A small gap is starting to appear.
  Gains per 4,000 steps fell from about +2.4 to about +0.1 points by the end.
- Held-out accuracy by distance: 8: 87%, 10: 58%, 12: 35%.

## self_v1: the user's self-training method, from scratch, with no exact labels

Method (`rubik_rl/train/self_train.py`):
- **Search:** value-ranked beam, width 5, all allowed moves as candidates.
- **Training data:** cubes scrambled 1..n, solved with a step limit of n.
- **Labels:** from solutions found. Policy target = move taken; value target = moves
  remaining. Cubes not solved within n give no path labels.
- **Self-consistency:** the value head is also trained on searched and unsolved cubes toward
  1 + min V_frozen(child). This is Option 1. Without it, the curriculum stuck at n = 2,
  because V rated unseen cubes as *near*.
- **Curriculum:** n starts at 1 and rises when the solve rate at n stays ≥ 0.95 for 3 rounds.
  Random symmetry per sample.
- **Exact labels:** used only for monitoring.

Results after 90 minutes:
- The curriculum reached n = 9 in 18 minutes, then **plateaued at n = 9** for 72 minutes.
  The solve rate at n = 9 was about 0.91 against a promotion threshold of 0.95. A width-5
  beam must find a nearly optimal 9-move path, and that's hard.
- Training targets from found solutions were **100% optimal**, moves and values, at every n.
- Held-out monitor (exact labels): policy accuracy **79.5%** (v3: 83.0%), value error 0.72.
  By distance: 8: 80%, 10: 49%, 12: 30%. That's close to v3, without ever using an exact
  label or seeing a cube beyond 9 moves.

## self_v2_w15: the same method with width 15 and the curriculum allowed up to n = 15

- The curriculum went 1 → 9 in 4.3 minutes (self_v1 took 18) and reached n = 10 at 24 minutes.
  It then **plateaued at n = 10 for 2.6 hours**: the solve rate at n hovered around 0.94
  against the 0.95 threshold. So it never got to n = 13–15.
- Training targets were 100% optimal throughout.
- Held-out monitor: policy accuracy **80.7%** (self_v1: 79.5%, v3: 83.0%), value error 0.68.
  By distance: 8: 83%, 10: 52%, 12: 30%.
- With value beam 1024 it was **100% optimal on all 50 depth_12 cubes** (12.00 moves).

## Search results

| Solver | depth_10 | depth_12 | random_1000 | Time per cube (random) |
|---|---|---|---|---|
| greedy, v1 | 3% solved | 1% | 1% | 0.27 s |
| greedy, v2_sym | 23% | 7% | 1% | 0.29 s |
| beam 1024, v1 | 98%, 12.08 moves | 96%, 21.6 | 88%, **31.95** | 0.25 s |
| beam 1024, v2_sym | 100%, 10.10 (96% optimal) | 100%, 14.02 (72% optimal) | **100%, 23.62** | 0.18 s |
| greedy, v3_sym40k | 43% | 10% | 5% | 0.29 s |
| beam 1024, v3_sym40k | 100%, 9.96 (100% optimal) | 100%, 13.04 (78% optimal) | **100%, 22.54** | 0.18 s |
| value beam 5, self_v1 | 86%, 10.67 | 24%, 13.92 | 0% | 0.40 s |
| value beam 1024, self_v1 | 100%, 9.96 (100% optimal) | 100%, **12.12 (98% optimal)** | 86%, **21.30** | 0.50 s |
| policy beam 1024, self_v1 | 100%, 10.02 | 100%, 15.66 | 100%, 26.14 | 0.26 s |
| value beam 5, v3 | 84%, 9.95 | 28%, 12.36 | 2% | 0.38 s |
| value beam 1024, v3 | 100%, 9.96 | 100%, 12.14 (98% optimal) | 94%, 21.83 | 0.48 s |
| value beam 15, self_v2_w15 | 100%, 10.02 | 60%, 14.93 | 10% | 0.31 s |
| value beam 15, self_v1 / v3 | 98% / 94% | 52% / 58% | – | – |
| value beam 1024, self_v2_w15 | 100%, 9.96 (100% optimal) | 100%, **12.00 (100% optimal)** | 86%, 21.60 | 0.40 s |
| two-phase, fast | 100%, 15.43 | 100%, 18.16 | 100%, **19.77** | 0.16 s |
| two-phase, 5 s | 100%, 9.93 | 100%, 12.00 | 100%, 18.75 | 5 s |

The beam-search rows used 50 cubes per set; the greedy rows used 100.

## r3.2.0 → r3.3.0 (self_v3 / self_v4 / self_v5)

These three runs used the halving length schedule (n: ½, n−1: ¼, …), promotion at 90%
solved plus a 90% value check, width 15, from scratch, 180 minutes each.

- **r3.2.0 (self_v3):** value check "V rounds to L". Stuck at n = 5–6. 1-move cubes were
  valued 1.82 because of a **BatchNorm bug**: path and self-consistency batches went through
  the network in separate forward passes, so the running statistics didn't match training.
- **r3.2.1 (self_v4):** fix, with both batches in one forward pass. Values at distances 1–4
  became essentially exact. Stuck at n = 7 from 28.5 min: the value spread at distance 7
  (±0.48) is too wide for 90% to round correctly. Value beam 1024: depth_12 12.36 (96%
  optimal); random 80% solved, 22.38 moves.
- **r3.3.0 (self_v5):** value check "within 10% of L". n = 7 at 19.1 min, **n = 8 at 145.7
  min**. Held-out policy accuracy 78.8%, value error 0.58. Value beam 1024: depth_12 12.16
  (98% optimal); random 78% solved, 21.62 moves.

r3.0.0 and r3.1.0 used the same two-pass code, so they also had the BatchNorm bug.

## r3.4.0 (self_v6) and r3.5.0 (self_v7)

- **r3.4.0:** promotion by per-depth value closeness, mean of 1 − |V − L| / L by scramble
  length: n ≥ 0.60, n−1 ≥ 0.80, n−2 ≥ 0.90, ≤ n−3 ≥ 0.95, plus 90% solved. n = 8 at 3.3 min,
  n = 10 at 22.5 min, n = 11 at 140 min. Stopped at 148 min. Monitor 81.3%, value error 0.47.
  Value beam 1024: depth_12 100% optimal, random 86% solved at 21.23 moves.
  - It was blocked at n = 10 by the ≤ n−3 group, because values were overestimated at
    depths 6–8.
  - Measured: the self-consistency targets were *less* biased than the network at 6–8, so
    they weren't the cause.
  - About 21% of self-consistency cubes are deeper than n (search candidates off the path),
    and their targets are too low (−0.7 at 11, −1.4 at 12).
- **r3.5.0:** all 4,096 scrambles at length n; promotion groups are path cubes by remaining
  moves. n = 11 at 56.3 min (2.5× faster than r3.4.0). All value groups passed; the solve rate
  at n = 11 (0.70 → 0.81) was the only gate. Monitor **81.9%**, value error **0.46**, the best
  of the self-trained runs. Value beam 1024: depth_12 100% optimal; depth_14 96% solved at
  15.94; **random 86% solved at 20.65 moves**, the shortest so far (two-phase fast: 19.77).
- **r3.5.0 continued (self_v7b):** from its checkpoint at n = 11. Solve rate flat at about
  0.82 for 100 rounds; the process died at 31 min. Monitor 82.4%, value error 0.445.

## r3.6.0 – r3.9.0: wider beams and ranking noise

Every run from here starts from the previous run's weights; the buffers start empty.

- **r3.6.0 (self_v8), width 64:** solve rate at n = 11 jumped to 0.954, against 0.82 at
  width 15. So the search, not the value head, was dropping the right path. Stopped by
  request after 5.8 min, with no checkpoint.
- **r3.7.0 (self_v9), width 32, ±10% proportional noise:** ranking by V·(1 + u). The solve
  rate fell to 0.70–0.72, because at V ≈ 10 the noise is about ±1 move, as large as the
  signal. Stopped at 31.9 min. Monitor 82.4%, value error 0.462.
- **r3.8.0 (self_v10), width 32, fixed ±0.3 noise:** ranking by V + u. **n = 12 at 11.2
  min**, the first self-trained run to reach it. Then the solve rate at n = 12 stayed at
  0.69–0.72 for the rest of the 180 min; all value groups passed. Monitor 83.7%, value error
  0.455. Value beam 1024: depth_14 100% solved at 15.48; random 92% solved at 20.43 moves.
- **r3.9.0 (self_v11), width 128:** the solve rate at n = 12 jumped at once to 0.89 and
  hovered at 0.89–0.90 until it got three rounds in a row at 0.90: **n = 13 at 140.6 min**.
  At n = 13 the solve rate stayed at 0.66–0.69. About 55 s per round (width 32: about 17 s).
  Monitor **83.9%**, value error **0.440**. Value beam 1024: depth_14 100% solved at 15.44;
  **random 92% solved at 20.17 moves**, the shortest so far (two-phase fast: 19.77; 50 cubes
  only, so the gap to r3.8.0 is within noise).

Lesson: a wider beam lifts the solve rate once, then it stays flat at each n. Each step up in
n costs about 0.22 in solve rate. Training only on successful paths doesn't improve the cubes
that fail, because they never produce a training example.

## r3.10.0 (self_v12): halving solve rule

Promotion needs a solve rate of at least 1 − 0.5^(k+1) at every depth n − k: n 50%, n−1 75%,
n−2 87.5%, n−3 93.75%, and so on, plus the unchanged value rule, for 3 rounds in a row.
Training scrambles are all at length n, so the lower depths are checked on 1,024 extra
scrambles each. Those are used only for measurement, and only in rounds where depth n and the
value check already pass. Continued from r3.9.0 with width 128 and ±0.3 noise.

- **n = 14 at 6.7 min.** At n = 13 the solve rate (0.67) already cleared 50%, and n−1 was
  0.90. At n = 14 the solve rate was 0.41 and rose to 0.45 by 180 min, still short of 0.50.
- **Value drift:** the value head drifted upward on cubes off the solved paths. Mean V − d at
  d = 9 went from +0.76 (r3.9.0) to +1.25; held-out value error rose from 0.44 to 0.67.
  - Path targets stayed exact: every checked target with 12 or fewer moves remaining had an
    exact value and an optimal move.
  - So the drift comes from the self-consistency targets on off-path and failed cubes, which
    are anchored only at the solved state.
- **Results:** held-out policy accuracy 84.1%, the best so far. Value beam 1024 on 200 random
  cubes: **99.5% solved at 20.28 moves**, against 96.0% at 20.47 for r3.9.0 on the same 200.
  depth_14: 100% solved at 15.60.

## r3.11.0 (self_v13): path-only value, from scratch

The value head is trained only on cubes along paths solved within the scramble length, with
target = moves remaining on that path (`--bellman-weight 0`). No failed or off-path cube is
trained. Otherwise as r3.10.0 (width 128, ±0.3 noise, halving solve rule), but from scratch.

- **Curriculum:** n = 9 at 27.7 min, n = 10 at 76.8 min, n = 11 at 158 min (r3.5.0 at width
  15 with self-consistency: n = 11 at 56 min).
  - The solve rate was never the gate (0.80–0.97).
  - The ≤ n−3 value group was: 0.945–0.949 against 0.95 needed.
- **Value bias (V − d):** deep cubes look near (d = 10: −0.49, d = 12: −2.05); distances 6–8
  are too high (+0.38 to +0.56). The upward drift of r3.10.0 is gone.
- **Results:** monitor 74.9%, value error 0.65. Value beam 1024: depth_12 92% optimal (12.42);
  depth_14 86% solved at 18.56; random (200 cubes) 75.5% solved at 23.13 moves.
- **Continued (self_v13b):** 180 more minutes from n = 11. n = 12 at 127.5 min; the ≤ n−3
  group stayed at 0.934–0.947, and promoted only on a lucky streak. After 6 h in total:
  monitor 77.2%, value error 0.55; value beam 1024: depth_12 96% optimal (12.28), depth_14
  92% solved (17.39), random (200 cubes) 87.5% solved at 22.05 moves.
