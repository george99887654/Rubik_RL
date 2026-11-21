# Experiments

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

## Search results

| Solver | depth_10 | depth_12 | random_1000 | Time per cube (random) |
|---|---|---|---|---|
| greedy, v1 | 3% solved | 1% | 1% | 0.27 s |
| greedy, v2_sym | 23% | 7% | 1% | 0.29 s |
| beam 1024, v1 | 98%, 12.08 moves | 96%, 21.6 | 88%, **31.95** | 0.25 s |
| beam 1024, v2_sym | 100%, 10.10 (96% optimal) | 100%, 14.02 (72% optimal) | **100%, 23.62** | 0.18 s |
| greedy, v3_sym40k | 43% | 10% | 5% | 0.29 s |
| beam 1024, v3_sym40k | 100%, 9.96 (100% optimal) | 100%, 13.04 (78% optimal) | **100%, 22.54** | 0.18 s |
| two-phase, fast | 100%, 15.43 | 100%, 18.16 | 100%, **19.77** | 0.16 s |
| two-phase, 5 s | 100%, 9.93 | 100%, 12.00 | 100%, 18.75 | 5 s |

The beam-search rows used 50 cubes per set; the greedy rows used 100.
