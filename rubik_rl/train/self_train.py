"""Self-training from scratch with a value-ranked beam search (the user's method).

    uv run python -m rubik_rl.train.self_train --name self_v1 --n-max 11 --minutes 60

Each round:
  1. scramble `cubes_per_round` cubes, all with exactly n moves (n = current curriculum
     bound); shorter distances are covered by the cubes along each solved path.
     (length_mode "halving" instead uses n: 1/2, n-1: 1/4, ...)
  2. solve them all with ValueBeam(width), step limit n (optionally with a random
     +-search_noise fluctuation of the ranking values, for exploration)
  3. no solution within n moves -> that cube gives no training data
     solution found -> for each cube on the path: policy target = the move taken there,
                                                  value target  = moves remaining on the path
  4. add these to a replay buffer; take `train_steps` gradient steps on buffer samples
     (random symmetry per sample)
  4b. self-consistency (Option 1): the start cubes and a sample of every candidate the
     search evaluated go into a second buffer; the value head is also trained toward
         y(s) = 0 if s is solved, else 1 + min over children of V_frozen(child)  (floored at 0)
     where V_frozen is a copy of the network refreshed every `target_sync` steps. This
     teaches V that cubes it has never seen are far, not near.
  5. promotion: raise n by 1 (until n_max) when, for `promote_rounds` rounds in a row,
       - the solve rate on cubes scrambled with exactly n moves is >= promote_at
         (solve_rule "halving": >= 1 - 0.5**(k+1) at every depth n-k, i.e. n: 50%, n-1: 75%,
         n-2: 87.5%, ...; depths below n are checked on extra scrambles after training), and
       - the value closeness score, mean of 1 - |V - r| / r over cubes along solved paths
         (r = moves remaining on the found path), is per group of r at least
         r = n: 0.60, n-1: 0.80, n-2: 0.90, <= n-3: 0.95 (value_need). Empty groups are skipped.
     Both are measured with the current network before that round's training.

Exact labels (data/exact_L12) are used ONLY for monitoring, never for training:
policy top-1 accuracy and value error per distance, and the share of training targets that
are actually optimal moves.
"""
from __future__ import annotations

import argparse
import copy
import json
import time
from dataclasses import asdict, dataclass, field
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F

from ..cube import cube as C
from ..models import CubeNet, CubeNetConfig
from ..search.value_beam import ValueBeam
from .train import Encoder, SymAugment, validate

ROOT = Path(__file__).resolve().parents[2]


@dataclass
class SelfTrainConfig:
    name: str = "self_v1"
    width: int = 5
    n_start: int = 1
    n_max: int = 11
    cubes_per_round: int = 4096
    train_steps: int = 20
    batch: int = 4096
    buffer_size: int = 2_000_000
    lr: float = 1e-3
    weight_decay: float = 1e-5
    value_weight: float = 1.0
    sym_aug: int = 1
    bellman_weight: float = 1.0
    collect_per_round: int = 16384
    bellman_buffer: int = 2_000_000
    target_sync: int = 200
    promote_at: float = 0.90          # solve rate on length-n cubes ("fixed" solve_rule)
    # "fixed": solve rate at n >= promote_at.
    # "halving": solve rate at n-k >= 1 - 0.5**(k+1) for every k = 0..n-1
    #            (n: 50%, n-1: 75%, n-2: 87.5%, n-3: 93.75%, ...). The lower depths are measured
    #            on `check_cubes` extra scrambles per depth (evaluation only, not trained on),
    #            only in rounds where depth n and the value check already pass.
    solve_rule: str = "fixed"
    check_cubes: int = 1024
    length_mode: str = "frontier"     # "frontier": all cubes scrambled with n moves; "halving": n:1/2, n-1:1/4, ...
    # value closeness needed per group [n, n-1, n-2, <= n-3] of path cubes by REMAINING moves;
    # score = mean(1 - |V - r| / r)
    value_need: tuple = (0.60, 0.80, 0.90, 0.95)
    promote_rounds: int = 3
    minutes: float = 60.0
    max_rounds: int = 1_000_000
    eval_every: int = 25
    monitor_data: str = "data/exact_L12/shard_007.npz"
    monitor_size: int = 50_000
    seed: int = 0
    search_noise: float = 0.0         # training searches rank by V * (1 + u), u ~ U[-noise, +noise]
    search_noise_abs: float = 0.0     # training searches rank by V + u, u ~ U[-noise_abs, +noise_abs]
    init_from: str = ""               # checkpoint to continue from (weights only); "" = from scratch
    model: CubeNetConfig = field(default_factory=CubeNetConfig)


class Replay:
    def __init__(self, cap, dev):
        self.s = torch.zeros((cap, C.N_STICKERS), dtype=torch.uint8, device=dev)
        self.m = torch.zeros(cap, dtype=torch.long, device=dev)
        self.r = torch.zeros(cap, dtype=torch.long, device=dev)
        self.cap, self.n, self.pos = cap, 0, 0

    def add(self, s, m, r):
        k = len(s)
        idx = (torch.arange(k, device=s.device) + self.pos) % self.cap
        self.s[idx], self.m[idx], self.r[idx] = s, m, r
        self.pos = (self.pos + k) % self.cap
        self.n = min(self.cap, self.n + k)

    def sample(self, b, g):
        i = torch.randint(0, self.n, (b,), device=self.s.device, generator=g)
        return self.s[i], self.m[i], self.r[i]


class StateBuffer:
    def __init__(self, cap, dev):
        self.s = torch.zeros((cap, C.N_STICKERS), dtype=torch.uint8, device=dev)
        self.cap, self.n, self.pos = cap, 0, 0

    def add(self, s):
        k = min(len(s), self.cap)
        idx = (torch.arange(k, device=s.device) + self.pos) % self.cap
        self.s[idx] = s[:k]
        self.pos = (self.pos + k) % self.cap
        self.n = min(self.cap, self.n + k)

    def sample(self, b, g):
        return self.s[torch.randint(0, self.n, (b,), device=self.s.device, generator=g)]


@torch.no_grad()
def bellman_targets(frozen, enc, perms, solved, s):
    """y(s) = 0 if solved, else 1 + min over the 18 children of max(0, V_frozen(child)),
    with solved children counting as 0."""
    B = len(s)
    kids = s[:, perms].reshape(-1, C.N_STICKERS)                     # (B*18, 54)
    with torch.autocast("cuda", dtype=torch.bfloat16):
        _, v = frozen(enc(kids))
    v = v.float().clamp_min(0)
    v = torch.where((kids == solved).all(-1), torch.zeros_like(v), v).view(B, C.N_MOVES)
    y = 1 + v.min(1).values
    return torch.where((s == solved).all(-1), torch.zeros_like(y), y)


def value_group_scores(vb, states, paths, solved, lengths, n, dev):
    """[score at n, n-1, n-2, <= n-3]; score = mean over solved cubes of 1 - |V(start) - L| / L,
    L = length of the solution found. None for a group with no solved cubes (e.g. small n)."""
    idx = np.flatnonzero(solved & (lengths > 0))
    out = [None, None, None, None]
    if len(idx) == 0:
        return out
    L = np.array([len(paths[i]) for i in idx])
    keep = L > 0                                   # drop cubes that were already solved
    idx, L = idx[keep], L[keep]
    if len(idx) == 0:
        return out
    v = vb.values(torch.tensor(states[idx].astype(np.uint8), device=dev)).cpu().numpy()
    score = 1 - np.abs(v - L) / L
    sl = lengths[idx]
    for g, sel in enumerate([sl == n, sl == n - 1, sl == n - 2, sl <= n - 3]):
        if sel.any():
            out[g] = float(score[sel].mean())
    return out


def value_scores_by_remaining(vb, ex, n, dev, rng, max_eval=20000):
    """[score for r = n, n-1, n-2, <= n-3] over cubes along SOLVED paths, grouped by moves
    remaining r on the path; score = mean of 1 - |V - r| / r. None for an empty group."""
    out = [None, None, None, None]
    if ex is None:
        return out
    S, _, R = ex
    if len(R) > max_eval:
        k = rng.choice(len(R), max_eval, replace=False)
        S, R = S[k], R[k]
    v = vb.values(torch.tensor(S.astype(np.uint8), device=dev)).cpu().numpy()
    score = 1 - np.abs(v - R) / R
    for g, sel in enumerate([R == n, R == n - 1, R == n - 2, R <= n - 3]):
        if sel.any():
            out[g] = float(score[sel].mean())
    return out


def fmt_scores(sc):
    return "/".join("-" if x is None else f"{x:.2f}" for x in sc)


def length_schedule(n: int, total: int) -> np.ndarray:
    """Scramble lengths for one round, halving per step down from n:
    n gets total/2, n-1 gets total/4, n-2 gets total/8, ... The few cubes left over
    (total / 2^n) go to length 1, so the counts add up to `total` exactly.
    e.g. n=10, total=4096: 10:2048 9:1024 8:512 7:256 6:128 5:64 4:32 3:16 2:8 1:8"""
    counts = {}
    left = total
    for L in range(n, 0, -1):
        c = total >> (n - L + 1)
        counts[L] = c
        left -= c
    counts[1] += left
    return np.concatenate([np.full(c, L, dtype=np.int64) for L, c in counts.items() if c > 0])


def solve_need(k: int) -> float:
    """"halving" rule: solve rate needed at depth n-k is 1 - 0.5**(k+1)."""
    return 1.0 - 0.5 ** (k + 1)


def check_lower_depths(vb, n: int, per_depth: int, rng) -> list[float]:
    """Solve rate on fresh scrambles of length n-1, n-2, ..., 1 (each within its own length)."""
    vb.net.eval()
    rates = []
    for L in range(n - 1, 0, -1):
        states = C.scramble(per_depth, L, rng)[0]
        paths = vb.solve_batch(states, max_steps=L)
        rates.append(float(np.mean([p is not None for p in paths])))
    return rates


def path_examples(start_states: np.ndarray, paths):
    """States along each found path, with move taken and moves remaining."""
    S, M, R = [], [], []
    for st, p in zip(start_states, paths):
        if not p:
            continue
        cur = st
        for t, m in enumerate(p):
            S.append(cur); M.append(m); R.append(len(p) - t)
            cur = C.apply_move(cur, m)
    if not S:
        return None
    return np.stack(S), np.array(M), np.array(R)


def train(cfg: SelfTrainConfig):
    dev = torch.device("cuda")
    torch.manual_seed(cfg.seed)
    rng = np.random.default_rng(cfg.seed)
    g = torch.Generator(device=dev).manual_seed(cfg.seed)
    run_dir, ckpt_dir = ROOT / "runs" / "train" / cfg.name, ROOT / "checkpoints" / cfg.name
    run_dir.mkdir(parents=True, exist_ok=True); ckpt_dir.mkdir(parents=True, exist_ok=True)
    (run_dir / "config.json").write_text(json.dumps(asdict(cfg), indent=1))
    log = open(run_dir / "log.jsonl", "w")

    net = CubeNet(cfg.model).to(dev)
    if cfg.init_from:
        net = CubeNet.from_checkpoint(torch.load(ROOT / cfg.init_from, map_location=dev)).to(dev)
        print(f"continuing from {cfg.init_from} (weights only; buffers and optimizer start fresh)", flush=True)
    opt = torch.optim.AdamW(net.parameters(), lr=cfg.lr, weight_decay=cfg.weight_decay)
    enc, aug = Encoder(dev), (SymAugment(dev) if cfg.sym_aug else None)
    vb = ValueBeam(net, cfg.width, noise=cfg.search_noise, generator=g, noise_abs=cfg.search_noise_abs)
    buf = Replay(cfg.buffer_size, dev)
    bbuf = StateBuffer(cfg.bellman_buffer, dev)
    frozen = copy.deepcopy(net).eval()
    perms = torch.tensor(C.PERMS, device=dev)
    solved_t = torch.tensor(C.SOLVED, dtype=torch.uint8, device=dev)

    z = np.load(ROOT / cfg.monitor_data)
    mon = {"states": torch.tensor(z["states"][:cfg.monitor_size].astype(np.uint8), device=dev),
           "mask": torch.tensor(z["mask"][:cfg.monitor_size], device=dev),
           "dist": torch.tensor(z["dist"][:cfg.monitor_size].astype(np.int64), device=dev)}
    from ..data.labels import ExactLabeler  # monitoring only: how many training targets are optimal
    labeler = ExactLabeler()

    n, streak, t0, rnd, step = cfg.n_start, 0, time.perf_counter(), 0, 0
    print(f"self-training {'(continued)' if cfg.init_from else 'from scratch'}: width {cfg.width}, noise {cfg.search_noise} (x V) + {cfg.search_noise_abs} (fixed), n {cfg.n_start}->{cfg.n_max}, "
          f"{cfg.cubes_per_round} cubes/round, {cfg.train_steps} steps/round", flush=True)
    while rnd < cfg.max_rounds and time.perf_counter() - t0 < cfg.minutes * 60:
        rnd += 1
        if cfg.length_mode == "frontier":
            lengths = np.full(cfg.cubes_per_round, n, dtype=np.int64)
        else:
            lengths = rng.permutation(length_schedule(n, cfg.cubes_per_round))
        states = np.empty((cfg.cubes_per_round, C.N_STICKERS), dtype=np.int8)
        for L in np.unique(lengths):
            sel = lengths == L
            states[sel] = C.scramble(int(sel.sum()), int(L), rng)[0]
        net.eval()
        use_b = cfg.bellman_weight > 0
        paths = vb.solve_batch(states, max_steps=n, collect=cfg.collect_per_round if use_b else 0)
        if use_b:
            bbuf.add(torch.tensor(states.astype(np.uint8), device=dev))   # includes unsolved cubes
            bbuf.add(vb.collected)
        net.train()
        solved = np.array([p is not None for p in paths])
        at_n = lengths == n
        rate_n = float(solved[at_n].mean()) if at_n.any() else 0.0
        lens = np.array([len(p) for p in paths if p is not None])
        # value check per depth group (by scramble length): score = mean of 1 - |V - L| / L
        # over solved cubes, L = length of the solution found. Groups: n, n-1, n-2, <= n-3.
        ex = path_examples(states, paths)
        group_scores = value_scores_by_remaining(vb, ex, n, dev, rng)
        value_ok = group_scores[0] if group_scores[0] is not None else 0.0
        value_pass = all(sc is None or sc >= need
                         for sc, need in zip(group_scores, cfg.value_need))
        if ex is not None:
            buf.add(*(torch.tensor(a.astype(np.uint8 if i == 0 else np.int64), device=dev)
                      for i, a in enumerate(ex)))

        losses = {"loss_p": 0.0, "loss_v": 0.0, "loss_b": 0.0}
        if buf.n >= cfg.batch:
            for _ in range(cfg.train_steps):
                s, m, r = buf.sample(cfg.batch, g)
                if aug is not None:
                    mask = F.one_hot(m, C.N_MOVES).bool()
                    s, mask = aug(s, mask, g)
                    m = mask.float().argmax(1)
                if cfg.bellman_weight > 0:
                    # self-consistency targets on searched / unsolved cubes
                    bs = bbuf.sample(cfg.batch, g)
                    if aug is not None:
                        bs, _ = aug(bs, torch.zeros((len(bs), C.N_MOVES), dtype=torch.bool, device=dev), g)
                    yb = bellman_targets(frozen, enc, perms, solved_t, bs)
                else:  # path cubes only: no failed or off-path cube is trained
                    bs = s[:0]
                # ONE forward pass over both batches, so BatchNorm sees a single consistent mix
                # (two separate passes gave running statistics that biased evaluation values)
                with torch.autocast("cuda", dtype=torch.bfloat16):
                    logits_all, v_all = net(enc(torch.cat([s, bs])))
                nb = len(s)
                lp = F.cross_entropy(logits_all[:nb].float(), m)
                lv = F.huber_loss(v_all[:nb].float(), r.float())
                lb = F.huber_loss(v_all[nb:].float(), yb) if cfg.bellman_weight > 0 else torch.zeros((), device=dev)
                loss = lp + cfg.value_weight * lv + cfg.bellman_weight * lb
                opt.zero_grad(set_to_none=True)
                loss.backward()
                torch.nn.utils.clip_grad_norm_(net.parameters(), 1.0)
                opt.step()
                step += 1
                if step % cfg.target_sync == 0:
                    frozen.load_state_dict(net.state_dict())
                losses["loss_b"] += float(lb.detach()) / cfg.train_steps
                losses["loss_p"] += float(lp.detach()) / cfg.train_steps
                losses["loss_v"] += float(lv.detach()) / cfg.train_steps

        below = None
        if cfg.solve_rule == "halving":
            solve_pass = rate_n >= solve_need(0)
            if solve_pass and value_pass and n > 1:
                below = check_lower_depths(vb, n, cfg.check_cubes, rng)
                solve_pass = all(rate >= solve_need(k) for k, rate in enumerate(below, start=1))
            net.train()
        else:
            solve_pass = rate_n >= cfg.promote_at
        streak = streak + 1 if (solve_pass and value_pass) else 0
        promoted = False
        if streak >= cfg.promote_rounds and n < cfg.n_max:
            n, streak, promoted = n + 1, 0, True

        rec = {"round": rnd, "step": step, "time": time.perf_counter() - t0, "n": n,
               "solve_rate": float(solved.mean()), "solve_rate_at_n": rate_n, "value_ok_at_n": value_ok, "value_scores": group_scores,
               "mean_len": float(lens.mean()) if len(lens) else None,
               "buffer": buf.n, **losses}
        if below is not None:
            rec["solve_rates_below"] = below   # depths n-1, n-2, ..., 1
        if rnd % cfg.eval_every == 0 or promoted:
            rec["monitor"] = validate(net, enc, mon, max_dist=12)
            # label quality of this round's new training examples. Only examples whose
            # remaining path is <= 12 can be checked exactly (path length bounds the distance).
            ok = np.flatnonzero(ex[2] <= 12) if ex is not None else np.zeros(0, int)
            if len(ok):
                k = rng.choice(ok, size=min(2000, len(ok)), replace=False)
                d, mk = labeler.label(ex[0][k], max_dist=int(min(n, 12)))
                rec["target_move_optimal"] = float(mk[np.arange(len(k)), ex[1][k]].mean())
                rec["target_value_exact"] = float((d == ex[2][k]).mean())
            torch.save(net.checkpoint(), ckpt_dir / "last.pt")
            mm = rec["monitor"]
            print(f"round {rnd:5d}  {rec['time']:6.0f}s  n={n:2d}  solved {rec['solve_rate']:.3f} "
                  f"(at n: {rate_n:.3f}, value scores n..n-3: {fmt_scores(group_scores)})  len {rec['mean_len'] or 0:.2f}  buffer {buf.n:,}  "
                  f"loss_p {losses['loss_p']:.3f} loss_v {losses['loss_v']:.3f} loss_b {losses['loss_b']:.3f}"
                  f"{'  PROMOTED' if promoted else ''}"
                  + ("\n     solve rates n-1, n-2, ...: " + " ".join(f"{x:.3f}" for x in below) if below else "")
                  + "\n"
                  f"     exact-label monitor: policy acc {mm['acc']:.3f}  value mae {mm['mae']:.3f}  "
                  f"per distance acc: " + " ".join(f"{k}:{p['acc']:.2f}" for k, p in mm["per_dist"].items())
                  + (f"\n     training targets: move optimal {rec['target_move_optimal']:.3f}, "
                     f"value exact {rec['target_value_exact']:.3f}" if "target_move_optimal" in rec else ""),
                  flush=True)
        log.write(json.dumps(rec) + "\n"); log.flush()
    torch.save(net.checkpoint(), ckpt_dir / "last.pt")
    log.close()
    return net


def main():
    ap = argparse.ArgumentParser()
    for name, f in SelfTrainConfig.__dataclass_fields__.items():
        if name == "model":
            continue
        if name == "value_need":
            ap.add_argument("--value-need", type=float, nargs=4, default=list(f.default),
                            metavar=("N", "N-1", "N-2", "LE_N-3"))
        else:
            ap.add_argument("--" + name.replace("_", "-"), type=type(f.default), default=f.default)
    a = vars(ap.parse_args())
    a["value_need"] = tuple(a["value_need"])
    train(SelfTrainConfig(**a))


if __name__ == "__main__":
    main()
