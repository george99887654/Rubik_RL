"""Supervised training on the exactly labelled dataset.

    uv run python -m rubik_rl.train.train --data data/exact_L12 --name exact_v1 --steps 10000

Loss per cube
    policy: -log( sum of p(m) over the optimal moves m )        ("any correct move counts")
    value:  Huber(predicted distance, exact distance)            (trained, not yet used by search)
    total = policy + value_weight * value

Batches are drawn uniformly from the dataset ("natural" composition: no rebalancing).
A held-out slice of the data is used for validation, reported per distance.
Outputs: runs/train/<name>/{config.json, log.jsonl}, checkpoints/<name>/{last.pt, best.pt}.
"""
from __future__ import annotations

import argparse
import json
import math
import time
from dataclasses import asdict, dataclass, field
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F

from ..cube import cube as C
from ..models import CubeNet, CubeNetConfig

ROOT = Path(__file__).resolve().parents[2]


@dataclass
class TrainConfig:
    data: str = "data/exact_L12"
    name: str = "exact_v1"
    steps: int = 10_000
    batch: int = 8192
    lr: float = 1e-3
    warmup: int = 500
    weight_decay: float = 1e-5
    value_weight: float = 1.0
    grad_clip: float = 1.0
    val_size: int = 50_000
    sym_aug: int = 0          # 1 = apply a random one of the 48 cube symmetries to every training cube
    log_every: int = 200
    val_every: int = 1000
    seed: int = 0
    model: CubeNetConfig = field(default_factory=CubeNetConfig)


# ------------------------------------------------------------------ data
def load_dataset(path: Path, device):
    shards = sorted(path.glob("shard_*.npz"))
    if not shards:
        raise FileNotFoundError(f"no shards in {path}")
    parts = [np.load(s) for s in shards]
    cat = lambda k: np.concatenate([p[k] for p in parts])  # noqa: E731
    return {
        "states": torch.tensor(cat("states").astype(np.uint8), device=device),
        "mask": torch.tensor(cat("mask"), device=device),
        "dist": torch.tensor(cat("dist").astype(np.int64), device=device),
    }, [s.name for s in shards]


class SymAugment:
    """Random cube symmetry per example, on the GPU. Distance labels are unchanged; the
    optimal-move mask is permuted with the moves (see cube/symmetry.py)."""

    def __init__(self, device):
        from ..cube import symmetry as S

        self.perm = torch.tensor(S.SYM_PERM, device=device)
        self.color = torch.tensor(S.SYM_COLOR.astype(np.int64), device=device)
        self.move_inv = torch.tensor(S.SYM_MOVE_INV, device=device)
        self.n = S.N_SYM

    def __call__(self, states, mask, generator=None):
        k = torch.randint(0, self.n, (len(states),), device=states.device, generator=generator)
        moved = torch.gather(states, 1, self.perm[k])
        states = torch.gather(self.color[k], 1, moved.long()).to(states.dtype)
        mask = torch.gather(mask, 1, self.move_inv[k])
        return states, mask


class Encoder:
    def __init__(self, device):
        self.idx = torch.tensor(C.NONCENTER_IDX, device=device)

    def __call__(self, states: torch.Tensor) -> torch.Tensor:
        return F.one_hot(states[:, self.idx].long(), C.N_FACES).flatten(1).float()


# ------------------------------------------------------------------ loss / metrics
def losses(net, enc, states, mask, dist, value_weight):
    with torch.autocast("cuda", dtype=torch.bfloat16, enabled=states.is_cuda):
        logits, value = net(enc(states))
    logp = F.log_softmax(logits.float(), dim=-1)
    lp_correct = logp.masked_fill(~mask, float("-inf")).logsumexp(-1)
    loss_p = -lp_correct.mean()
    loss_v = F.huber_loss(value.float(), dist.float()) if value is not None else torch.zeros((), device=states.device)
    with torch.no_grad():
        acc = mask.gather(1, logits.argmax(-1, keepdim=True)).float().mean()
        mae = (value.float() - dist).abs().mean() if value is not None else torch.zeros(())
    return loss_p + value_weight * loss_v, {"loss_p": loss_p, "loss_v": loss_v, "acc": acc, "mae": mae}


@torch.no_grad()
def validate(net, enc, val, max_dist=20, bs=16384):
    net.eval()
    hits, errs, lps = [], [], []
    for i in range(0, len(val["dist"]), bs):
        s, m, d = val["states"][i:i + bs], val["mask"][i:i + bs], val["dist"][i:i + bs]
        with torch.autocast("cuda", dtype=torch.bfloat16, enabled=s.is_cuda):
            logits, value = net(enc(s))
        logp = F.log_softmax(logits.float(), -1)
        lps.append(logp.masked_fill(~m, float("-inf")).logsumexp(-1))
        hits.append(m.gather(1, logits.argmax(-1, keepdim=True))[:, 0].float())
        errs.append((value.float() - d).abs() if value is not None else torch.zeros_like(d, dtype=torch.float))
    hit, err, lp, d = torch.cat(hits), torch.cat(errs), torch.cat(lps), val["dist"]
    per = {}
    for k in range(1, max_dist + 1):
        sel = d == k
        if sel.any():
            per[k] = {"n": int(sel.sum()), "acc": float(hit[sel].mean()), "mae": float(err[sel].mean())}
    net.train()
    return {"acc": float(hit.mean()), "loss_p": float(-lp.mean()), "mae": float(err.mean()), "per_dist": per}


def lr_at(step, cfg):
    if step < cfg.warmup:
        return cfg.lr * (step + 1) / cfg.warmup
    t = (step - cfg.warmup) / max(1, cfg.steps - cfg.warmup)
    return cfg.lr * 0.5 * (1 + math.cos(math.pi * min(1.0, t)))


# ------------------------------------------------------------------ main loop
def train(cfg: TrainConfig):
    dev = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    torch.manual_seed(cfg.seed)
    run_dir = ROOT / "runs" / "train" / cfg.name
    ckpt_dir = ROOT / "checkpoints" / cfg.name
    run_dir.mkdir(parents=True, exist_ok=True)
    ckpt_dir.mkdir(parents=True, exist_ok=True)

    data, shard_names = load_dataset(ROOT / cfg.data, dev)
    n = len(data["dist"])
    g = torch.Generator(device=dev).manual_seed(cfg.seed)
    perm = torch.randperm(n, device=dev, generator=g)
    val_idx, train_idx = perm[: cfg.val_size], perm[cfg.val_size:]
    val = {k: v[val_idx] for k, v in data.items()}
    train_data = {k: v[train_idx] for k, v in data.items()}
    n_train = len(train_idx)
    del data

    net = CubeNet(cfg.model).to(dev)
    opt = torch.optim.AdamW(net.parameters(), lr=cfg.lr, weight_decay=cfg.weight_decay)
    enc = Encoder(dev)
    aug = SymAugment(dev) if cfg.sym_aug else None
    (run_dir / "config.json").write_text(json.dumps(
        {**asdict(cfg), "n_train": n_train, "n_val": cfg.val_size, "shards": shard_names,
         "params": net.n_params()}, indent=1))
    log = open(run_dir / "log.jsonl", "w")
    print(f"train {n_train:,} cubes, val {cfg.val_size:,}, model {net.n_params() / 1e6:.1f}M params, "
          f"{cfg.steps} steps x {cfg.batch} = {cfg.steps * cfg.batch / n_train:.1f} epochs", flush=True)

    best, order, pos, t0 = -1.0, torch.randperm(n_train, device=dev, generator=g), 0, time.perf_counter()
    acc_sum = {}
    for step in range(cfg.steps):
        if pos + cfg.batch > n_train:
            order, pos = torch.randperm(n_train, device=dev, generator=g), 0
        idx = order[pos:pos + cfg.batch]
        pos += cfg.batch
        for gr in opt.param_groups:
            gr["lr"] = lr_at(step, cfg)
        b_states, b_mask = train_data["states"][idx], train_data["mask"][idx]
        if aug is not None:
            b_states, b_mask = aug(b_states, b_mask, g)
        loss, m = losses(net, enc, b_states, b_mask, train_data["dist"][idx], cfg.value_weight)
        opt.zero_grad(set_to_none=True)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(net.parameters(), cfg.grad_clip)
        opt.step()
        for k, v in m.items():  # keep on GPU: no sync every step
            acc_sum[k] = acc_sum.get(k, 0.0) + v.detach()

        if (step + 1) % cfg.log_every == 0:
            rec = {"step": step + 1, "lr": lr_at(step, cfg), "time": time.perf_counter() - t0,
                   **{k: float(v) / cfg.log_every for k, v in acc_sum.items()}}
            acc_sum = {}
            if (step + 1) % cfg.val_every == 0 or step + 1 == cfg.steps:
                rec["val"] = validate(net, enc, val)
                if rec["val"]["acc"] > best:
                    best = rec["val"]["acc"]
                    torch.save(net.checkpoint(), ckpt_dir / "best.pt")
                torch.save(net.checkpoint(), ckpt_dir / "last.pt")
            log.write(json.dumps(rec) + "\n"); log.flush()
            msg = (f"step {rec['step']:6d}  loss_p {rec['loss_p']:.4f}  acc {rec['acc']:.4f}  "
                   f"loss_v {rec['loss_v']:.4f}  mae {rec['mae']:.3f}  lr {rec['lr']:.2e}  "
                   f"{rec['time']:.0f}s")
            if "val" in rec:
                v = rec["val"]
                msg += f"\n   VAL acc {v['acc']:.4f}  mae {v['mae']:.3f}  per distance acc: " + " ".join(
                    f"{k}:{p['acc']:.3f}" for k, p in v["per_dist"].items())
            print(msg, flush=True)
    log.close()
    return net


def main():
    ap = argparse.ArgumentParser()
    for f_name, f in TrainConfig.__dataclass_fields__.items():
        if f_name == "model":
            continue
        ap.add_argument("--" + f_name.replace("_", "-"), type=type(f.default), default=f.default)
    ap.add_argument("--width", type=int, default=1024)
    ap.add_argument("--blocks", type=int, default=4)
    ap.add_argument("--stem", type=int, default=4096)
    a = vars(ap.parse_args())
    model = CubeNetConfig(width=a.pop("width"), n_blocks=a.pop("blocks"), stem_dim=a.pop("stem"))
    train(TrainConfig(**a, model=model))


if __name__ == "__main__":
    main()
