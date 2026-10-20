"""DL3DV benchmark scenes: metadata and the fixed train / eval split.

Layout (as downloaded):

    <root>/benchmark-meta.csv                    hash, environment (bounded | unbounded), scene category, ...
    <root>/<hash>/gaussian_splat/images_4/       960x540 frames, named as in COLMAP
    <root>/<hash>/gaussian_splat/sparse/0/       COLMAP model (full-resolution cameras)

The split is written once to a JSON file (`python -m data.dl3dv`) and read by
`core.trainer_dl3dv`, so it never changes between runs:

    train         scenes the model trains on
    eval_seen     a subset of `train`, evaluated on their held-out views
    eval_unseen   scenes never trained on, evaluated on their held-out views
    environment   {hash: bounded | unbounded} for every listed scene

Only scenes with EDGS features (`<edgs_root>/<hash>/features.npz`) are used.
Eval scenes are picked per environment with a fixed seed, round-robin over the
scene categories (Supermarkets, Botanical-Gardens, ...) so they spread across them.
"""

import argparse
import csv
import json
import os
import random
from collections import defaultdict
from typing import Dict, List


def read_meta(root: str) -> List[Dict[str, str]]:
    with open(os.path.join(root, "benchmark-meta.csv")) as f:
        return list(csv.DictReader(f))


def _pick(rows: List[Dict[str, str]], n: int, rng: random.Random) -> List[Dict[str, str]]:
    """`n` rows, one category at a time (categories in random order, a random
    scene of each), until `n` are taken."""
    by_cat = defaultdict(list)
    for row in rows:
        by_cat[row["scene"]].append(row)
    cats = sorted(by_cat)
    rng.shuffle(cats)
    for cat in cats:
        rng.shuffle(by_cat[cat])
    picked = []
    while len(picked) < n:
        for cat in cats:
            if by_cat[cat] and len(picked) < n:
                picked.append(by_cat[cat].pop())
    return picked


def make_split(root: str, edgs_root: str, unseen: Dict[str, int], seen: Dict[str, int], seed: int) -> Dict:
    rows = [r for r in read_meta(root) if os.path.exists(os.path.join(edgs_root, r["hash"], "features.npz"))]
    rng = random.Random(seed)
    split = {"eval_unseen": [], "eval_seen": [], "train": []}
    for env in ("bounded", "unbounded"):
        env_rows = sorted((r for r in rows if r["environment"] == env), key=lambda r: r["hash"])
        held = _pick(env_rows, unseen[env], rng)
        held_ids = {r["hash"] for r in held}
        train = [r for r in env_rows if r["hash"] not in held_ids]
        split["eval_unseen"] += [r["hash"] for r in held]
        split["eval_seen"] += [r["hash"] for r in _pick(train, seen[env], rng)]
        split["train"] += [r["hash"] for r in train]
    split = {k: sorted(v) for k, v in split.items()}
    split["environment"] = {r["hash"]: r["environment"] for r in rows}
    split["category"] = {r["hash"]: r["scene"] for r in rows}
    split["seed"] = seed
    return split


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--root", default="/projects/hpc/radub/datasets/dl3dv")
    ap.add_argument("--edgs_root", default="/projects/hpc/radub/experiments/edgs_feat_dl3dv")
    ap.add_argument("--out", required=True)
    ap.add_argument("--unseen_bounded", type=int, default=7)
    ap.add_argument("--unseen_unbounded", type=int, default=6)
    ap.add_argument("--seen_bounded", type=int, default=5)
    ap.add_argument("--seen_unbounded", type=int, default=5)
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()
    if os.path.exists(args.out):
        raise SystemExit(f"{args.out} exists, not overwriting (the split must stay fixed)")
    split = make_split(
        args.root, args.edgs_root,
        unseen={"bounded": args.unseen_bounded, "unbounded": args.unseen_unbounded},
        seen={"bounded": args.seen_bounded, "unbounded": args.seen_unbounded},
        seed=args.seed,
    )
    os.makedirs(os.path.dirname(os.path.abspath(args.out)), exist_ok=True)
    with open(args.out, "w") as f:
        json.dump(split, f, indent=1)
    for key in ("train", "eval_seen", "eval_unseen"):
        envs = [split["environment"][h] for h in split[key]]
        print(f"{key}: {len(envs)} ({envs.count('bounded')} bounded, {envs.count('unbounded')} unbounded)")
        if key != "train":
            for h in split[key]:
                print(f"  {h[:8]}  {split['environment'][h]:9s}  {split['category'][h]}")


if __name__ == "__main__":
    main()
