#!/usr/bin/env python3
"""Entry point for the three-seed Dex standing/walking reproduction package."""
from pathlib import Path
import argparse
import gzip
import hashlib
import json
import os
import shutil
import subprocess
import sys
import time

ROOT = Path(__file__).resolve().parent
PYTHON = os.environ.get("DEX_PYTHON", sys.executable)


def sha256(path):
    h = hashlib.sha256()
    with Path(path).open("rb") as f:
        for block in iter(lambda: f.read(8 << 20), b""):
            h.update(block)
    return h.hexdigest()


def environment():
    env = os.environ.copy()
    env.update(DEX_PROJECT_ROOT=str(ROOT), DEX_PYTHON=PYTHON,
               PYTHONPATH=str(ROOT / "src") + os.pathsep + str(ROOT / "runtime"),
               PYTHONDONTWRITEBYTECODE="1", OMP_NUM_THREADS="1", MKL_NUM_THREADS="1")
    return env


def verify():
    data = json.loads((ROOT / "MANIFEST.json").read_text())
    errors = []
    for name, expected in data["files"].items():
        p = ROOT / name
        if not p.is_file() or sha256(p) != expected:
            errors.append(name)
    if errors:
        raise SystemExit("Missing or changed files: " + ", ".join(errors))
    print(f"OK: {len(data['files'])} files match the release SHA-256 manifest.")


def physics_for(checkpoint):
    adjacent = checkpoint.parent / "physics_scene.mjb"
    if adjacent.is_file():
        return adjacent
    bundled = [ROOT / "models" / f"seed_{s}" / "policy.pt" for s in range(3)]
    if checkpoint not in bundled:
        raise SystemExit("A new training checkpoint must have its own adjacent physics_scene.mjb.")
    cache = ROOT / ".cache" / "physics_scene.mjb"
    expected = json.loads((ROOT / "models/physics.json").read_text())["uncompressed_sha256"]
    if not cache.exists():
        cache.parent.mkdir(exist_ok=True)
        tmp = cache.with_name(cache.name + f".{os.getpid()}.tmp")
        with gzip.open(ROOT / "models/physics_scene.mjb.gz", "rb") as source, tmp.open("xb") as dest:
            shutil.copyfileobj(source, dest, 8 << 20)
        if sha256(tmp) != expected:
            raise SystemExit("Bundled physics model checksum mismatch.")
        os.replace(tmp, cache)
    elif sha256(cache) != expected:
        raise SystemExit("Cached physics model changed; move .cache/physics_scene.mjb aside and retry.")
    return cache


def train(args):
    name = args.name or f"reproduce_seed{args.seed}"
    if not name or any(c not in "abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789_-" for c in name):
        raise SystemExit("Use letters, digits, underscores or hyphens for --name.")
    if (ROOT / "runs" / name).exists() or (ROOT / "audit" / (name + "_JOB.json")).exists():
        raise SystemExit("Run already exists. Choose a new --name; existing results are never overwritten.")
    recipe = json.loads((ROOT / "configs/success_recipe.json").read_text())["training_args"]
    cmd = [PYTHON, str(ROOT / "tools/launch.py"), name, *recipe,
           "--seed", str(args.seed), "--seconds", str(args.seconds)]
    if args.iterations is not None:
        cmd += ["--iterations", str(args.iterations)]
    env = environment()
    # Cache reuse is explicit; an inherited variable must not silently change a fresh run.
    env.pop("DEX_REUSE_CACHE_FROM", None)
    if args.cache_from:
        env["DEX_REUSE_CACHE_FROM"] = args.cache_from
    subprocess.run(cmd, cwd=ROOT, env=env, check=True)
    result = ROOT / "audit" / (name + "_JOB_RESULT.json")
    last_status = None
    while not result.exists():
        status = ROOT / "runs" / name / "status.json"
        if status.exists():
            row = json.loads(status.read_text())
            current = (row["checkpoint"], row["iteration"])
            if current != last_status:
                print(f"{name}: {row['checkpoint']} at {row['training_seconds']:.2f} training seconds", flush=True)
                last_status = current
        time.sleep(2)
    outcome = json.loads(result.read_text())
    if outcome["returncode"] != 0:
        raise SystemExit(f"Training failed. Read runs/{name}.log; exit={outcome['returncode']}")
    print((ROOT / "runs" / name / "FINISHED.json").read_text())


def evaluate(args):
    ck = Path(args.checkpoint).expanduser().resolve() if args.checkpoint else ROOT / "models" / f"seed_{args.seed}" / "policy.pt"
    if not ck.is_file():
        raise SystemExit("Checkpoint not found: " + str(ck))
    default = f"seed_{args.seed}" + ("_heldout" if args.heldout else "")
    out = Path(args.out).expanduser().resolve() if args.out else ROOT / "outputs" / default
    if out.exists():
        raise SystemExit("Output already exists. Choose a new --out to preserve prior results.")
    physics = physics_for(ck)
    out.mkdir(parents=True)
    env = environment()
    env.update(DEX_EVAL_OUTPUT=str(out), DEX_PHYSICS_MODEL=str(physics))
    seeds = list(range(2000, 2010)) if args.heldout else [1000, 1001, 1002]
    cmd = [PYTHON, str(ROOT / "tools/evaluate_cpu.py"), str(ck), "--variant", "ema",
           "--scenario", "both", "--speed", "0.4", "--seeds", *map(str, seeds)]
    with (out / "evaluation.log").open("x") as log:
        result = subprocess.run(cmd, cwd=ROOT, env=env, stdout=log, stderr=subprocess.STDOUT)
    if result.returncode:
        raise SystemExit("Evaluation failed; read " + str(out / "evaluation.log"))
    for path in sorted((out / "v0.4").glob("*.json")):
        data = json.loads(path.read_text())
        walking = sum(x.get("walking28_pass") is True for x in data["results"])
        print(f"{data['scenario']}: passed={data['passed']}/{len(data['results'])}, walking28={walking}; {path}")


def video(args):
    src = Path(args.evaluation).expanduser().resolve() if args.evaluation else (
        ROOT / "outputs" / f"seed_{args.seed}" / "v0.4" / "policy_ema_primary_seeds1000-1001-1002.json")
    if not src.is_file() or not src.with_suffix(".npz").is_file():
        raise SystemExit("Run eval first, or specify an evaluation JSON with its adjacent NPZ.")
    out = Path(args.out).expanduser().resolve() if args.out else ROOT / "outputs" / f"video_seed_{args.seed}"
    env = environment()
    env.setdefault("MUJOCO_GL", "egl" if sys.platform.startswith("linux") else "glfw")
    subprocess.run([PYTHON, str(ROOT / "tools/render_trajectory.py"), str(src),
                    "--out", str(out), "--index", str(args.index)], cwd=ROOT, env=env, check=True)


def main():
    p = argparse.ArgumentParser(description=__doc__)
    sub = p.add_subparsers(dest="command", required=True)
    sub.add_parser("verify", help="Verify all delivered files; standard Python only")
    t = sub.add_parser("train", help="Train a fresh random actor and world model")
    t.add_argument("--seed", type=int, choices=range(3), required=True)
    t.add_argument("--name")
    t.add_argument("--seconds", type=float, default=300)
    t.add_argument("--iterations", type=int, help=argparse.SUPPRESS)
    t.add_argument("--cache-from", help="Copy compile caches from a run in this package; never reads learned weights")
    e = sub.add_parser("eval", help="Evaluate bundled EMA policy or a newly trained checkpoint")
    e.add_argument("--seed", type=int, choices=range(3), default=0)
    e.add_argument("--checkpoint")
    e.add_argument("--heldout", action="store_true", help="Use ten held-out initial states instead of three screening states")
    e.add_argument("--out")
    v = sub.add_parser("video", help="Render a recorded evaluation at real-time speed")
    v.add_argument("--seed", type=int, choices=range(3), default=0)
    v.add_argument("--evaluation")
    v.add_argument("--index", type=int, default=0)
    v.add_argument("--out")
    args = p.parse_args()
    {"verify": lambda a: verify(), "train": train, "eval": evaluate, "video": video}[args.command](args)


if __name__ == "__main__":
    main()
