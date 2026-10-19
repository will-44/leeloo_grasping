#!/usr/bin/env python3
"""Benchmark GraspGenX sur Jetson AGX Orin.

Modes :
  sweep   : latence + mémoire GPU de pointe de GraspGenXSampler.sample()
            (diffusion + discriminateur) selon num_grasps x nombre de points.
  planner : run_planner_on_object, GraspMoE vs diffusion seule, sur les nuages
            d'objets d'exemple ; latence + nombre de prises retenues.

Chaque config : warmup puis >= --iters inférences ; moyenne et p95 en ms.
"""
import argparse
import glob
import json
import logging
import os
import platform
import time

import numpy as np
import torch

from graspgenx.grasp_server import GraspGenXSampler
from graspgenx.samplers import run_planner_on_object
from graspgenx.utils.checkpoint_io import load_model_cfg

REPO = "/bench/GraspGenX"
CKPT = f"{REPO}/ext/graspgenx_checkpoints/release"


def stats(lat_s):
    a = np.asarray(lat_s) * 1000.0
    return {
        "n": int(a.size),
        "mean_ms": float(a.mean()),
        "p50_ms": float(np.percentile(a, 50)),
        "p95_ms": float(np.percentile(a, 95)),
        "max_ms": float(a.max()),
    }


def load_objects():
    objs = []
    for f in sorted(glob.glob(f"{REPO}/assets/sample_data/object_pc/*.json")):
        pc = np.asarray(json.load(open(f))["pc"], dtype=np.float32)
        objs.append((os.path.basename(f), pc))
    return objs


def resample(pc, n, rng):
    idx = rng.choice(len(pc), n, replace=n > len(pc))
    out = pc[idx].copy()
    if n > len(pc):
        out += rng.normal(scale=0.001, size=out.shape).astype(np.float32)
    return out


def make_sampler(args):
    if args.tensorrt:
        import sys
        sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
        import trt_onnx_accel

        trt_onnx_accel.patch()
    cfg = load_model_cfg(f"{CKPT}/gen", f"{CKPT}/dis", None, None)
    return GraspGenXSampler(
        cfg,
        args.gripper,
        assets_dir=f"{REPO}/assets",
        use_tensorrt=args.tensorrt,
        tensorrt_precision=args.tensorrt_precision,
    )


def run_sweep(sampler, args, rng):
    _, base_pc = load_objects()[0]
    results = []
    for n_pts in args.points:
        pc = torch.from_numpy(resample(base_pc, n_pts, rng)).cuda()
        for ng in args.num_grasps:
            for _ in range(args.warmup):
                sampler.sample(pc, num_grasps=ng)
            torch.cuda.synchronize()
            torch.cuda.reset_peak_memory_stats()
            lat, lat_model = [], []
            for _ in range(args.iters):
                torch.cuda.synchronize()
                t0 = time.perf_counter()
                sampler.sample(pc, num_grasps=ng)
                torch.cuda.synchronize()
                lat.append(time.perf_counter() - t0)
            # modèle seul (sans retrait des points aberrants, fait sur CPU)
            for _ in range(args.iters):
                torch.cuda.synchronize()
                t0 = time.perf_counter()
                sampler.sample(pc, num_grasps=ng, remove_outliers=False)
                torch.cuda.synchronize()
                lat_model.append(time.perf_counter() - t0)
            r = {
                "points": n_pts,
                "num_grasps": ng,
                **stats(lat),
                "model_only": stats(lat_model),
                "peak_alloc_MB": torch.cuda.max_memory_allocated() / 2**20,
                "peak_reserved_MB": torch.cuda.max_memory_reserved() / 2**20,
            }
            print(json.dumps(r), flush=True)
            results.append(r)
    return results


def run_planner(sampler, args, rng):
    objs = load_objects()
    results = []
    per_obj = max(1, args.iters // len(objs))
    for planner in args.planners:
        for name, pc in objs:
            pc_c = pc - pc.mean(axis=0)
            kw = dict(
                planner=planner,
                grasp_threshold=args.threshold,
                num_grasps=args.planner_num_grasps,
                topk_num_grasps=-1,
                moe_obb_density=args.moe_obb_density,
            )
            for _ in range(args.warmup):
                run_planner_on_object(pc_c, sampler, **kw)
            torch.cuda.synchronize()
            torch.cuda.reset_peak_memory_stats()
            lat, kept, n_diff, n_obb = [], [], [], []
            for _ in range(per_obj):
                torch.cuda.synchronize()
                t0 = time.perf_counter()
                g, conf, tags, _ = run_planner_on_object(pc_c, sampler, **kw)
                torch.cuda.synchronize()
                lat.append(time.perf_counter() - t0)
                conf = np.asarray(
                    conf.cpu() if torch.is_tensor(conf) else conf, dtype=np.float32
                )
                kept.append(int((conf >= args.threshold).sum()))
                n_diff.append(tags.count("diff") if tags else len(conf))
                n_obb.append(tags.count("obb") if tags else 0)
            r = {
                "planner": planner,
                "object": name,
                "points": len(pc),
                **stats(lat),
                "kept_mean": float(np.mean(kept)),
                "kept_min": int(np.min(kept)),
                "diff_mean": float(np.mean(n_diff)),
                "obb_mean": float(np.mean(n_obb)),
                "peak_alloc_MB": torch.cuda.max_memory_allocated() / 2**20,
            }
            print(json.dumps(r), flush=True)
            results.append(r)
    return results


def main():
    p = argparse.ArgumentParser()
    p.add_argument("mode", choices=["sweep", "planner"])
    p.add_argument("--gripper", default="robotiq_2f_85")
    p.add_argument("--iters", type=int, default=100)
    p.add_argument("--warmup", type=int, default=5)
    p.add_argument("--points", type=int, nargs="+", default=[1000, 2000, 3500, 7000, 10000])
    p.add_argument("--num_grasps", type=int, nargs="+", default=[50, 100, 200])
    p.add_argument("--planners", nargs="+", default=["diffusion", "graspmoe"])
    p.add_argument("--planner_num_grasps", type=int, default=200)
    p.add_argument("--threshold", type=float, default=0.7)
    p.add_argument("--moe_obb_density", default="dense-topandside")
    p.add_argument("--tensorrt", action="store_true")
    p.add_argument("--tensorrt_precision", default="fp32")
    p.add_argument("--tag", default="")
    p.add_argument("--out", default="/bench/results")
    args = p.parse_args()

    logging.disable(logging.INFO)
    rng = np.random.default_rng(0)
    torch.manual_seed(0)
    t0 = time.perf_counter()
    sampler = make_sampler(args)
    load_s = time.perf_counter() - t0
    print(f"model load {load_s:.1f}s", flush=True)

    res = run_sweep(sampler, args, rng) if args.mode == "sweep" else run_planner(sampler, args, rng)
    os.makedirs(args.out, exist_ok=True)
    out = f"{args.out}/{args.mode}{'_' + args.tag if args.tag else ''}.json"
    json.dump(
        {
            "args": vars(args),
            "torch": torch.__version__,
            "host": platform.node(),
            "model_load_s": load_s,
            "trt_build": __import__("trt_onnx_accel").BUILD_LOG if args.tensorrt else None,
            "results": res,
        },
        open(out, "w"),
        indent=1,
    )
    print("wrote", out)


if __name__ == "__main__":
    main()
