#!/usr/bin/env python3
"""Benchmark segmentation sur Orin.

  sam2 : SAM2.1 hiera_tiny — encodeur (set_image) + décodeur (1 clic, 1 boîte),
         en fp32 et bf16 (autocast). Image RGB réelle redimensionnée.
  geom : segmentation géométrique sur nuage : voxel 5 mm, plan de table RANSAC,
         clustering DBSCAN (Open3D, CPU) ; scène réelle GraspGenX (profondeur).
"""
import argparse
import json
import logging
import os
import time

import numpy as np
import torch
from PIL import Image


def stats(lat_s):
    a = np.asarray(lat_s) * 1000.0
    return {"n": int(a.size), "mean_ms": float(a.mean()), "p50_ms": float(np.percentile(a, 50)),
            "p95_ms": float(np.percentile(a, 95)), "max_ms": float(a.max())}


def timed(fn, iters, warmup, cuda=True):
    for _ in range(warmup):
        fn()
    lat = []
    for _ in range(iters):
        if cuda:
            torch.cuda.synchronize()
        t0 = time.perf_counter()
        fn()
        if cuda:
            torch.cuda.synchronize()
        lat.append(time.perf_counter() - t0)
    return stats(lat)


def bench_sam2(args):
    from sam2.build_sam import build_sam2
    from sam2.sam2_image_predictor import SAM2ImagePredictor

    model = build_sam2("configs/sam2.1/sam2.1_hiera_t.yaml", args.sam2_ckpt, device="cuda")
    pred = SAM2ImagePredictor(model)
    img = np.array(Image.open(args.image).convert("RGB").resize((args.width, args.height)))
    h, w = img.shape[:2]
    pt = np.array([[w // 2, h // 2]])
    box = np.array([w * 0.35, h * 0.35, w * 0.65, h * 0.65])
    out = []
    for dtype in ["fp32", "bf16"]:
        ctx = (lambda: torch.autocast("cuda", dtype=torch.bfloat16)) if dtype == "bf16" else (lambda: torch.autocast("cuda", enabled=False))
        torch.cuda.reset_peak_memory_stats()
        with torch.inference_mode(), ctx():
            enc = timed(lambda: pred.set_image(img), args.iters, args.warmup)
            dec_pt = timed(lambda: pred.predict(point_coords=pt, point_labels=np.array([1]), multimask_output=False), args.iters, args.warmup)
            dec_box = timed(lambda: pred.predict(box=box, multimask_output=False), args.iters, args.warmup)

            def full():
                pred.set_image(img)
                pred.predict(point_coords=pt, point_labels=np.array([1]), multimask_output=False)
            tot = timed(full, args.iters, args.warmup)
        r = {"model": "sam2.1_hiera_tiny", "dtype": dtype, "image": f"{w}x{h}", "encoder": enc,
             "decoder_point": dec_pt, "decoder_box": dec_box, "total_1click": tot,
             "peak_alloc_MB": torch.cuda.max_memory_allocated() / 2**20}
        print(json.dumps(r), flush=True)
        out.append(r)
    return out


def bench_geom(args):
    import open3d as o3d

    root = "/bench/GraspGenX/assets/sample_data/real_world/00"
    meta = json.load(open(f"{root}/meta_data.json"))
    depth = np.load(f"{root}/depth.npy").astype(np.float32)
    K = np.array(meta["intrinsics"])
    v, u = np.nonzero((depth > 0.1) & (depth < 2.0))
    z = depth[v, u]
    pts = np.stack([(u - K[0, 2]) * z / K[0, 0], (v - K[1, 2]) * z / K[1, 1], z], 1)

    res_holder = {}

    def run():
        pc = o3d.geometry.PointCloud(o3d.utility.Vector3dVector(pts))
        pc = pc.voxel_down_sample(0.005)
        _, inl = pc.segment_plane(distance_threshold=0.01, ransac_n=3, num_iterations=200)
        obj = pc.select_by_index(inl, invert=True)
        labels = np.asarray(obj.cluster_dbscan(eps=0.015, min_points=30))
        res_holder.update(n_in=len(pts), n_vox=len(pc.points), n_obj_pts=len(obj.points),
                          n_clusters=int(labels.max() + 1) if labels.size else 0)

    s = timed(run, args.iters, args.warmup, cuda=False)
    r = {"method": "voxel5mm+ransac_plane+dbscan (open3d CPU)", **res_holder, **s}
    print(json.dumps(r), flush=True)
    return [r]


def main():
    p = argparse.ArgumentParser()
    p.add_argument("mode", choices=["sam2", "geom"])
    p.add_argument("--sam2_ckpt", default="/bench/ckpt/sam2.1_hiera_tiny.pt")
    p.add_argument("--image", default="/bench/HGGD/images/demo_rgb.png")
    p.add_argument("--width", type=int, default=1280)
    p.add_argument("--height", type=int, default=720)
    p.add_argument("--iters", type=int, default=100)
    p.add_argument("--warmup", type=int, default=10)
    p.add_argument("--tag", default="")
    p.add_argument("--out", default="/bench/results")
    args = p.parse_args()
    logging.disable(logging.INFO)
    res = bench_sam2(args) if args.mode == "sam2" else bench_geom(args)
    os.makedirs(args.out, exist_ok=True)
    out = f"{args.out}/{args.mode}{'_' + args.tag if args.tag else ''}.json"
    json.dump({"args": vars(args), "results": res}, open(out, "w"), indent=1)
    print("wrote", out)


if __name__ == "__main__":
    main()
