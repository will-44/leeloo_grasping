#!/usr/bin/env python3
"""Test d'utilisation : capture Kinect -> SAM2 (boîte) -> GraspGenX 2F-85 -> filtre collision.

Pas de mouvement robot. Repère « table » : plan dominant (RANSAC) = z vers le
haut (GraspMoE suppose la gravité selon -z). Sorties dans --out :
  masks.png        masques SAM2 sur le RGB
  grasps.png       top-N prises projetées sur le RGB (couleur = score)
  scene.glb        nuage de scène + pinces (top-N par objet), ouvrable dans un viewer 3D
  <obj>_grasps.npz poses 4x4 (repère caméra RGB et repère table), scores, branche
  summary.json     comptes et temps par étape
"""
import argparse
import json
import logging
import os
import time

import cv2
import numpy as np
import open3d as o3d
import torch
import trimesh

from graspgenx.grasp_server import GraspGenXSampler
from graspgenx.samplers import run_planner_on_object
from graspgenx.utils.checkpoint_io import load_model_cfg
from graspgenx.utils.collision_filter import filter_colliding_grasps

REPO = "/bench/GraspGenX"
CKPT = f"{REPO}/ext/graspgenx_checkpoints/release"


class Timer:
    def __init__(self):
        self.t = {}

    def __call__(self, name):
        tm = self

        class _C:
            def __enter__(s):
                torch.cuda.synchronize()
                s.t0 = time.perf_counter()

            def __exit__(s, *a):
                torch.cuda.synchronize()
                tm.t[name] = (time.perf_counter() - s.t0) * 1000.0
        return _C()


def backproject(depth, K, mask=None):
    m = depth > 0.1
    if mask is not None:
        m &= mask
    v, u = np.nonzero(m)
    z = depth[v, u]
    return np.stack([(u - K[0, 2]) * z / K[0, 0], (v - K[1, 2]) * z / K[1, 1], z], 1).astype(np.float32), (v, u)


def table_frame(scene_cam):
    """T_table_cam : repère z = normale du plan dominant (orientée vers la caméra)."""
    pc = o3d.geometry.PointCloud(o3d.utility.Vector3dVector(scene_cam.astype(np.float64))).voxel_down_sample(0.01)
    (a, b, c, d), inl = pc.segment_plane(0.01, 3, 500)
    n = np.array([a, b, c])
    if np.dot(n, -np.asarray(pc.points)[inl].mean(0)) < 0:  # normale vers la caméra
        n, d = -n, -d
    z = n / np.linalg.norm(n)
    x = np.cross([0, 1, 0], z) if abs(z[1]) < 0.9 else np.cross([1, 0, 0], z)
    x /= np.linalg.norm(x)
    y = np.cross(z, x)
    origin = np.asarray(pc.points)[inl].mean(0)
    T_cam_table = np.eye(4)
    T_cam_table[:3, :3] = np.stack([x, y, z], 1)
    T_cam_table[:3, 3] = origin
    return np.linalg.inv(T_cam_table), len(inl)


def tf(T, p):
    return p @ T[:3, :3].T + T[:3, 3]


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--capture", required=True)
    p.add_argument("--frame", type=int, default=0)
    p.add_argument("--obj", action="append", required=True, help="nom:x0,y0,x1,y1 (pixels RGB)")
    p.add_argument("--planner", default="graspmoe", choices=["graspmoe", "diffusion"])
    p.add_argument("--num_grasps", type=int, default=200)
    p.add_argument("--threshold", type=float, default=0.7)
    p.add_argument("--max_obj_points", type=int, default=2000)
    p.add_argument("--collision_threshold", type=float, default=0.01)
    p.add_argument("--topn", type=int, default=10)
    p.add_argument("--out", required=True)
    args = p.parse_args()
    logging.disable(logging.INFO)
    os.makedirs(args.out, exist_ok=True)
    rng = np.random.default_rng(0)
    T = Timer()

    rgb = cv2.imread(f"{args.capture}/rgb_{args.frame}.png")
    depth = np.load(f"{args.capture}/depth_{args.frame}.npy")
    K = np.array(json.load(open(f"{args.capture}/intrinsics.json"))["K"])
    objs = []
    for o in args.obj:
        name, box = o.split(":")
        objs.append((name, np.array([float(x) for x in box.split(",")])))

    # --- modèles
    from sam2.build_sam import build_sam2
    from sam2.sam2_image_predictor import SAM2ImagePredictor

    sam = SAM2ImagePredictor(build_sam2("configs/sam2.1/sam2.1_hiera_t.yaml", "/bench/ckpt/sam2.1_hiera_tiny.pt", device="cuda"))
    cfg = load_model_cfg(f"{CKPT}/gen", f"{CKPT}/dis", None, None)
    sampler = GraspGenXSampler(cfg, "robotiq_2f_85", assets_dir=f"{REPO}/assets")
    gripper = sampler.get_gripper_info()
    grip_pts = np.asarray(trimesh.sample.sample_surface(gripper.collision_mesh, 2000)[0], dtype=np.float32)

    # warmup (kernels, cudnn) pour des temps représentatifs
    with torch.inference_mode(), torch.autocast("cuda", dtype=torch.bfloat16):
        sam.set_image(rgb[..., ::-1].copy())
        sam.predict(box=objs[0][1], multimask_output=False)
    run_planner_on_object(np.random.rand(500, 3).astype(np.float32) * 0.05, sampler, planner=args.planner,
                          grasp_threshold=args.threshold, num_grasps=args.num_grasps)

    # --- scène + repère table
    with T("scene_table_ransac"):
        scene_cam, _ = backproject(depth, K, depth < 2.5)
        T_table_cam, n_inl = table_frame(scene_cam)
    scene_tab = tf(T_table_cam, scene_cam)

    # --- SAM2
    with T("sam2_encoder"), torch.inference_mode(), torch.autocast("cuda", dtype=torch.bfloat16):
        sam.set_image(rgb[..., ::-1].copy())
    masks, overlay = {}, rgb.copy()
    palette = [(0, 200, 255), (255, 80, 0), (0, 255, 0), (255, 0, 255), (0, 0, 255)]
    for i, (name, box) in enumerate(objs):
        with T(f"sam2_decoder_{name}"), torch.inference_mode(), torch.autocast("cuda", dtype=torch.bfloat16):
            m, s, _ = sam.predict(box=box, multimask_output=False)
        masks[name] = m[0] > 0
        col = palette[i % len(palette)]
        overlay[masks[name]] = (0.5 * overlay[masks[name]] + 0.5 * np.array(col)).astype(np.uint8)
        x0, y0, x1, y1 = box.astype(int)
        cv2.rectangle(overlay, (x0, y0), (x1, y1), col, 1)
        cv2.putText(overlay, f"{name} {float(s[0]):.2f}", (x0, max(10, y0 - 3)), 0, 0.4, col, 1)
    cv2.imwrite(f"{args.out}/masks.png", overlay)

    all_masks = np.zeros(depth.shape, bool)
    for m in masks.values():
        all_masks |= m

    summary = {"capture": args.capture, "frame": args.frame, "planner": args.planner,
               "num_grasps": args.num_grasps, "threshold": args.threshold,
               "table_inliers_1cm_vox": n_inl, "objects": {}}
    glb = trimesh.Scene()
    sub = scene_tab[rng.choice(len(scene_tab), min(150000, len(scene_tab)), replace=False)]
    glb.add_geometry(trimesh.PointCloud(sub, colors=np.tile([150, 150, 150, 255], (len(sub), 1))), node_name="scene")
    vis = rgb.copy()
    cmap = lambda c: tuple(int(x) for x in cv2.applyColorMap(np.uint8([[255 * c]]), cv2.COLORMAP_JET)[0, 0])

    for i, (name, box) in enumerate(objs):
        info = {}
        mk = cv2.erode(masks[name].astype(np.uint8), np.ones((3, 3), np.uint8)).astype(bool)
        obj_cam, _ = backproject(depth, K, mk)
        info["mask_px"] = int(masks[name].sum())
        info["obj_points_raw"] = int(len(obj_cam))
        if len(obj_cam) < 30:
            info["status"] = "trop peu de points de profondeur"
            summary["objects"][name] = info
            continue
        # garder la composante en profondeur dominante (bords / fond derrière l'objet)
        zmed = np.median(obj_cam[:, 2])
        obj_cam = obj_cam[np.abs(obj_cam[:, 2] - zmed) < 0.08]
        if len(obj_cam) > args.max_obj_points:
            obj_cam = obj_cam[rng.choice(len(obj_cam), args.max_obj_points, replace=False)]
        obj_tab = tf(T_table_cam, obj_cam)
        info["obj_points_used"] = int(len(obj_tab))
        info["obj_extent_m"] = (obj_tab.max(0) - obj_tab.min(0)).round(3).tolist()

        with T(f"graspgenx_{name}"):
            grasps, conf, tags, _ = run_planner_on_object(
                obj_tab, sampler, planner=args.planner, grasp_threshold=args.threshold,
                num_grasps=args.num_grasps, topk_num_grasps=-1)
        grasps = np.asarray(grasps.cpu() if torch.is_tensor(grasps) else grasps, dtype=np.float32).reshape(-1, 4, 4)
        conf = np.asarray(conf.cpu() if torch.is_tensor(conf) else conf, dtype=np.float32).reshape(-1)
        tags = list(tags) if tags else ["diff"] * len(conf)
        info["grasps_above_thr"] = int(len(conf))
        info["by_branch"] = {"diff": tags.count("diff"), "obb": tags.count("obb")}

        # collision avec la scène hors objet (table, étagère, autres objets)
        others_cam, _ = backproject(depth, K, (depth < 2.5) & ~cv2.dilate(masks[name].astype(np.uint8), np.ones((5, 5), np.uint8)).astype(bool))
        others = tf(T_table_cam, others_cam)
        near = np.linalg.norm(others - obj_tab.mean(0), axis=1) < 0.4
        others = others[near]
        if len(others) > 8192:
            others = others[rng.choice(len(others), 8192, replace=False)]
        if len(conf):
            with T(f"collision_{name}"):
                free = filter_colliding_grasps(scene_pc=others, grasp_poses=grasps,
                                               collision_threshold=args.collision_threshold,
                                               gripper_surface_points=grip_pts)
            grasps, conf = grasps[free], conf[free]
            tags = [t for t, k in zip(tags, free) if k]
        info["collision_free"] = int(len(conf))
        info["score_max"] = float(conf.max()) if len(conf) else None
        info["score_median"] = float(np.median(conf)) if len(conf) else None

        order = np.argsort(-conf)
        grasps, conf, tags = grasps[order], conf[order], [tags[j] for j in order]
        T_cam_table = np.linalg.inv(T_table_cam)
        np.savez(f"{args.out}/{name}_grasps.npz", grasps_table=grasps, grasps_cam=T_cam_table @ grasps,
                 scores=conf, branch=np.array(tags), T_table_cam=T_table_cam, obj_points_table=obj_tab)

        # visualisations
        glb.add_geometry(trimesh.PointCloud(obj_tab, colors=np.tile([255, 200, 0, 255], (len(obj_tab), 1))), node_name=f"{name}_pc")
        for j in range(min(args.topn, len(conf))):
            mesh = gripper.collision_mesh.copy()
            mesh.apply_transform(grasps[j])
            c = cmap(conf[j])
            mesh.visual.face_colors = [c[2], c[1], c[0], 160]
            glb.add_geometry(mesh, node_name=f"{name}_grasp{j}")
            pts = tf(T_cam_table, tf(grasps[j], grip_pts[::8]))
            uv = (pts[:, :2] / pts[:, 2:3]) * [K[0, 0], K[1, 1]] + [K[0, 2], K[1, 2]]
            for u, v in uv.astype(int):
                if 0 <= u < vis.shape[1] and 0 <= v < vis.shape[0]:
                    vis[v, u] = c
        x0, y0, _, _ = box.astype(int)
        cv2.putText(vis, f"{name}: {info['collision_free']} prises", (x0, max(10, y0 - 3)), 0, 0.4, (0, 0, 255), 1)
        summary["objects"][name] = info

    cv2.imwrite(f"{args.out}/grasps.png", vis)
    glb.export(f"{args.out}/scene.glb")
    summary["timings_ms"] = {k: round(v, 1) for k, v in T.t.items()}
    json.dump(summary, open(f"{args.out}/summary.json", "w"), indent=1)
    print(json.dumps(summary, indent=1))


if __name__ == "__main__":
    main()
