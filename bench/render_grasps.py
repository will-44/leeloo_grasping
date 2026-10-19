#!/usr/bin/env python3
"""Rendu matplotlib (repère table, z vers le haut) : nuage local + top-N pinces par objet.

Usage : render_grasps.py <dossier use_test> [--topn 3]
Écrit <dossier>/<obj>_top.png (vue de côté + vue de dessus) et affiche l'angle
d'approche par rapport à la verticale.
"""
import argparse
import glob
import os

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import trimesh

p = argparse.ArgumentParser()
p.add_argument("dir")
p.add_argument("--topn", type=int, default=3)
p.add_argument("--mesh", default="/bench/GraspGenX/ext/gripper_descriptions/gripper_descriptions/assets/x_grippers/robotiq_2f_85/coll_mesh.obj")
a = p.parse_args()

grip = trimesh.load(a.mesh, force="mesh")
edges = grip.vertices[grip.edges_unique[np.random.default_rng(0).choice(len(grip.edges_unique), min(600, len(grip.edges_unique)), replace=False)]]
cols = ["tab:red", "tab:green", "tab:blue", "tab:purple", "tab:orange"]

for f in sorted(glob.glob(f"{a.dir}/*_grasps.npz")):
    name = os.path.basename(f).replace("_grasps.npz", "")
    d = np.load(f)
    G, S, pts = d["grasps_table"], d["scores"], d["obj_points_table"]
    fig = plt.figure(figsize=(12, 6))
    for k, (elev, azim, title) in enumerate([(10, -60, "côté"), (89, -90, "dessus")]):
        ax = fig.add_subplot(1, 2, k + 1, projection="3d")
        ax.scatter(*pts.T, s=2, c="gold")
        for j in range(min(a.topn, len(S))):
            e = edges @ G[j][:3, :3].T + G[j][:3, 3]
            for seg in e:
                ax.plot(*seg.T, c=cols[j], lw=0.4)
            appr = G[j][:3, 2]
            ang = np.degrees(np.arccos(np.clip(-appr[2], -1, 1)))
            if k == 0:
                print(f"{name} #{j}: score {S[j]:.3f}, approche à {ang:.0f}° de la verticale descendante, "
                      f"pince en z={G[j][2, 3]:.3f} m (objet z {pts[:, 2].min():.3f}..{pts[:, 2].max():.3f})")
        c = pts.mean(0)
        r = 0.15
        ax.set_xlim(c[0] - r, c[0] + r); ax.set_ylim(c[1] - r, c[1] + r); ax.set_zlim(c[2] - r, c[2] + r)
        ax.view_init(elev, azim)
        ax.set_title(f"{name} — {title} (top {min(a.topn, len(S))}, scores {', '.join(f'{s:.2f}' for s in S[:a.topn])})", fontsize=9)
        ax.set_xlabel("x"); ax.set_ylabel("y"); ax.set_zlabel("z (haut)")
    fig.tight_layout()
    fig.savefig(f"{a.dir}/{name}_top.png", dpi=110)
    plt.close(fig)
