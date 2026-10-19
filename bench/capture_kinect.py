#!/usr/bin/env python3
"""Capture Kinect (driver azure_kinect_ros_driver) -> fichiers consultables.

Pour chaque frame k : rgb_k.png, depth_mm_k.png (uint16, aligné RGB),
(topics /rgb/image_raw, /depth_to_rgb/image_raw, /rgb/camera_info)
depth_k.npy (m, float32), points_k.ply (nuage coloré, repère caméra RGB).
Plus : intrinsics.json, preview.jpg (RGB | profondeur colorisée).
"""
import argparse
import json
import os
import time

import cv2
import numpy as np
import open3d as o3d
import rclpy
from rclpy.node import Node
from rclpy.qos import QoSProfile, ReliabilityPolicy
from sensor_msgs.msg import CameraInfo, Image, PointCloud2


class Grab(Node):
    def __init__(self):
        super().__init__("grasp_bench_capture")
        self.last = {}
        for topic, typ in [("/rgb/image_raw", Image), ("/depth_to_rgb/image_raw", Image),
                           ("/rgb/camera_info", CameraInfo)]:
            qos = QoSProfile(depth=1, reliability=ReliabilityPolicy.RELIABLE)
            self.create_subscription(typ, topic, lambda m, t=topic: self.last.__setitem__(t, m), qos)


def img_to_np(m):
    if m.encoding in ("bgra8", "rgba8"):
        a = np.frombuffer(m.data, np.uint8).reshape(m.height, m.width, 4)
        return cv2.cvtColor(a, cv2.COLOR_BGRA2BGR if m.encoding == "bgra8" else cv2.COLOR_RGBA2BGR)
    if m.encoding == "16UC1":
        return np.frombuffer(m.data, np.uint16).reshape(m.height, m.width).copy()
    raise ValueError(m.encoding)


def depth_to_cloud(d_mm, bgr, K):
    """Nuage coloré (repère caméra RGB) depuis la profondeur alignée sur le RGB."""
    v, u = np.nonzero(d_mm > 0)
    z = d_mm[v, u].astype(np.float32) / 1000.0
    xyz = np.stack([(u - K[0, 2]) * z / K[0, 0], (v - K[1, 2]) * z / K[1, 1], z], 1)
    return xyz, bgr[v, u, ::-1].astype(np.float32) / 255.0


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--out", required=True)
    p.add_argument("--frames", type=int, default=5)
    p.add_argument("--gap", type=float, default=0.5)
    args = p.parse_args()
    os.makedirs(args.out, exist_ok=True)

    rclpy.init()
    n = Grab()
    need = {"/rgb/image_raw", "/depth_to_rgb/image_raw", "/rgb/camera_info"}
    t0 = time.time()
    while not need <= n.last.keys():
        rclpy.spin_once(n, timeout_sec=0.1)
        if time.time() - t0 > 60:
            raise SystemExit(f"topics manquants : {need - n.last.keys()}")

    ci = n.last["/rgb/camera_info"]
    json.dump({"K": np.array(ci.k).reshape(3, 3).tolist(), "D": list(ci.d), "distortion_model": ci.distortion_model,
               "width": ci.width, "height": ci.height, "frame_id": ci.header.frame_id,
               "depth_units": "depth_mm_*.png = mm (uint16), depth_*.npy = m",
               "points": "points_*.ply reconstruit depuis depth_to_rgb + K, repère caméra RGB (m)"},
              open(f"{args.out}/intrinsics.json", "w"), indent=1)

    for k in range(args.frames):
        n.last.clear()
        while not need - {"/rgb/camera_info"} <= n.last.keys():
            rclpy.spin_once(n, timeout_sec=0.1)
        rgb = img_to_np(n.last["/rgb/image_raw"])
        d = img_to_np(n.last["/depth_to_rgb/image_raw"])
        xyz, col = depth_to_cloud(d, rgb, np.array(ci.k).reshape(3, 3))
        cv2.imwrite(f"{args.out}/rgb_{k}.png", rgb)
        cv2.imwrite(f"{args.out}/depth_mm_{k}.png", d)
        np.save(f"{args.out}/depth_{k}.npy", d.astype(np.float32) / 1000.0)
        pc = o3d.geometry.PointCloud(o3d.utility.Vector3dVector(xyz.astype(np.float64)))
        if col is not None:
            pc.colors = o3d.utility.Vector3dVector(col.astype(np.float64))
        o3d.io.write_point_cloud(f"{args.out}/points_{k}.ply", pc)
        print(f"frame {k}: rgb {rgb.shape}, depth valid {(d > 0).mean():.2f}, points {len(xyz)}", flush=True)
        if k == 0:
            dc = cv2.applyColorMap(cv2.convertScaleAbs(np.clip(d, 0, 2500), alpha=255 / 2500), cv2.COLORMAP_TURBO)
            dc[d == 0] = 0
            cv2.imwrite(f"{args.out}/preview.jpg", np.hstack([rgb, dc]))
        time.sleep(args.gap)
    n.destroy_node()
    rclpy.shutdown()


if __name__ == "__main__":
    main()
