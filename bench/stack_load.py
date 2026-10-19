#!/usr/bin/env python3
"""Charge « stack » reproductible + sondes de gigue.

  curobo : boucle MPC cuRobo (optimize_action_sequence) cadencée comme le node
           m1013 (publication de l'horizon puis attente ~n_pts x 30 ms). Le but
           alterne de +-20 cm toutes les --switch itérations pour garder le
           solveur actif. Log : durée de solve et période réelle par cycle.
  rm     : cycle RM du reachability_pipeline CapaciNet : toutes les --rm_period s,
           UNet3D TensorRT (engine réel unet3d.trt) en lot de --rm_batch (grille 3x3).
           Log : durée d'inférence et période réelle.
  base   : sonde de la boucle de commande de la base (timer 10 Hz, CPU seul
           comme base_commander.publish_command). Log : période réelle.

Les deux écrivent un CSV (t, durée, période) dans --out ; arrêt par SIGTERM.
"""
import argparse
import csv
import signal
import time

import torch

STOP = False


def _stop(*_):
    global STOP
    STOP = True


def run_curobo(args, w):
    from curobo.model_predictive_control import ModelPredictiveControl, ModelPredictiveControlCfg
    from curobo.types import GoalToolPose, JointState

    cfg = ModelPredictiveControlCfg.create(
        robot="franka.yml",
        scene_model="collision_table.yml",
        use_cuda_graph=True,
        optimization_dt=0.025,
        interpolation_steps=4,
    )
    mpc = ModelPredictiveControl(cfg)
    state = JointState.from_position(
        mpc.default_joint_position.clone().unsqueeze(0), joint_names=mpc.joint_names
    )
    state.velocity = torch.zeros_like(state.position)
    state.acceleration = torch.zeros_like(state.position)
    mpc.setup(state)
    base_goal = mpc.compute_kinematics(state).tool_poses.to_dict()
    link = mpc.tool_frames[0]

    def set_goal(sign):
        g = {k: v.clone() for k, v in base_goal.items()}
        g[link].position[..., 1] += 0.2 * sign
        mpc.update_goal_tool_poses(
            GoalToolPose.from_poses(g, ordered_tool_frames=mpc.tool_frames, num_goalset=1),
            run_ik=True,
        )

    set_goal(1)
    i, sign, last = 0, 1, time.perf_counter()
    while not STOP:
        t0 = time.perf_counter()
        res = mpc.optimize_action_sequence(state)
        torch.cuda.synchronize()
        solve = time.perf_counter() - t0
        seq = res.action_sequence
        if seq is not None and seq.position.shape[1] > 0:
            state = JointState.from_position(seq.position[:, -1, :].clone(), joint_names=mpc.joint_names)
            state.velocity = seq.velocity[:, -1, :]
            state.acceleration = seq.acceleration[:, -1, :]
        w.writerow([f"{t0:.6f}", f"{solve*1e3:.3f}", f"{(t0-last)*1e3:.3f}"])
        last = t0
        i += 1
        if i % args.switch == 0:
            sign = -sign
            set_goal(sign)
        # gate de republication du node réel : attendre le drainage de l'horizon
        time.sleep(max(0.0, args.period - (time.perf_counter() - t0)))


def run_rm(args, w):
    import tensorrt as trt

    rt = trt.Runtime(trt.Logger(trt.Logger.WARNING))
    eng = rt.deserialize_cuda_engine(open(args.rm_engine, "rb").read())
    ctx = eng.create_execution_context()
    i_name, o_name = eng.get_tensor_name(0), eng.get_tensor_name(1)
    x = (torch.rand(args.rm_batch, 1, 128, 128, 128, device="cuda") > 0.9).float()
    ctx.set_input_shape(i_name, tuple(x.shape))
    y = torch.empty(tuple(ctx.get_tensor_shape(o_name)), device="cuda")
    ctx.set_tensor_address(i_name, x.data_ptr())
    ctx.set_tensor_address(o_name, y.data_ptr())
    stream = torch.cuda.Stream()
    nxt = time.perf_counter()
    last = nxt
    while not STOP:
        t0 = time.perf_counter()
        ctx.execute_async_v3(stream.cuda_stream)
        stream.synchronize()
        _ = y.mean().item()
        work = time.perf_counter() - t0
        w.writerow([f"{t0:.6f}", f"{work*1e3:.3f}", f"{(t0-last)*1e3:.3f}"])
        last = t0
        nxt += args.rm_period
        time.sleep(max(0.0, nxt - time.perf_counter()))


def run_base(args, w):
    import numpy as np

    rm = np.random.rand(3, 3).astype(np.float32)
    period = 1.0 / args.base_hz
    nxt = time.perf_counter()
    last = nxt
    while not STOP:
        t0 = time.perf_counter()
        _ = np.gradient(rm)
        work = time.perf_counter() - t0
        w.writerow([f"{t0:.6f}", f"{work*1e3:.3f}", f"{(t0-last)*1e3:.3f}"])
        last = t0
        nxt += period
        time.sleep(max(0.0, nxt - time.perf_counter()))


def main():
    p = argparse.ArgumentParser()
    p.add_argument("what", choices=["curobo", "rm", "base"])
    p.add_argument("--out", required=True)
    p.add_argument("--period", type=float, default=0.24)
    p.add_argument("--switch", type=int, default=40)
    p.add_argument("--base_hz", type=float, default=10.0)
    p.add_argument("--rm_engine", default="/bench/rm/unet3d.trt")
    p.add_argument("--rm_batch", type=int, default=9)
    p.add_argument("--rm_period", type=float, default=1.0)
    args = p.parse_args()
    signal.signal(signal.SIGTERM, _stop)
    signal.signal(signal.SIGINT, _stop)
    with open(args.out, "w", newline="", buffering=1) as f:
        w = csv.writer(f)
        w.writerow(["t", "work_ms", "period_ms"])
        {"curobo": run_curobo, "rm": run_rm, "base": run_base}[args.what](args, w)


if __name__ == "__main__":
    main()
