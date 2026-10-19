#!/usr/bin/env python3
"""Résumé d'une phase : gigue des boucles stack (CSV) + tegrastats.

Usage : analyze.py <phase> [<phase> ...]   (dans /data/grasp_bench/results)
"""
import csv
import re
import sys

import numpy as np

R = "/data/grasp_bench/results"
SKIP_S = 15  # ignorer le démarrage (warmup cuRobo / TRT)


def loop_stats(path):
    rows = list(csv.DictReader(open(path)))
    if not rows:
        return None
    t0 = float(rows[0]["t"])
    rows = [r for r in rows if float(r["t"]) - t0 > SKIP_S]
    w = np.array([float(r["work_ms"]) for r in rows])
    p = np.array([float(r["period_ms"]) for r in rows])
    return {"n": len(rows), "work_mean": w.mean(), "work_p95": np.percentile(w, 95), "work_max": w.max(),
            "period_mean": p.mean(), "period_p95": np.percentile(p, 95), "period_max": p.max(),
            "jitter_std": p.std()}


def tegra_stats(path):
    ram, gpu, cpu, pw = [], [], [], []
    for line in open(path):
        m = re.search(r"RAM (\d+)/(\d+)MB", line)
        g = re.search(r"GR3D_FREQ (\d+)%", line)
        c = re.search(r"CPU \[([^\]]+)\]", line)
        v = re.search(r"VDD_GPU_SOC (\d+)mW", line)
        if not (m and g and c):
            continue
        ram.append(int(m.group(1)))
        tot = int(m.group(2))
        gpu.append(int(g.group(1)))
        cpu.append(np.mean([int(x.split("%")[0]) for x in c.group(1).split(",") if "%" in x]))
        if v:
            pw.append(int(v.group(1)))
    a = lambda x: np.array(x, dtype=float)
    return {"samples": len(ram), "ram_total_MB": tot, "ram_mean_MB": a(ram).mean(), "ram_max_MB": a(ram).max(),
            "gpu_util_mean": a(gpu).mean(), "gpu_util_p95": np.percentile(gpu, 95),
            "cpu_util_mean": a(cpu).mean(), "gpu_soc_mW_mean": a(pw).mean() if pw else None}


for ph in sys.argv[1:]:
    print(f"## {ph}")
    for w in ["rm", "curobo", "base"]:
        try:
            s = loop_stats(f"{R}/{ph}_{w}.csv")
        except FileNotFoundError:
            continue
        if s:
            print(f"  {w:7s} n={s['n']:5d} travail {s['work_mean']:7.1f}/{s['work_p95']:7.1f}/{s['work_max']:7.1f} ms"
                  f" | période {s['period_mean']:7.1f}/{s['period_p95']:7.1f}/{s['period_max']:7.1f} ms (moy/p95/max) std {s['jitter_std']:.1f}")
    try:
        t = tegra_stats(f"{R}/{ph}_tegra.log")
        print(f"  tegra  RAM {t['ram_mean_MB']:.0f} (max {t['ram_max_MB']:.0f}) / {t['ram_total_MB']} MB"
              f" | GPU {t['gpu_util_mean']:.0f}% (p95 {t['gpu_util_p95']:.0f}%) | CPU {t['cpu_util_mean']:.0f}%"
              f" | VDD_GPU_SOC {t['gpu_soc_mW_mean']:.0f} mW")
    except FileNotFoundError:
        pass
