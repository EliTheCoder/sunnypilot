#!/usr/bin/env python3
"""
Validate a fitted plant over the horizons the planner actually uses.

Instantaneous accel R^2 is a poor proxy: the planner consumes multi-step speed
rollouts, so this measures open-loop *speed* error at 1/2/4/8s from many start
points, against two baselines -- constant speed, and constant current accel.

usage: sim.py <dataset.npz> <params.json>
"""
import json
import sys

import numpy as np

from openpilot.sunnypilot.selfdrive.car.cruise_button_control.plant import PlantParams, rollout
from openpilot.sunnypilot.selfdrive.car.cruise_button_control.tools.fit import load

MPH = 2.23694
HORIZONS = (1.0, 2.0, 4.0, 8.0)


def main():
  wins = load(sys.argv[1])
  p = PlantParams(**json.load(open(sys.argv[2])))
  dt_ctrl = 0.05  # plan at 20Hz; data is 100Hz
  res = {h: {"model": [], "hold": [], "const_a": []} for h in HORIZONS}

  for w in wins:
    dt = w["dt"]
    step = max(1, int(round(dt_ctrl / dt)))
    n = len(w["t"])
    dmax = int(round(p.deadtime / dt_ctrl)) + 1
    for s in range(int(2.0 / dt), n - int(max(HORIZONS) / dt), int(1.0 / dt)):
      hist = w["sp"][max(0, s - dmax * step):s:step]
      for h in HORIZONS:
        k = int(h / dt)
        if s + k >= n:
          continue
        sp_seq = w["sp"][s:s + k:step]
        if len(sp_seq) < 2:
          continue
        pitch = float(np.mean(w["pitch"][s:s + k]))
        v_pred, _ = rollout(p, sp_seq, float(w["v"][s]), float(w["a"][s]), pitch, dt_ctrl,
                            sp_hist=hist if len(hist) else None)
        v_true = w["v"][s + k - 1]
        res[h]["model"].append(v_pred[-1] - v_true)
        res[h]["hold"].append(w["v"][s] - v_true)
        res[h]["const_a"].append(w["v"][s] + w["a"][s] * h - v_true)

  print("open-loop speed prediction error (n start points per horizon shown)\n")
  print(f"  {'horizon':>8}  {'model':>16}  {'hold-speed':>16}  {'const-accel':>16}   n")
  for h in HORIZONS:
    r = res[h]
    if not r["model"]:
      continue
    m = np.array(r["model"]) * MPH
    ho = np.array(r["hold"]) * MPH
    ca = np.array(r["const_a"]) * MPH
    print(f"  {h:6.0f}s  {np.sqrt((m**2).mean()):7.3f} mph rms  " +
          f"{np.sqrt((ho**2).mean()):7.3f} mph rms  " +
          f"{np.sqrt((ca**2).mean()):7.3f} mph rms   {len(m)}")
  print("\n  (model must beat hold-speed to be worth anything to the planner)")


if __name__ == "__main__":
  main()
