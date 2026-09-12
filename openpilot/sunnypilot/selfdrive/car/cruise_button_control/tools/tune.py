#!/usr/bin/env python3
"""
Sweep the press-cost weight to expose the tracking-vs-button-spam tradeoff.

Tracking error is cheap to buy with more presses, but every press flickers the
cluster setpoint and exercises the BCM, so the useful operating point is the knee
of this curve, not the minimum error.

usage: tune.py <params.json>
"""
import json
import sys

import numpy as np

from openpilot.sunnypilot.selfdrive.car.cruise_button_control.mpc import (
  MPH_TO_MS, MS_TO_MPH, ButtonMpc, MpcConfig)
from openpilot.sunnypilot.selfdrive.car.cruise_button_control.plant import PlantParams
from openpilot.sunnypilot.selfdrive.car.cruise_button_control.tools.closed_loop import run


def main():
  p = PlantParams(**json.load(open(sys.argv[1])))
  print(f"{'w_press':>8} {'hold@.5 rms':>12} {'pr/min':>7} {'hold@.0 rms':>12} {'pr/min':>7} " +
        f"{'track rms':>10} {'pr/min':>7}")
  for w in (0.02, 0.05, 0.12, 0.30, 0.80, 2.00):
    cfg = MpcConfig(w_press=w)
    mpc = ButtonMpc(p, cfg)
    dt = cfg.dt
    row = []
    for i, (fn, seconds) in enumerate((
        (lambda t: (74.5 * MPH_TO_MS) - p.offset, 90.0),
        (lambda t: (74.0 * MPH_TO_MS) - p.offset, 90.0),
        (lambda t: (70.0 * MPH_TO_MS) - p.offset + 1.2 * np.sin(2 * np.pi * t / 30.0), 120.0),
    )):
      V, SP, DES, PR = run(mpc, p, fn, duration=seconds, seed=i)
      k = int(20.0 / dt)
      err = (V[k:] - DES[k:]) * MS_TO_MPH
      row.append((np.sqrt((err ** 2).mean()), PR.sum() / (len(V) * dt / 60.0)))
    print(f"{w:8.2f} {row[0][0]:12.3f} {row[0][1]:7.1f} {row[1][0]:12.3f} {row[1][1]:7.1f} " +
          f"{row[2][0]:10.3f} {row[2][1]:7.1f}")


if __name__ == "__main__":
  main()
