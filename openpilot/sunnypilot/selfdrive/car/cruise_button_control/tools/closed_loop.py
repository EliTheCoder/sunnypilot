#!/usr/bin/env python3
"""
Closed-loop evaluation of the button MPC against the identified plant.

Checks the things that actually decide whether this is drivable:
  - can it hold a speed that falls BETWEEN 1 mph increments (the dither question)
  - does it oscillate
  - how many button presses per minute does it cost
  - does it track a moving reference (lead-follow proxy)

usage: closed_loop.py <params.json>
"""
import json
import sys

import numpy as np

from openpilot.sunnypilot.selfdrive.car.cruise_button_control.mpc import (
  MPH_TO_MS, MS_TO_MPH, ButtonMpc, MpcConfig)
from openpilot.sunnypilot.selfdrive.car.cruise_button_control.plant import PlantParams, rollout


def run(mpc, p, v_des_fn, duration=90.0, v0=None, sp0=None, pitch=0.0, seed=0):
  mpc.rng = np.random.default_rng(seed)
  dt = mpc.cfg.dt
  n = int(duration / dt)
  v_des0 = v_des_fn(0.0)
  v = v0 if v0 is not None else v_des0
  a = 0.0
  sp = float(sp0 if sp0 is not None else round((v_des0 + p.offset) * MS_TO_MPH))
  hist = [sp] * (int(round(p.deadtime / dt)) + 1)
  last_press = -99.0

  V, SP, DES, PRESS = [], [], [], []
  for i in range(n):
    t = i * dt
    des = np.array([v_des_fn(t + k * dt) for k in range(mpc.cfg.n_steps)])
    act = 0
    if t - last_press >= mpc.cfg.min_press_interval:
      act, _ = mpc.plan(des, v, a, sp, pitch, np.array(hist))
      if act != 0:
        sp = float(np.clip(sp + act, mpc.cfg.sp_min_mph, mpc.cfg.sp_max_mph))
        last_press = t
    hist.append(sp)
    # advance true plant one tick
    vv, aa = rollout(p, np.array([sp * MPH_TO_MS]), v, a, pitch, dt,
                     sp_hist=np.array(hist[:-1]) * MPH_TO_MS)
    v, a = float(vv[0]), float(aa[0])
    V.append(v)
    SP.append(sp)
    DES.append(v_des_fn(t))
    PRESS.append(abs(act))
  return map(np.array, (V, SP, DES, PRESS))


def report(name, V, SP, DES, PRESS, dt, settle_s=20.0):
  k = int(settle_s / dt)
  err = (V[k:] - DES[k:]) * MS_TO_MPH
  presses = int(PRESS.sum())
  minutes = len(V) * dt / 60.0
  uniq = np.unique(SP[k:])
  # oscillation: zero crossings of the error signal per minute
  zc = int((np.diff(np.sign(err)) != 0).sum())
  print(f"\n{name}")
  print(f"  speed error (after {settle_s:.0f}s):  mean={err.mean():+.3f}  " +
        f"rms={np.sqrt((err**2).mean()):.3f}  max|e|={np.abs(err).max():.3f} mph")
  print(f"  setpoint values used: {[int(x) for x in uniq]}  " +
        f"({'DITHERING' if len(uniq) > 1 else 'static'})")
  print(f"  presses: {presses} total = {presses/minutes:.1f}/min")
  print(f"  error sign changes: {zc} ({zc/minutes:.1f}/min)")


def main():
  p = PlantParams(**json.load(open(sys.argv[1])))
  cfg = MpcConfig()
  mpc = ButtonMpc(p, cfg)
  dt = cfg.dt

  # 1. hold a target squarely between two mph increments
  tgt = (74.5 * MPH_TO_MS) - p.offset   # so steady-state v should be 74.5 mph-equivalent
  V, SP, DES, PR = run(mpc, p, lambda t: tgt, duration=90.0)
  report("hold speed BETWEEN increments (74.5 mph equivalent)", V, SP, DES, PR, dt)

  # 2. hold a target right on an increment (should not dither)
  tgt2 = (74.0 * MPH_TO_MS) - p.offset
  V, SP, DES, PR = run(mpc, p, lambda t: tgt2, duration=90.0, seed=1)
  report("hold speed ON an increment (74.0 mph equivalent)", V, SP, DES, PR, dt)

  # 3. moving reference: slow sinusoid, lead-follow proxy
  base = (70.0 * MPH_TO_MS) - p.offset
  V, SP, DES, PR = run(mpc, p, lambda t: base + 1.2 * np.sin(2 * np.pi * t / 30.0),
                       duration=120.0, seed=2)
  report("track moving reference (+/-2.7mph, 30s period)", V, SP, DES, PR, dt)


if __name__ == "__main__":
  main()
