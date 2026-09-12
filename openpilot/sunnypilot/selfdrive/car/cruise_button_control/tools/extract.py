#!/usr/bin/env python3
"""
Extract a plant-identification dataset from openpilot logs.

Pulls contiguous windows where stock cruise is engaged and the driver is off the
pedals, which is the only regime where the setpoint -> speed response is clean.

usage: extract.py <route_prefix> [out.npz]
"""
import glob
import sys

import numpy as np

from openpilot.tools.lib.logreader import LogReader

ROOT = "/data/media/0/realdata"
MIN_WINDOW_S = 3.0


def load_segment(path: str):
  rows, pose = [], []
  for m in LogReader(path):
    w = m.which()
    if w == "carState":
      c = m.carState
      rows.append((m.logMonoTime * 1e-9, c.vEgo, c.aEgo, float(c.cruiseState.enabled),
                   c.cruiseState.speed, float(c.gasPressed), float(c.brakePressed)))
    elif w == "liveLocationKalman":
      o = m.liveLocationKalman.orientationNED
      if o.valid:
        pose.append((m.logMonoTime * 1e-9, o.value[1]))
  if not rows:
    return None
  a = np.array(rows, dtype=np.float64)
  if pose:
    p = np.array(pose, dtype=np.float64)
    pitch = np.interp(a[:, 0], p[:, 0], p[:, 1])
  else:
    pitch = np.zeros(len(a))
  return a, pitch


def windows(route: str):
  paths = sorted(glob.glob(f"{ROOT}/{route}--*/rlog.zst"),
                 key=lambda p: int(p.split("--")[-1].split("/")[0]))
  out = []
  for path in paths:
    seg = int(path.split("--")[-1].split("/")[0])
    try:
      loaded = load_segment(path)
    except Exception as e:
      print(f"  seg {seg}: read error {e}")
      continue
    if loaded is None:
      continue
    a, pitch = loaded
    t, v, acc, en, sp, gas, brk = a.T
    ok = (en > 0) & (gas == 0) & (brk == 0) & (sp > 1)
    # split into contiguous runs
    idx = np.where(ok)[0]
    if not len(idx):
      continue
    splits = np.where(np.diff(idx) > 1)[0] + 1
    for run in np.split(idx, splits):
      if len(run) < 2:
        continue
      duration_s = t[run[-1]] - t[run[0]]
      if duration_s < MIN_WINDOW_S:
        continue
      out.append({"seg": seg, "t": t[run], "v": v[run], "a": acc[run],
                  "sp": sp[run], "pitch": pitch[run]})
  return out


def main():
  route = sys.argv[1]
  out_path = sys.argv[2] if len(sys.argv) > 2 else "/tmp/cbc_dataset.npz"
  ws = windows(route)
  if not ws:
    print("no usable windows")
    return
  total = sum(len(w["t"]) for w in ws)
  duration_s = sum(w["t"][-1] - w["t"][0] for w in ws)
  print(f"windows={len(ws)}  samples={total}  duration={duration_s:.0f}s")
  for w in ws:
    print(f"  seg {w['seg']:3d}  {w['t'][-1]-w['t'][0]:6.1f}s  " +
          f"v={w['v'].mean()*2.23694:5.1f}mph  sp_changes={int((np.abs(np.diff(w['sp']))>0.1).sum())}")
  flat = {}
  for k in ("t", "v", "a", "sp", "pitch"):
    flat[k] = np.concatenate([w[k] for w in ws])
  flat["wid"] = np.concatenate([np.full(len(w["t"]), i) for i, w in enumerate(ws)])
  flat["seg"] = np.concatenate([np.full(len(w["t"]), w["seg"]) for w in ws])
  np.savez_compressed(out_path, t=flat["t"], v=flat["v"], a=flat["a"], sp=flat["sp"],
                      pitch=flat["pitch"], wid=flat["wid"], seg=flat["seg"])
  print(f"\nwrote {out_path}")


if __name__ == "__main__":
  main()
