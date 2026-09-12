#!/usr/bin/env python3
"""
Fit PlantParams to an extracted dataset by simulation-error minimization.

usage: fit.py <dataset.npz> [--holdout N] [--out params.json]
"""
import argparse
import json

import numpy as np

from openpilot.sunnypilot.selfdrive.car.cruise_button_control.plant import (
  BOUNDS, PlantParams, simulate_accel)


def load(path):
  """
  liveLocationKalman orientationNED pitch is *device* pitch: it carries the camera
  mount angle and the vehicle's own suspension pitch, not just road grade. The
  mount angle is a constant bias, so de-mean per window to recover the part that
  actually varies with the road. Without this the grade coefficient fits to ~0
  and the term is silently useless.
  """
  d = np.load(path)
  wins = []
  for w in np.unique(d["wid"]):
    m = d["wid"] == w
    t = d["t"][m]
    pitch = d["pitch"][m]
    wins.append({"t": t, "v": d["v"][m], "a": d["a"][m], "sp": d["sp"][m],
                 "pitch": pitch - pitch.mean(), "seg": int(d["seg"][m][0]),
                 "dt": float(np.median(np.diff(t)))})
  return wins


def loss(x, wins):
  p = PlantParams.from_vector(x)
  if p.a_min0 >= p.a_max0 or p.tau <= 0 or p.deadtime < 0:
    return 1e9
  se = 0.0
  n = 0
  for w in wins:
    pred = simulate_accel(p, w["sp"], w["v"], w["pitch"], w["dt"], a0=w["a"][0])
    se += float(np.sum((pred - w["a"]) ** 2))
    n += len(pred)
  return se / max(n, 1)


def coordinate_descent(x0, wins, iters=6, grid=11, verbose=True):
  x = x0.copy()
  best = loss(x, wins)
  lo, hi = BOUNDS[:, 0].copy(), BOUNDS[:, 1].copy()
  span = (hi - lo) / 2.0
  for it in range(iters):
    improved = False
    for j in range(len(x)):
      cand = np.linspace(max(lo[j], x[j] - span[j]), min(hi[j], x[j] + span[j]), grid)
      vals = []
      for c in cand:
        xx = x.copy()
        xx[j] = c
        vals.append(loss(xx, wins))
      k = int(np.argmin(vals))
      if vals[k] < best - 1e-12:
        best = vals[k]
        x[j] = cand[k]
        improved = True
    span *= 0.45
    if verbose:
      print(f"  iter {it}: rmse={np.sqrt(best):.5f} m/s^2")
    if not improved and it > 1:
      break
  return x, best


def report(p, wins, label):
  errs, base = [], []
  for w in wins:
    pred = simulate_accel(p, w["sp"], w["v"], w["pitch"], w["dt"], a0=w["a"][0])
    errs.append(pred - w["a"])
    base.append(w["a"] - w["a"].mean())
  e = np.concatenate(errs)
  b = np.concatenate(base)
  rmse = float(np.sqrt(np.mean(e ** 2)))
  r2 = 1.0 - float(np.sum(e ** 2) / np.sum(b ** 2))
  print(f"  {label:9s} rmse={rmse:.4f} m/s^2   R^2={r2:+.3f}   n={len(e)}")
  return rmse, r2


def main():
  ap = argparse.ArgumentParser()
  ap.add_argument("dataset")
  ap.add_argument("--holdout", type=int, default=3)
  ap.add_argument("--out", default=None)
  a = ap.parse_args()

  wins = load(a.dataset)
  print(f"windows={len(wins)} samples={sum(len(w['t']) for w in wins)}")
  # Hold out windows spanning the speed range, not simply the last N. Taking the
  # tail put every low-speed window in test and made the result a speed
  # extrapolation test rather than a generalization test.
  order = np.argsort([w["v"].mean() for w in wins])
  held = set(order[:: max(1, len(wins) // max(a.holdout, 1))][:a.holdout].tolist())
  train = [w for i, w in enumerate(wins) if i not in held]
  test = [w for i, w in enumerate(wins) if i in held]
  mph = 2.23694
  print(f"train={len(train)} windows  test={len(test)} windows " +
        f"(segs {[w['seg'] for w in test]}, mean speeds " +
        f"{[round(float(w['v'].mean()*mph)) for w in test]} mph)\n")

  x0 = PlantParams().as_vector()
  print("fitting (simulation-error, coordinate descent):")
  x, _ = coordinate_descent(x0, train, verbose=True)
  p = PlantParams.from_vector(x)

  print("\nfitted parameters:")
  for k, v in p.to_dict().items():
    print(f"  {k:10s} {v:+.4f}")

  print("\nperformance:")
  report(PlantParams(), train, "prior/tr")
  report(p, train, "fit/train")
  report(p, test, "fit/TEST")

  if a.out:
    with open(a.out, "w") as f:
      json.dump(p.to_dict(), f, indent=2)
    print(f"\nwrote {a.out}")


if __name__ == "__main__":
  main()
