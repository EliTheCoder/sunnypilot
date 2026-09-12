"""
Plant model for stock (non-adaptive) cruise control.

The car's stock cruise controller is itself a closed-loop speed regulator. We do not
control it directly -- we only move its setpoint, in whole mph steps. This module
models the response from setpoint to longitudinal acceleration so a planner can
reason about what a button press will actually do.

Structure is deliberately physical rather than a black box: a deadtime, a saturated
proportional law on speed error, a first-order actuator lag, and a grade term. Every
parameter means something, which keeps it honest outside the data it was fit on and
makes it hard for a planner to exploit.

    e(t)  = sp(t - Td) - v(t) - offset
    u(t)  = clip(K * e(t), a_min, a_max)
    a'(t) = (u(t) - a(t)) / tau  +  k_grade * sin(pitch)

Fit by simulation-error minimization (not one-step) because the planner consumes
multi-step rollouts.
"""
from dataclasses import dataclass, asdict, fields

import numpy as np

G = 9.81


@dataclass
class PlantParams:
  """
  Defaults are FITTED to a 2019 Kia Optima non-SCC, from 1007s of engaged cruise
  across 26 windows over two drives (28-85mph, grades +/-2.4deg), by
  simulation-error minimization. They are not
  hand-chosen priors -- refit with tools/fit.py for a different car, and re-check
  the horizon in MpcConfig if the deadtime comes out much longer, since dithering
  needs usable planning window left after the delay.

  Open-loop speed prediction on held-out windows: 0.20mph rms @1s, 0.34 @2s,
  0.51 @4s, 0.61 @8s (vs 0.37/0.70/1.23/1.96 for assuming constant speed).
  """
  deadtime: float = 0.8000   # s, setpoint change -> measurable response
  tau: float = 1.1060        # s, first-order accel lag
  gain: float = 0.3226       # 1/s, accel per m/s of speed error
  offset: float = 0.8781     # m/s, setpoint reads high vs vEgo (speedo calibration)
  k_grade: float = -0.8811    # m/s^2 per unit sin(pitch)
  # Authority is strongly speed dependent: throttle authority falls with speed
  # while coast-down authority grows with it (aero drag). A single pair of
  # constants does not generalize across the speed range -- these are affine in v.
  a_min0: float = -0.0500    # m/s^2 at v=0, coast-down authority
  a_min_v: float = -0.01464  # m/s^2 per m/s
  a_max0: float = 0.7073    # m/s^2 at v=0, throttle authority
  a_max_v: float = -0.01018  # m/s^2 per m/s

  def a_min_at(self, v):
    return np.minimum(self.a_min0 + self.a_min_v * v, -0.02)

  def a_max_at(self, v):
    return np.maximum(self.a_max0 + self.a_max_v * v, 0.02)

  def as_vector(self) -> np.ndarray:
    return np.array([getattr(self, f.name) for f in fields(self)], dtype=np.float64)

  @staticmethod
  def from_vector(x: np.ndarray) -> "PlantParams":
    return PlantParams(**{f.name: float(v) for f, v in zip(fields(PlantParams), x, strict=True)})

  def to_dict(self) -> dict:
    return asdict(self)


# (lo, hi) search bounds, same order as fields
# (lo, hi) search bounds, same order as fields.
# deadtime is bounded near its directly measured value (median 1.48s over 40 clean
# setpoint steps). Left free it collapses toward zero and is absorbed into tau --
# the two are badly identifiable from this data, and a wrong split ruins the
# planner's multi-step prediction even when one-step error looks fine.
BOUNDS = np.array([
  (0.80, 2.50),    # deadtime
  (0.05, 3.00),    # tau
  (0.02, 1.50),    # gain
  (0.00, 2.00),    # offset
  (-15.0, 0.0),    # k_grade
  (-2.00, -0.05),  # a_min0
  (-0.05, 0.05),   # a_min_v
  (0.05, 2.50),    # a_max0
  (-0.05, 0.05),   # a_max_v
])


def simulate_accel(p: PlantParams, sp: np.ndarray, v: np.ndarray,
                   pitch: np.ndarray, dt: float, a0: float = 0.0) -> np.ndarray:
  """Open-loop accel prediction given the *measured* speed trace (for fitting)."""
  n = len(sp)
  d = int(round(p.deadtime / dt))
  # setpoint as seen by the controller, delayed
  spd = np.empty(n)
  if d > 0:
    spd[:d] = sp[0]
    spd[d:] = sp[:n - d]
  else:
    spd[:] = sp
  e = spd - v - p.offset
  u = np.clip(p.gain * e, p.a_min_at(v), p.a_max_at(v)) + p.k_grade * np.sin(pitch)
  a = np.empty(n)
  acc = a0
  alpha = dt / max(p.tau, 1e-3)
  for i in range(n):
    acc += alpha * (u[i] - acc)
    a[i] = acc
  return a


def rollout(p: PlantParams, sp_seq: np.ndarray, v0: float, a0: float,
            pitch: float, dt: float, sp_hist: np.ndarray | None = None) -> tuple[np.ndarray, np.ndarray]:
  """
  Closed-loop rollout used by the planner: integrate speed forward under a
  planned setpoint sequence. `sp_hist` is the recent setpoint history needed to
  fill the deadtime buffer (oldest first); if omitted, sp_seq[0] is assumed.
  """
  n = len(sp_seq)
  d = int(round(p.deadtime / dt))
  if sp_hist is None or len(sp_hist) < d:
    pad = np.full(max(d, 0), sp_seq[0] if n else 0.0)
  else:
    pad = sp_hist[-d:] if d > 0 else np.empty(0)
  spd = np.concatenate([pad, sp_seq])[:n] if d > 0 else sp_seq

  v = np.empty(n)
  a = np.empty(n)
  vv, acc = v0, a0
  alpha = dt / max(p.tau, 1e-3)
  gterm = p.k_grade * np.sin(pitch)
  for i in range(n):
    e = spd[i] - vv - p.offset
    u = min(max(p.gain * e, p.a_min_at(vv)), p.a_max_at(vv)) + gterm
    acc += alpha * (u - acc)
    vv += acc * dt
    v[i] = vv
    a[i] = acc
  return v, a
