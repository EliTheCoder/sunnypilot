"""
Discrete MPC over cruise button presses.

The actuator is a quantized integer setpoint in whole mph, moved one step at a time,
behind a deadtime. That rules out PID: by the time speed error is observable the
press that would have corrected it is already ~1.5s stale, so any reactive loop
either oscillates or has to be detuned into uselessness.

Instead we plan. Actions are {-1, 0, +1} mph per decision tick; candidate sequences
are rolled out through the identified plant and scored against the desired speed
trajectory. Because the deadtime is *inside* the model, the planner commands through
it rather than fighting it.

Sub-mph resolution is not special-cased. If holding 74.5 mph is best served by
dithering the setpoint between 74 and 75, the optimizer finds that on its own --
it is just another action sequence. This is why the desired speed must arrive
unrounded; rounding it upstream destroys the information that makes dither optimal.

Search is vectorized random shooting with CEM refinement: cheap, derivative-free,
and it respects the press-rate constraint exactly rather than by penalty.
"""
from dataclasses import dataclass

import numpy as np

from openpilot.sunnypilot.selfdrive.car.cruise_button_control.plant import PlantParams


@dataclass
class MpcConfig:
  # Defaults are sized for the target hardware, not for a laptop. On the comma the
  # solve is ~17x slower than on a dev machine: dt=0.05/384/3 measures ~6ms locally
  # but 100ms median and 184ms p95 on device, which misses a real-time deadline.
  # dt=0.10/256/2 measures ~33ms on device. Planning faster than the press rate
  # (5Hz, see min_press_interval) buys nothing anyway.
  dt: float = 0.10             # planner tick, s
  # Horizon is deliberately short. Longer is worse here, not better: with a fixed
  # sample budget a longer horizon adds decision variables without improving the
  # first action, and measured closed-loop tracking degrades (4s -> 0.12mph rms,
  # 8s -> 0.23mph rms on the same budget). The plant settles in ~3s, so 4s covers
  # everything that can affect the press being decided now.
  horizon: float = 4.0         # s
  # 0.5s decisions rather than 0.25s: it halves the number of decision variables,
  # which both shrinks the solve (32ms -> 16ms on device) and *improves* tracking,
  # because the same sample budget converges better on a smaller search space
  # (0.105mph rms holding between increments vs 0.128 at 0.25s).
  decision_dt: float = 0.50    # s between allowed setpoint changes
  min_press_interval: float = 0.20  # s, hardware accepts ~6.7Hz; stay under it
  # Iterations buy convergence far more cheaply than samples: at 256 samples the
  # run-to-run cost spread is 9.1% at 2 iters, 4.8% at 3, 2.7% at 4; going to 512
  # samples at 3 iters only matches 256 at 4 for 1.5x the work.
  # Sized for plannerd's 20Hz/50ms budget, shared with its own acados solve:
  # this config measures ~16ms on device (~32% duty). It must never go back into
  # selfdrived, which is 100Hz/10ms -- the original 32ms solve there was a 317%
  # overrun and produced "system lagging" alerts on the road.
  n_samples: int = 96
  n_elite: int = 16
  n_iters: int = 3
  w_speed: float = 1.0         # per (m/s)^2
  # Press cost sets the tracking-vs-button-spam tradeoff. Measured knee (see
  # tools/tune.py): 0.02 -> 0.06mph rms at 64 presses/min; 0.12 -> 0.16mph at 21/min;
  # 0.30 -> 0.35mph at 9/min; >=2.0 stops dithering entirely and sits half a step
  # off (0.50mph, the no-dither baseline). 0.12 keeps most of the sub-mph benefit at
  # a third of the presses. Raise it if the cluster flicker is distracting.
  w_press: float = 0.12        # per press; buys smoothness, costs tracking
  w_terminal: float = 2.0
  # The planner commands the CLUSTER setpoint, and the car settles at
  # (setpoint - offset), so the setpoint must be able to exceed the fastest speed
  # openpilot will ever ask for by at least the offset, plus room to dither above
  # it. openpilot caps v_cruise at V_CRUISE_MAX = 145kph = 90.1mph and the measured
  # offset is ~2.08mph, so holding 90.1mph needs a setpoint of ~92.2mph. A 90.0
  # ceiling silently made the top of the range unreachable.
  # test_ceiling_allows_holding_max_cruise_speed guards this.
  # The floor matches ICBM's get_minimum_set_speed() for imperial units.
  sp_min_mph: float = 20.0
  sp_max_mph: float = 95.0

  @property
  def n_steps(self) -> int:
    return int(round(self.horizon / self.dt))

  @property
  def n_decisions(self) -> int:
    return int(round(self.horizon / self.decision_dt))

  @property
  def steps_per_decision(self) -> int:
    return max(1, int(round(self.decision_dt / self.dt)))


MPH_TO_MS = 0.44704
MS_TO_MPH = 2.23694


def _rollout_batch(p: PlantParams, sp_ms: np.ndarray, v0: float, a0: float,
                   pitch: float, dt: float, sp_hist_ms: np.ndarray) -> np.ndarray:
  """
  Roll out M candidate setpoint trajectories at once.
  sp_ms: (M, N) commanded setpoint in m/s. Returns (M, N) predicted speed.
  """
  m, n = sp_ms.shape
  d = int(round(p.deadtime / dt))
  if d > 0:
    pad = np.empty((m, d))
    if len(sp_hist_ms) >= d:
      pad[:] = sp_hist_ms[-d:]
    else:
      pad[:] = sp_hist_ms[0] if len(sp_hist_ms) else sp_ms[:, :1]
    seen = np.concatenate([pad, sp_ms], axis=1)[:, :n]
  else:
    seen = sp_ms

  v = np.full(m, v0)
  acc = np.full(m, a0)
  alpha = dt / max(p.tau, 1e-3)
  gterm = p.k_grade * np.sin(pitch)
  out = np.empty((m, n))
  for i in range(n):
    e = seen[:, i] - v - p.offset
    u = np.clip(p.gain * e, p.a_min_at(v), p.a_max_at(v)) + gterm
    acc += alpha * (u - acc)
    v = v + acc * dt
    out[:, i] = v
  return out


class ButtonMpc:
  def __init__(self, params: PlantParams, cfg: MpcConfig | None = None, seed: int = 0):
    self.p = params
    self.cfg = cfg or MpcConfig()
    # Seeded by default: identical inputs must produce an identical action. A
    # sampling planner that wanders between ticks is unreproducible on the road
    # and flaky in tests, and CEM refinement makes the draw itself unimportant.
    self.rng = np.random.default_rng(seed)
    self.prev_seq: np.ndarray | None = None

  def _expand(self, actions: np.ndarray, sp0_mph: float) -> np.ndarray:
    """
    (M, D) integer actions -> (M, n_steps) setpoint in mph.

    decision_dt need not be an integer multiple of dt, so repeat-then-pad rather
    than assuming the product lands exactly on n_steps.
    """
    cfg = self.cfg
    sp = sp0_mph + np.cumsum(actions, axis=1)
    np.clip(sp, cfg.sp_min_mph, cfg.sp_max_mph, out=sp)
    wide = np.repeat(sp, cfg.steps_per_decision, axis=1)
    if wide.shape[1] < cfg.n_steps:
      pad = np.repeat(wide[:, -1:], cfg.n_steps - wide.shape[1], axis=1)
      wide = np.concatenate([wide, pad], axis=1)
    return wide[:, :cfg.n_steps]

  def plan(self, v_desired: np.ndarray, v0: float, a0: float, sp0_mph: float,
           pitch: float = 0.0, sp_hist_mph: np.ndarray | None = None) -> tuple[int, dict]:
    """
    v_desired: (n_steps,) desired speed in m/s, UNROUNDED.
    Returns (action, debug) where action is -1, 0 or +1 for this tick.
    """
    cfg = self.cfg
    n, d = cfg.n_steps, cfg.n_decisions
    v_des = np.asarray(v_desired, dtype=np.float64)
    if len(v_des) < n:
      v_des = np.concatenate([v_des, np.full(n - len(v_des), v_des[-1])])
    v_des = v_des[:n]

    hist = (np.asarray(sp_hist_mph, dtype=np.float64) * MPH_TO_MS
            if sp_hist_mph is not None and len(sp_hist_mph)
            else np.array([sp0_mph * MPH_TO_MS]))

    # Seed the search with known-good candidates before sampling. Pure random
    # shooting at this sample count is under-converged: costs varied ~8% run to
    # run and the *first* action flipped with the draw, which is what actually
    # reaches the car. Warm-starting fixes both.
    seeds = []
    if self.prev_seq is not None and len(self.prev_seq) == d:
      shifted = np.concatenate([self.prev_seq[1:], [0]])  # last plan, advanced one tick
      seeds.append(shifted)
    feedforward = self._feedforward_seq(v_des, sp0_mph, d)
    seeds.append(feedforward)
    seeds.append(np.zeros(d, dtype=int))
    seeds = np.array(seeds, dtype=int)

    # action distribution, biased to "do nothing"; refined by CEM
    probs = np.tile(np.array([0.15, 0.70, 0.15]), (d, 1))
    # Start from the feedforward rather than None: if every rollout scored NaN or
    # the loop were configured away, falling through with no plan would crash the
    # control path. The search can then only improve on a sane default.
    best_seq, best_cost = feedforward.copy(), np.inf

    for _ in range(cfg.n_iters):
      u = self.rng.random((cfg.n_samples, d, 1))
      cum = np.cumsum(probs, axis=1)[None, :, :]
      actions = (u > cum).sum(axis=2) - 1  # -> {-1,0,1}
      actions[:len(seeds)] = seeds
      actions[len(seeds)] = best_seq  # keep incumbent

      sp_mph = self._expand(actions, sp0_mph)
      v_pred = _rollout_batch(self.p, sp_mph * MPH_TO_MS, v0, a0, pitch, cfg.dt, hist)

      err = v_pred - v_des
      cost = cfg.w_speed * np.sum(err ** 2, axis=1) * cfg.dt
      cost += cfg.w_terminal * err[:, -1] ** 2
      cost += cfg.w_press * np.sum(np.abs(actions), axis=1)

      order = np.argsort(cost)
      if cost[order[0]] < best_cost:
        best_cost = float(cost[order[0]])
        best_seq = actions[order[0]].copy()

      elite = actions[order[:cfg.n_elite]]
      counts = np.stack([(elite == k).sum(axis=0) for k in (-1, 0, 1)], axis=1).astype(float)
      probs = (counts + 1.0) / (counts.sum(axis=1, keepdims=True) + 3.0)

    self.prev_seq = best_seq
    action = int(best_seq[0])
    return action, {"cost": best_cost, "seq": best_seq}

  def _feedforward_seq(self, v_des: np.ndarray, sp0_mph: float, d: int) -> np.ndarray:
    """
    The obvious answer, always in the candidate set: at steady state the plant
    settles at (setpoint - offset), so drive the setpoint to the target as
    directly as the one-step-per-tick actuator allows.
    """
    target_mph = (float(np.mean(v_des[-max(1, len(v_des) // 4):])) + self.p.offset) * MS_TO_MPH
    need = int(round(np.clip(target_mph - sp0_mph, -d, d)))
    seq = np.zeros(d, dtype=int)
    step = 1 if need > 0 else -1
    seq[:abs(need)] = step
    return seq
