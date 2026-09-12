"""
Online cruise-button controller.

Replaces ICBM's round-then-relay scheme. ICBM quantizes the desired speed to whole
mph before deciding anything (`round(v_target * speed_conv)`) and then runs a bang-bang
state machine on the integer error, so a target of 74.4 mph is indistinguishable from
74.0 and sub-mph tracking is impossible by construction. Here the unrounded target
goes straight to the planner and the quantization lives only in the actuator, where
it belongs.

Kept deliberately from ICBM, because they encode real vehicle behavior:
  - press blocking while the driver is on the stalk (fighting the driver is bad)
  - a minimum interval between presses
  - a floor on the commanded setpoint
"""
from dataclasses import dataclass

import numpy as np

from openpilot.sunnypilot.selfdrive.car.cruise_button_control.mpc import (
  ButtonMpc, MpcConfig)
from openpilot.sunnypilot.selfdrive.car.cruise_button_control.plant import PlantParams


@dataclass
class ControllerState:
  action: int = 0          # -1, 0, +1 for this tick
  planned_setpoint: float = 0.0
  solve_ms: float = 0.0
  active: bool = False
  reason: str = "inactive"


class CruiseButtonController:
  def __init__(self, params: PlantParams, cfg: MpcConfig | None = None):
    self.p = params
    self.cfg = cfg or MpcConfig()
    self.mpc = ButtonMpc(params, self.cfg)
    self.sp_hist: list[float] = []
    self.t_last_press = -1e9
    self.state = ControllerState()

  def reset(self):
    self.sp_hist.clear()
    self.t_last_press = -1e9
    self.state = ControllerState()

  def update(self, t: float, v_ego: float, a_ego: float, setpoint_mph: float,
             v_desired: np.ndarray, ready: bool, driver_pressing: bool,
             pitch: float = 0.0) -> ControllerState:
    """
    v_desired: desired speed trajectory in m/s at cfg.dt spacing, UNROUNDED.
               Pass the planner's speed trajectory directly; do not pre-round it.
    ready:     longitudinal control engaged and not overridden.
    """
    st = ControllerState(planned_setpoint=setpoint_mph)

    # One sample per tick, unconditionally. This feeds the plant's deadtime
    # buffer, which indexes by *time* -- appending only on change would make
    # sp_hist[-d:] span far more than `deadtime` seconds and mispredict what the
    # car has already acted on.
    self.sp_hist.append(setpoint_mph)
    self.sp_hist = self.sp_hist[-256:]

    if not ready:
      self.reset()
      st.reason = "not ready"
      self.state = st
      return st

    if driver_pressing:
      # the driver owns the stalk; stay out of the way and resync next tick
      self.t_last_press = t
      st.reason = "driver pressing"
      self.state = st
      return st

    if t - self.t_last_press < self.cfg.min_press_interval:
      st.active = True
      st.reason = "press cooldown"
      self.state = st
      return st

    import time as _time
    t0 = _time.perf_counter()
    action, _dbg = self.mpc.plan(v_desired, v_ego, a_ego, setpoint_mph, pitch,
                                 np.array(self.sp_hist, dtype=np.float64))
    st.solve_ms = (_time.perf_counter() - t0) * 1e3
    st.active = True

    if action != 0:
      lo, hi = self.cfg.sp_min_mph, self.cfg.sp_max_mph
      new_sp = float(np.clip(setpoint_mph + action, lo, hi))
      if new_sp == setpoint_mph:
        action = 0  # clamped; do not burn a press that cannot move the setpoint
      else:
        st.planned_setpoint = new_sp
        self.t_last_press = t

    st.action = action
    st.reason = "planning"
    self.state = st
    return st

  @staticmethod
  def desired_from_target(v_target: float, n: int) -> np.ndarray:
    """Constant-speed trajectory helper for callers that only have a scalar target."""
    return np.full(n, v_target, dtype=np.float64)
