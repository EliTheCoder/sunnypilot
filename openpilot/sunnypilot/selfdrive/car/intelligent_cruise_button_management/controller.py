"""
Copyright (c) 2021-, Haibin Wen, sunnypilot, and a number of other contributors.

This file is part of sunnypilot and is licensed under the MIT License.
See the LICENSE.md file in the root directory for more details.
"""
import numpy as np

from openpilot.cereal import custom
from opendbc.car.structs import car
from opendbc.car import structs, apply_hysteresis
from openpilot.common.constants import CV
from openpilot.common.params import Params
from openpilot.common.realtime import DT_CTRL
from openpilot.selfdrive.modeld.constants import ModelConstants
from openpilot.sunnypilot.selfdrive.car.cruise_button_control.controller import CruiseButtonController
from openpilot.sunnypilot.selfdrive.car.cruise_button_control.mpc import MS_TO_MPH, MpcConfig
from openpilot.sunnypilot.selfdrive.car.cruise_button_control.plant import PlantParams
from openpilot.sunnypilot.selfdrive.car.intelligent_cruise_button_management.helpers import get_minimum_set_speed
from openpilot.sunnypilot.selfdrive.car.cruise_ext import CRUISE_BUTTON_TIMER, update_manual_button_timers

LongitudinalPlanSource = custom.LongitudinalPlanSP.LongitudinalPlanSource
State = custom.IntelligentCruiseButtonManagement.IntelligentCruiseButtonManagementState
SendButtonState = custom.IntelligentCruiseButtonManagement.SendButtonState

ALLOWED_SPEED_THRESHOLD = 1.8  # m/s, ~4 MPH
CONTROL_N_LON = 17  # longitudinalPlan.speeds length; ModelConstants.T_IDXS[:17] spans 2.5s
HYST_GAP = 0.0  # currently disabled; TODO-SP: might need to be brand-specific
INACTIVE_TIMER = 0.4


SEND_BUTTONS = {
  State.increasing: SendButtonState.increase,
  State.decreasing: SendButtonState.decrease,
}


class IntelligentCruiseButtonManagement:
  def __init__(self, CP: structs.CarParams, CP_SP: structs.CarParamsSP):
    self.CP = CP
    self.CP_SP = CP_SP

    self.v_target = 0
    self.v_cruise_cluster = 0
    self.v_cruise_min = 0
    self.cruise_button = SendButtonState.none
    self.state = State.inactive
    self.pre_active_timer = 0

    self.is_ready = False
    self.is_ready_prev = False
    self.v_target_ms_last = 0.0
    self.is_metric = False

    self.cruise_button_timers = CRUISE_BUTTON_TIMER

    # Optional MPC button planner. ICBM rounds the target to whole mph before
    # deciding anything, so a target between increments is unrepresentable; the
    # MPC plans on the unrounded trajectory and can dither the setpoint instead.
    self.mpc_enabled = Params().get_bool("CruiseButtonMpc")
    self.mpc = CruiseButtonController(PlantParams(), MpcConfig()) if self.mpc_enabled else None
    self._plan_t = np.array(ModelConstants.T_IDXS[:CONTROL_N_LON], dtype=np.float64)
    self._mpc_t = None
    self._frame = 0

  @property
  def v_cruise_equal(self) -> bool:
    return self.v_target == self.v_cruise_cluster

  def update_calculations(self, CS: car.CarState, LP_SP: custom.LongitudinalPlanSP) -> None:
    speed_conv = CV.MS_TO_KPH if self.is_metric else CV.MS_TO_MPH
    ms_conv = CV.KPH_TO_MS if self.is_metric else CV.MPH_TO_MS

    self.v_target_ms_last = apply_hysteresis(LP_SP.vTarget, self.v_target_ms_last, HYST_GAP * ms_conv)

    self.v_target = round(self.v_target_ms_last * speed_conv)
    self.v_cruise_min = get_minimum_set_speed(self.is_metric)
    self.v_cruise_cluster = round(CS.cruiseState.speedCluster * speed_conv)

  def update_state_machine(self) -> custom.IntelligentCruiseButtonManagement.SendButtonState:
    self.pre_active_timer = max(0, self.pre_active_timer - 1)

    # HOLDING, ACCELERATING, DECELERATING, PRE_ACTIVE
    if self.state != State.inactive:
      if not self.is_ready:
        self.state = State.inactive

      else:
        # PRE_ACTIVE
        if self.state == State.preActive:
          if self.pre_active_timer <= 0:
            if self.v_cruise_equal:
              self.state = State.holding

            elif self.v_target > self.v_cruise_cluster:
              self.state = State.increasing

            elif self.v_target < self.v_cruise_cluster and self.v_cruise_cluster > self.v_cruise_min:
              self.state = State.decreasing

        # HOLDING
        elif self.state == State.holding:
          if not self.v_cruise_equal:
            self.state = State.preActive

        # ACCELERATING
        elif self.state == State.increasing:
          if self.v_target <= self.v_cruise_cluster:
            self.state = State.holding

        # DECELERATING
        elif self.state == State.decreasing:
          if self.v_target >= self.v_cruise_cluster or self.v_cruise_cluster <= self.v_cruise_min:
            self.state = State.holding

    # INACTIVE
    elif self.state == State.inactive:
      if self.is_ready and not self.is_ready_prev:
        self.pre_active_timer = int(INACTIVE_TIMER / DT_CTRL)
        self.state = State.preActive

    send_button = SEND_BUTTONS.get(self.state, SendButtonState.none)

    return send_button

  def update_readiness(self, CS: car.CarState, CC: car.CarControl) -> None:
    update_manual_button_timers(CS, self.cruise_button_timers)

    ready = CC.enabled and not CC.cruiseControl.override and not CC.cruiseControl.cancel and not CC.cruiseControl.resume
    button_pressed = any(self.cruise_button_timers[k] > 0 for k in self.cruise_button_timers)

    self.is_ready = ready and not button_pressed

  def _desired_trajectory(self, LP) -> np.ndarray | None:
    """
    Resample the planner's speed trajectory onto the MPC's uniform grid.

    longitudinalPlan.speeds spans only 2.5s on a non-uniform time base while the
    planner needs 4s -- below ~3s the deadtime leaves too little window for a
    dither to pay off, so the horizon cannot simply be shortened to match. Beyond
    2.5s the final planned speed is held: it is the plan's own steady-state
    estimate, and the far horizon barely influences the press chosen now.
    """
    speeds = getattr(LP, "speeds", None)
    if speeds is None or len(speeds) < 2:
      return None
    if self._mpc_t is None:
      self._mpc_t = np.arange(self.mpc.cfg.n_steps) * self.mpc.cfg.dt
    v = np.asarray(speeds, dtype=np.float64)
    t = self._plan_t[:len(v)]
    return np.interp(self._mpc_t, t, v)  # np.interp holds the edge value past t[-1]

  def _run_mpc(self, CS: car.CarState, LP) -> custom.IntelligentCruiseButtonManagement.SendButtonState:
    v_des = self._desired_trajectory(LP)
    if v_des is None:
      self.state = State.inactive
      return SendButtonState.none

    self._frame += 1
    t = self._frame * DT_CTRL
    setpoint_mph = float(CS.cruiseState.speedCluster * MS_TO_MPH)
    driver_pressing = any(self.cruise_button_timers[k] > 0 for k in self.cruise_button_timers)

    st = self.mpc.update(t, float(CS.vEgo), float(CS.aEgo), setpoint_mph, v_des,
                         ready=self.is_ready, driver_pressing=driver_pressing)

    if st.action > 0:
      self.state = State.increasing
      return SendButtonState.increase
    if st.action < 0:
      self.state = State.decreasing
      return SendButtonState.decrease
    self.state = State.holding if st.active else State.inactive
    return SendButtonState.none

  def run(self, CS: car.CarState, CC: car.CarControl, LP_SP: custom.LongitudinalPlanSP,
          is_metric: bool, LP=None) -> None:
    if self.CP_SP.pcmCruiseSpeed:
      return

    self.is_metric = is_metric

    self.update_calculations(CS, LP_SP)
    self.update_readiness(CS, CC)

    if self.mpc_enabled and LP is not None:
      self.cruise_button = self._run_mpc(CS, LP)
    else:
      self.cruise_button = self.update_state_machine()

    self.is_ready_prev = self.is_ready
