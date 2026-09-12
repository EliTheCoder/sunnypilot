"""
Copyright (c) 2021-, Haibin Wen, sunnypilot, and a number of other contributors.

This file is part of sunnypilot and is licensed under the MIT License.
See the LICENSE.md file in the root directory for more details.
"""
import numpy as np
import cereal.messaging as messaging
from openpilot.common.params import Params
from openpilot.common.realtime import DT_MDL
from openpilot.selfdrive.car.cruise import V_CRUISE_UNSET
from openpilot.sunnypilot import PARAMS_UPDATE_PERIOD


class ModelLongitudinalController:
  """ICBM source: drives cruise set-speed using the MPC's planned trajectory.

  When enabled (ICBMModelLong param), replaces the lead-follow P-controller.

  Uses v_desired_trajectory[-1] (the previous cycle's MPC planned speed at the
  planning horizon) to read the MPC's intent. The MPC already incorporates radar
  lead data, safety constraints, and speed limits.

  Key design: MLC output is used ONLY for the ICBM target (LP_SP.vTarget).
  It is NOT fed back into the MPC as v_cruise — that would create a downward
  spiral where every value below v_cruise pushes the MPC lower each cycle.
  The caller (update_targets) must return the MPC target separately.

  Deceleration gate: only decrease cruise when v_trajectory < v_ego. This means
  the MPC is actively planning to slow (lead, curve, etc.), not just that the car
  hasn't finished accelerating to the set speed yet. When no deceleration is
  needed, falls back to cruiseState.speed so ICBM holds steady and lets native
  cruise handle acceleration to target.
  """

  FALLBACK_HORIZON = 2.0  # seconds, used only when no MPC trajectory is available

  def __init__(self):
    self.params = Params()
    self.frame = -1
    self.enabled = self.params.get_bool("ICBMModelLong")
    self.is_enabled = False
    self.is_active = False
    self.output_v_target: float = V_CRUISE_UNSET
    self.output_a_target: float = 0.0

  def _update_params(self) -> None:
    if self.frame % int(PARAMS_UPDATE_PERIOD / DT_MDL) == 0:
      self.enabled = self.params.get_bool("ICBMModelLong")

  def update(self, sm: messaging.SubMaster, long_enabled: bool, long_override: bool, v_ego: float,
             prev_speeds: np.ndarray | None = None) -> None:
    self.frame += 1
    self._update_params()

    self.is_enabled = self.enabled and long_enabled
    self.is_active = self.is_enabled and not long_override

    if self.is_active:
      cruise_speed = sm['carState'].cruiseState.speed  # m/s, from car's CAN (e.g. LVR12 for non-SCC)

      if prev_speeds is not None and len(prev_speeds) > 0 and prev_speeds[-1] > 0:
        v_trajectory = float(prev_speeds[-1])

        if v_trajectory < v_ego:
          # MPC is planning to decelerate (lead car, curve, etc.): tell ICBM
          # to lower the cruise set-speed to match the MPC's planned trajectory.
          self.output_v_target = max(0.0, v_trajectory)
        elif cruise_speed > 0:
          # MPC is accelerating toward cruise or car is already there: hold at
          # the car's actual reported cruise speed so ICBM doesn't interfere
          # while native cruise handles acceleration.
          self.output_v_target = cruise_speed
        else:
          self.output_v_target = v_trajectory
      else:
        # Fallback: no valid trajectory yet (first frame). Use desiredAcceleration.
        a_model = sm['modelV2'].action.desiredAcceleration
        if cruise_speed > 0 and a_model >= 0:
          self.output_v_target = cruise_speed
        else:
          self.output_v_target = max(0.0, v_ego + a_model * self.FALLBACK_HORIZON)

      self.output_a_target = 0.0
    else:
      self.output_v_target = V_CRUISE_UNSET
      self.output_a_target = 0.0
