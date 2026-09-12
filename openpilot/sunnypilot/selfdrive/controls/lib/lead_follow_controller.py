"""
Copyright (c) 2021-, Haibin Wen, sunnypilot, and a number of other contributors.

This file is part of sunnypilot and is licensed under the MIT License.
See the LICENSE.md file in the root directory for more details.
"""
import cereal.messaging as messaging
from openpilot.common.params import Params
from openpilot.common.realtime import DT_MDL
from openpilot.selfdrive.car.cruise import V_CRUISE_UNSET
from openpilot.sunnypilot import PARAMS_UPDATE_PERIOD

# Goal zone expressed as time headways behind the lead.
T_GAP_MIN = 2.0          # seconds — lower zone boundary (too close if below)
T_GAP_MAX = 3.0          # seconds — upper zone boundary (too far if above)
MIN_FOLLOW_DIST = 8.0    # meters — floor for lower boundary at very low speeds

# P controller gain: speed adjustment per meter of distance error outside the zone.
KP = 0.15  # (m/s) / m

# Speed adjustment limits outside the zone.
# Snaps to at least 1 mph on first exit from zone (cruise control resolution).
MIN_SPEED_ADJ = 0.447    # m/s ≈ 1 mph
MAX_SPEED_ADJ = 2.0      # m/s

# EMA smoothing for dRel and vLead. Lower alpha = smoother, more lag.
EMA_ALPHA = 0.15

# Minimum modelProb to treat a radar lead as valid.
MIN_LEAD_PROB = 0.5


class LeadFollowController:
  """ICBM source: goal-zone P controller that adjusts vTarget around the lead car's speed.

  Maintains a following distance in the zone [T_GAP_MIN, T_GAP_MAX] seconds behind the lead
  (floored at MIN_FOLLOW_DIST). Inside the zone, matches lead speed exactly. Outside, a P
  controller produces a speed adjustment of at least MIN_SPEED_ADJ (1 mph) to ensure the
  cruise control's 1 mph resolution is always crossed:

    Inside zone:  output_v_target = v_lead_ema
    Too close:    output_v_target = v_lead_ema - clamp(KP * |error|, MIN_SPEED_ADJ, MAX_SPEED_ADJ)
    Too far:      output_v_target = v_lead_ema + clamp(KP * |error|, MIN_SPEED_ADJ, MAX_SPEED_ADJ)

  dRel and vLead are smoothed with an EMA that resets on new lead acquisition.

  No I term: ICBM acts as the integrator (pressing buttons until set speed reaches v_target).

  When no lead is present, output_v_target is V_CRUISE_UNSET (no constraint).
  """

  def __init__(self):
    self.params = Params()
    self.frame = -1
    self.enabled = self.params.get_bool("ICBMLeadFollow")
    self.is_enabled = False
    self.is_active = False
    self.output_v_target: float = V_CRUISE_UNSET
    self.output_a_target: float = 0.0

    self._d_rel: float = 0.0
    self._v_lead: float = 0.0
    self._prev_lead_status: bool = False
    self.d_rel_ema: float = 0.0

  def _update_params(self) -> None:
    if self.frame % int(PARAMS_UPDATE_PERIOD / DT_MDL) == 0:
      self.enabled = self.params.get_bool("ICBMLeadFollow")

  def _update_ema(self, lead) -> None:
    if lead.status and not self._prev_lead_status:
      self._d_rel = lead.dRel
      self._v_lead = lead.vLead
    elif lead.status:
      self._d_rel += EMA_ALPHA * (lead.dRel - self._d_rel)
      self._v_lead += EMA_ALPHA * (lead.vLead - self._v_lead)
    self.d_rel_ema = self._d_rel

  def update(self, sm: messaging.SubMaster, long_enabled: bool, long_override: bool, v_ego: float) -> None:
    self._update_params()

    self.is_enabled = self.enabled and long_enabled

    lead = sm['radarState'].leadOne
    self._update_ema(lead)
    self._prev_lead_status = lead.status

    self.is_active = self.is_enabled and lead.status and lead.modelProb >= MIN_LEAD_PROB and not long_override

    if self.is_active:
      zone_min = max(v_ego * T_GAP_MIN, MIN_FOLLOW_DIST)
      zone_max = v_ego * T_GAP_MAX

      if self._d_rel < zone_min:
        adj = -min(max(KP * (zone_min - self._d_rel), MIN_SPEED_ADJ), MAX_SPEED_ADJ)
        self.output_v_target = max(self._v_lead + adj, 0.0)
      elif self._d_rel > zone_max:
        adj = min(max(KP * (self._d_rel - zone_max), MIN_SPEED_ADJ), MAX_SPEED_ADJ)
        self.output_v_target = max(self._v_lead + adj, 0.0)
      else:
        self.output_v_target = max(self._v_lead, 0.0)
    else:
      self.output_v_target = V_CRUISE_UNSET

    self.frame += 1
