import numpy as np

from openpilot.cereal import custom
from opendbc.car import structs
from openpilot.common.test import OpenpilotTestCase
from openpilot.sunnypilot.selfdrive.car.cruise_button_control.mpc import MPH_TO_MS
from openpilot.sunnypilot.selfdrive.car.intelligent_cruise_button_management.controller import (
  IntelligentCruiseButtonManagement)

SendButtonState = custom.IntelligentCruiseButtonManagement.SendButtonState
State = custom.IntelligentCruiseButtonManagement.IntelligentCruiseButtonManagementState


class FakeLPSP:
  """Minimal stand-in for longitudinalPlanSP carrying only what the relay reads."""
  def __init__(self, button):
    self.cruiseButton = button


def make_cs(v_ego, setpoint_mph):
  cs = structs.CarState()
  cs.vEgo = v_ego
  cs.aEgo = 0.0
  cs.cruiseState.speed = setpoint_mph * MPH_TO_MS
  cs.cruiseState.speedCluster = setpoint_mph * MPH_TO_MS
  return cs


class TestCruiseButtonRelay(OpenpilotTestCase):
  """
  The button is planned in plannerd and only relayed here. selfdrived runs at
  100Hz and the MPC solve costs ~32ms on device, so this path must stay free of
  any planning work.
  """

  def make_icbm(self, mocker, mpc_enabled):
    mod = "openpilot.sunnypilot.selfdrive.car.intelligent_cruise_button_management.controller"
    mocker.patch(mod + ".Params.get_bool", return_value=mpc_enabled)
    CP = structs.CarParams()
    CP_SP = structs.CarParamsSP()
    CP_SP.pcmCruiseSpeed = False
    icbm = IntelligentCruiseButtonManagement(CP, CP_SP)
    icbm.is_ready = True
    return icbm

  def test_off_by_default(self, mocker):
    assert self.make_icbm(mocker, False).mpc_enabled is False

  def test_relays_increase(self, mocker):
    icbm = self.make_icbm(mocker, True)
    out = icbm._relay_planned_button(FakeLPSP(SendButtonState.increase))
    assert out == SendButtonState.increase
    assert icbm.state == State.increasing

  def test_relays_decrease(self, mocker):
    icbm = self.make_icbm(mocker, True)
    out = icbm._relay_planned_button(FakeLPSP(SendButtonState.decrease))
    assert out == SendButtonState.decrease
    assert icbm.state == State.decreasing

  def test_relays_none(self, mocker):
    icbm = self.make_icbm(mocker, True)
    assert icbm._relay_planned_button(FakeLPSP(SendButtonState.none)) == SendButtonState.none

  def test_not_ready_blocks_relay(self, mocker):
    """Readiness is still enforced downstream of the planner."""
    icbm = self.make_icbm(mocker, True)
    icbm.is_ready = False
    assert icbm._relay_planned_button(FakeLPSP(SendButtonState.increase)) == SendButtonState.none
    assert icbm.state == State.inactive

  def test_missing_field_is_safe(self, mocker):
    """An older longitudinalPlanSP without the field must not raise."""
    icbm = self.make_icbm(mocker, True)

    class Empty:
      pass

    assert icbm._relay_planned_button(Empty()) == SendButtonState.none

  def test_pcm_cruise_speed_gate_still_returns_early(self, mocker):
    icbm = self.make_icbm(mocker, True)
    icbm.CP_SP.pcmCruiseSpeed = True
    icbm.cruise_button = SendButtonState.none
    icbm.run(make_cs(31.0, 70.0), structs.CarControl(),
             custom.LongitudinalPlanSP.new_message(), False)
    assert icbm.cruise_button == SendButtonState.none


class TestDecelAuthorityWarning(OpenpilotTestCase):
  """
  The cruise buttons cannot brake. When the longitudinal plan asks for more
  deceleration than coasting provides, the driver has to be told -- otherwise the
  car simply closes on the lead, which is what happened on the road.
  """

  def make_planner(self):
    from openpilot.sunnypilot.selfdrive.controls.lib.longitudinal_planner import LongitudinalPlannerSP
    from openpilot.sunnypilot.selfdrive.car.cruise_button_control.controller import CruiseButtonController
    from openpilot.sunnypilot.selfdrive.car.cruise_button_control.mpc import MpcConfig
    from openpilot.sunnypilot.selfdrive.car.cruise_button_control.plant import PlantParams
    from openpilot.sunnypilot.selfdrive.selfdrived.events import EventsSP

    p = object.__new__(LongitudinalPlannerSP)
    p.events_sp = EventsSP()
    p.cruise_button_mpc = CruiseButtonController(PlantParams(), MpcConfig())
    p._decel_short_frames = 0
    return p

  @staticmethod
  def _raised(p):
    from openpilot.cereal import custom
    name = custom.OnroadEventSP.EventName.insufficientDecelAuthority
    return any(e.name == name for e in p.events_sp.to_msg())

  def test_quiet_when_plan_is_achievable(self):
    p = self.make_planner()
    v = 30.0
    # ask for gentle decel well inside coast authority
    p.a_desired_trajectory = np.full(17, -0.1)
    for _ in range(40):
      p.events_sp.clear()
      p.update_decel_authority(True, v)
    assert not self._raised(p)

  def test_warns_when_plan_exceeds_coast_authority(self):
    p = self.make_planner()
    v = 30.0
    p.a_desired_trajectory = np.full(17, -3.0)  # hard braking, impossible by coasting
    for _ in range(40):
      p.events_sp.clear()
      p.update_decel_authority(True, v)
    assert self._raised(p)

  def test_debounced_not_instant(self):
    """A momentary dip in the trajectory must not chime."""
    p = self.make_planner()
    p.a_desired_trajectory = np.full(17, -3.0)
    p.events_sp.clear()
    p.update_decel_authority(True, 30.0)
    assert not self._raised(p), "warned on the very first frame"

  def test_clears_when_plan_becomes_achievable(self):
    p = self.make_planner()
    p.a_desired_trajectory = np.full(17, -3.0)
    for _ in range(40):
      p.events_sp.clear()
      p.update_decel_authority(True, 30.0)
    assert self._raised(p)
    p.a_desired_trajectory = np.full(17, -0.1)
    p.events_sp.clear()
    p.update_decel_authority(True, 30.0)
    assert not self._raised(p)

  def test_silent_when_not_engaged(self):
    p = self.make_planner()
    p.a_desired_trajectory = np.full(17, -3.0)
    for _ in range(40):
      p.events_sp.clear()
      p.update_decel_authority(False, 30.0)
    assert not self._raised(p)

  def test_silent_at_crawl(self):
    p = self.make_planner()
    p.a_desired_trajectory = np.full(17, -3.0)
    for _ in range(40):
      p.events_sp.clear()
      p.update_decel_authority(True, 1.0)
    assert not self._raised(p)

  def test_uses_speed_dependent_authority(self):
    """Coast authority grows with speed, so a decel impossible at low speed may be
    achievable at highway speed; the threshold must follow it."""
    p = self.make_planner()
    a_min_slow = p.cruise_button_mpc.p.a_min_at(13.0)
    a_min_fast = p.cruise_button_mpc.p.a_min_at(35.0)
    assert a_min_fast < a_min_slow
