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
