import numpy as np
import pytest

from openpilot.cereal import custom
from opendbc.car import structs
from openpilot.sunnypilot.selfdrive.car.cruise_button_control.mpc import MPH_TO_MS
from openpilot.sunnypilot.selfdrive.car.intelligent_cruise_button_management.controller import (
  CONTROL_N_LON, IntelligentCruiseButtonManagement)

SendButtonState = custom.IntelligentCruiseButtonManagement.SendButtonState


class FakeLP:
  def __init__(self, speeds):
    self.speeds = list(speeds)


def make_icbm(mocker, mpc_enabled):
  mocker.patch("openpilot.sunnypilot.selfdrive.car.intelligent_cruise_button_management."
               "controller.Params.get_bool", return_value=mpc_enabled)
  CP = structs.CarParams()
  CP_SP = structs.CarParamsSP()
  CP_SP.pcmCruiseSpeed = False
  icbm = IntelligentCruiseButtonManagement(CP, CP_SP)
  icbm.is_ready = True
  return icbm


def make_cs(v_ego, setpoint_mph):
  cs = structs.CarState()
  cs.vEgo = v_ego
  cs.aEgo = 0.0
  cs.cruiseState.speed = setpoint_mph * MPH_TO_MS
  cs.cruiseState.speedCluster = setpoint_mph * MPH_TO_MS
  return cs


class TestCruiseButtonMpcIntegration:
  def test_off_by_default(self, mocker):
    assert make_icbm(mocker, False).mpc is None

  def test_enabled_constructs_planner(self, mocker):
    assert make_icbm(mocker, True).mpc is not None

  def test_trajectory_resampled_to_planner_grid(self, mocker):
    icbm = make_icbm(mocker, True)
    out = icbm._desired_trajectory(FakeLP(np.linspace(30.0, 32.0, CONTROL_N_LON)))
    assert out.shape == (icbm.mpc.cfg.n_steps,)
    assert out[0] == pytest.approx(30.0, abs=1e-6)

  def test_trajectory_holds_past_plan_horizon(self, mocker):
    """longitudinalPlan.speeds spans 2.5s but the planner horizon is longer; the
    tail must hold the last planned speed, never extrapolate past it."""
    icbm = make_icbm(mocker, True)
    out = icbm._desired_trajectory(FakeLP(np.linspace(30.0, 32.0, CONTROL_N_LON)))
    assert out[-1] == pytest.approx(32.0, abs=1e-6)
    assert out.max() <= 32.0 + 1e-6

  def test_missing_plan_is_safe(self, mocker):
    icbm = make_icbm(mocker, True)
    assert icbm._desired_trajectory(FakeLP([])) is None
    assert icbm._run_mpc(make_cs(31.0, 70.0), FakeLP([])) == SendButtonState.none

  def test_increase_when_plan_faster(self, mocker):
    icbm = make_icbm(mocker, True)
    out = icbm._run_mpc(make_cs(31.0, 70.0), FakeLP(np.full(CONTROL_N_LON, 34.0)))
    assert out == SendButtonState.increase

  def test_decrease_when_plan_slower(self, mocker):
    icbm = make_icbm(mocker, True)
    out = icbm._run_mpc(make_cs(31.0, 70.0), FakeLP(np.full(CONTROL_N_LON, 27.0)))
    assert out == SendButtonState.decrease

  def test_not_ready_emits_nothing(self, mocker):
    icbm = make_icbm(mocker, True)
    icbm.is_ready = False
    assert icbm._run_mpc(make_cs(31.0, 70.0), FakeLP(np.full(CONTROL_N_LON, 34.0))) == SendButtonState.none

  def test_pcm_cruise_speed_gate_still_returns_early(self, mocker):
    """The existing gate must keep ICBM (and therefore the MPC) inert."""
    icbm = make_icbm(mocker, True)
    icbm.CP_SP.pcmCruiseSpeed = True
    icbm.cruise_button = SendButtonState.none
    icbm.run(make_cs(31.0, 70.0), structs.CarControl(),
             custom.LongitudinalPlanSP.new_message(), False,
             LP=FakeLP(np.full(CONTROL_N_LON, 34.0)))
    assert icbm.cruise_button == SendButtonState.none
