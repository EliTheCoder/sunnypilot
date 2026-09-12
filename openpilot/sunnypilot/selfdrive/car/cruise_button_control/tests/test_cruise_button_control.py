import numpy as np
import pytest

from openpilot.sunnypilot.selfdrive.car.cruise_button_control.controller import CruiseButtonController
from openpilot.sunnypilot.selfdrive.car.cruise_button_control.mpc import (
  MPH_TO_MS, MS_TO_MPH, ButtonMpc, MpcConfig)
from openpilot.sunnypilot.selfdrive.car.cruise_button_control.plant import PlantParams, rollout


@pytest.fixture
def p():
  return PlantParams()


@pytest.fixture
def mpc(p):
  return ButtonMpc(p, MpcConfig())


class TestPlant:
  def test_rollout_shapes(self, p):
    v, a = rollout(p, np.full(20, 33.0), 32.0, 0.0, 0.0, 0.1)
    assert v.shape == a.shape == (20,)
    assert np.all(np.isfinite(v)) and np.all(np.isfinite(a))

  def test_converges_to_setpoint_minus_offset(self, p):
    """Steady state speed must land at setpoint - offset; the offset is the ~2mph
    speedo calibration bias measured on two separate drives."""
    sp = 33.0
    v, _ = rollout(p, np.full(4000, sp), 25.0, 0.0, 0.0, 0.05)
    assert v[-1] == pytest.approx(sp - p.offset, abs=0.05)

  def test_authority_is_speed_dependent(self, p):
    """Throttle authority falls with speed, coast authority grows with it."""
    assert p.a_max_at(13.0) > p.a_max_at(35.0)
    assert p.a_min_at(13.0) > p.a_min_at(35.0)

  def test_authority_never_degenerate(self, p):
    for v in (0.0, 10.0, 50.0, 200.0):
      assert p.a_min_at(v) < 0.0 < p.a_max_at(v)

  def test_deadtime_delays_response(self, p):
    """A *new* setpoint must not move the car until the deadtime has elapsed.
    This needs explicit history at the old setpoint: with sp_hist omitted the
    rollout assumes sp_seq[0] was already in effect, so there is nothing to delay.
    """
    dt = 0.05
    n = int(p.deadtime / dt)
    old = 30.0 + p.offset  # setpoint that holds 30 m/s at steady state
    hist = np.full(n + 5, old)
    v, _ = rollout(p, np.full(300, 40.0), 30.0, 0.0, 0.0, dt, sp_hist=hist)
    assert abs(v[max(n - 2, 0)] - 30.0) < 0.05, "moved before deadtime elapsed"
    assert v[-1] > 30.5, "never responded after deadtime"

  def test_grade_pushes_speed_down_uphill(self, p):
    flat, _ = rollout(p, np.full(100, 33.0), 33.0, 0.0, 0.0, 0.1)
    up, _ = rollout(p, np.full(100, 33.0), 33.0, 0.0, 0.05, 0.1)
    assert up[-1] < flat[-1]


class TestMpc:
  def test_expand_length_when_decision_dt_not_multiple_of_dt(self, p):
    """decision_dt/dt need not be integral; regression for a shape bug that only
    appeared once dt moved off 0.05."""
    cfg = MpcConfig(dt=0.1, decision_dt=0.25, horizon=8.0)
    m = ButtonMpc(p, cfg)
    out = m._expand(np.zeros((4, cfg.n_decisions), dtype=int), 70.0)
    assert out.shape == (4, cfg.n_steps)

  def test_action_is_valid(self, mpc):
    a, _ = mpc.plan(np.full(mpc.cfg.n_steps, 33.0), 33.0, 0.0, 75.0)
    assert a in (-1, 0, 1)

  def test_commands_up_when_target_above(self, mpc, p):
    sp = 70.0
    v = sp * MPH_TO_MS - p.offset
    a, _ = mpc.plan(np.full(mpc.cfg.n_steps, v + 2.0), v, 0.0, sp)
    assert a == 1

  def test_commands_down_when_target_below(self, mpc, p):
    sp = 70.0
    v = sp * MPH_TO_MS - p.offset
    a, _ = mpc.plan(np.full(mpc.cfg.n_steps, v - 2.0), v, 0.0, sp)
    assert a == -1

  def test_holds_when_on_target(self, mpc, p):
    """At steady state on an exact increment there is nothing to gain, and the
    press cost should stop it fidgeting."""
    sp = 70.0
    v = sp * MPH_TO_MS - p.offset
    acts = [mpc.plan(np.full(mpc.cfg.n_steps, v), v, 0.0, sp)[0] for _ in range(8)]
    assert sum(a != 0 for a in acts) <= 2

  def test_respects_setpoint_limits(self, p):
    cfg = MpcConfig(sp_min_mph=60.0, sp_max_mph=62.0)
    m = ButtonMpc(p, cfg)
    sp_mph = m._expand(np.full((1, cfg.n_decisions), 1, dtype=int), 61.0)
    assert sp_mph.max() <= 62.0
    sp_mph = m._expand(np.full((1, cfg.n_decisions), -1, dtype=int), 61.0)
    assert sp_mph.min() >= 60.0


class TestController:
  def test_inactive_when_not_ready(self, p):
    c = CruiseButtonController(p)
    st = c.update(0.0, 30.0, 0.0, 70.0, np.full(c.cfg.n_steps, 30.0),
                  ready=False, driver_pressing=False)
    assert st.action == 0 and not st.active

  def test_yields_to_driver(self, p):
    c = CruiseButtonController(p)
    st = c.update(0.0, 30.0, 0.0, 70.0, np.full(c.cfg.n_steps, 35.0),
                  ready=True, driver_pressing=True)
    assert st.action == 0
    assert st.reason == "driver pressing"

  def test_press_rate_limited(self, p):
    c = CruiseButtonController(p)
    vd = np.full(c.cfg.n_steps, 40.0)
    first = c.update(0.0, 30.0, 0.0, 70.0, vd, True, False)
    assert first.action != 0
    nxt = c.update(0.01, 30.0, 0.0, 71.0, vd, True, False)
    assert nxt.action == 0 and nxt.reason == "press cooldown"

  def test_no_press_when_clamped(self, p):
    """Do not spend a press that cannot move the setpoint."""
    cfg = MpcConfig(sp_max_mph=70.0)
    c = CruiseButtonController(p, cfg)
    st = c.update(0.0, 40.0, 0.0, 70.0, np.full(cfg.n_steps, 60.0), True, False)
    assert st.action == 0

  def test_tracks_between_increments_closed_loop(self, p):
    """The whole point: hold a speed that falls between two mph increments."""
    cfg = MpcConfig()
    c = CruiseButtonController(p, cfg)
    dt = cfg.dt
    target = 74.5 * MPH_TO_MS - p.offset
    v, a, sp = target, 0.0, 75.0
    hist = [sp]
    errs = []
    for i in range(int(70 / dt)):
      t = i * dt
      st = c.update(t, v, a, sp, np.full(cfg.n_steps, target), True, False)
      if st.action:
        sp = st.planned_setpoint
      hist.append(sp)
      vv, aa = rollout(p, np.array([sp * MPH_TO_MS]), v, a, 0.0, dt,
                       sp_hist=np.array(hist[:-1]) * MPH_TO_MS)
      v, a = float(vv[0]), float(aa[0])
      if t > 20:
        errs.append((v - target) * MS_TO_MPH)
    rms = float(np.sqrt(np.mean(np.square(errs))))
    # a non-dithering controller is stuck at 0.5mph here; require clearly better
    assert rms < 0.35, f"rms={rms:.3f} mph"
