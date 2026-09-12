import numpy as np

from openpilot.common.test import OpenpilotTestCase
from openpilot.sunnypilot.selfdrive.car.cruise_button_control.controller import CruiseButtonController
from openpilot.sunnypilot.selfdrive.car.cruise_button_control.mpc import (
  MPH_TO_MS, MS_TO_MPH, ButtonMpc, MpcConfig)
from openpilot.sunnypilot.selfdrive.car.cruise_button_control.plant import PlantParams, rollout


class TestPlant(OpenpilotTestCase):
  def setUp(self):
    self.p = PlantParams()

  def test_rollout_shapes(self):
    v, a = rollout(self.p, np.full(20, 33.0), 32.0, 0.0, 0.0, 0.1)
    assert v.shape == a.shape == (20,)
    assert np.all(np.isfinite(v)) and np.all(np.isfinite(a))

  def test_converges_to_setpoint_minus_offset(self):
    """Steady state speed must land at setpoint - offset; the offset is the ~2mph
    speedo calibration bias measured on two separate drives."""
    sp = 33.0
    v, _ = rollout(self.p, np.full(4000, sp), 25.0, 0.0, 0.0, 0.05)
    self.assertAlmostEqual(float(v[-1]), sp - self.p.offset, delta=0.05)

  def test_authority_is_speed_dependent(self):
    """Throttle authority falls with speed, coast authority grows with it."""
    assert self.p.a_max_at(13.0) > self.p.a_max_at(35.0)
    assert self.p.a_min_at(13.0) > self.p.a_min_at(35.0)

  def test_authority_never_degenerate(self):
    for v in (0.0, 10.0, 50.0, 200.0):
      assert self.p.a_min_at(v) < 0.0 < self.p.a_max_at(v)

  def test_deadtime_delays_response(self):
    """A *new* setpoint must not move the car until the deadtime has elapsed.
    This needs explicit history at the old setpoint: with sp_hist omitted the
    rollout assumes sp_seq[0] was already in effect, so there is nothing to delay.
    """
    dt = 0.05
    n = int(self.p.deadtime / dt)
    hist = np.full(n + 5, 30.0 + self.p.offset)
    v, _ = rollout(self.p, np.full(300, 40.0), 30.0, 0.0, 0.0, dt, sp_hist=hist)
    assert abs(v[max(n - 2, 0)] - 30.0) < 0.05, "moved before deadtime elapsed"
    assert v[-1] > 30.5, "never responded after deadtime"

  def test_grade_pushes_speed_down_uphill(self):
    flat, _ = rollout(self.p, np.full(100, 33.0), 33.0, 0.0, 0.0, 0.1)
    up, _ = rollout(self.p, np.full(100, 33.0), 33.0, 0.0, 0.05, 0.1)
    assert up[-1] < flat[-1]


class TestMpc(OpenpilotTestCase):
  def setUp(self):
    self.p = PlantParams()
    self.mpc = ButtonMpc(self.p, MpcConfig())

  def test_expand_length_when_decision_dt_not_multiple_of_dt(self):
    """decision_dt/dt need not be integral; regression for a shape bug that only
    appeared once dt moved off 0.05."""
    cfg = MpcConfig(dt=0.1, decision_dt=0.25, horizon=8.0)
    m = ButtonMpc(self.p, cfg)
    out = m._expand(np.zeros((4, cfg.n_decisions), dtype=int), 70.0)
    assert out.shape == (4, cfg.n_steps)

  def test_action_is_valid(self):
    a, _ = self.mpc.plan(np.full(self.mpc.cfg.n_steps, 33.0), 33.0, 0.0, 75.0)
    assert a in (-1, 0, 1)

  def test_never_returns_empty_plan(self):
    """best_seq must always be a usable sequence; a None plan would crash the
    control path on the first tick."""
    _, dbg = self.mpc.plan(np.full(self.mpc.cfg.n_steps, 33.0), 33.0, 0.0, 75.0)
    assert dbg["seq"] is not None
    assert len(dbg["seq"]) == self.mpc.cfg.n_decisions

  def test_commands_up_when_target_above(self):
    sp = 70.0
    v = sp * MPH_TO_MS - self.p.offset
    a, _ = self.mpc.plan(np.full(self.mpc.cfg.n_steps, v + 2.0), v, 0.0, sp)
    assert a == 1

  def test_commands_down_when_target_below(self):
    sp = 70.0
    v = sp * MPH_TO_MS - self.p.offset
    a, _ = self.mpc.plan(np.full(self.mpc.cfg.n_steps, v - 2.0), v, 0.0, sp)
    assert a == -1

  def test_holds_when_on_target(self):
    """At steady state on an exact increment there is nothing to gain, and the
    press cost should stop it fidgeting."""
    sp = 70.0
    v = sp * MPH_TO_MS - self.p.offset
    acts = [self.mpc.plan(np.full(self.mpc.cfg.n_steps, v), v, 0.0, sp)[0] for _ in range(8)]
    assert sum(a != 0 for a in acts) <= 2

  def test_ceiling_allows_holding_max_cruise_speed(self):
    """The setpoint ceiling must clear openpilot's fastest target plus the speedo
    offset, or the top of the usable range is silently unreachable. Guards against
    V_CRUISE_MAX or the fitted offset drifting away from this constant."""
    v_cruise_max_mph = 145.0 * 0.621371  # V_CRUISE_MAX kph
    needed = v_cruise_max_mph + self.p.offset * MS_TO_MPH
    cfg = MpcConfig()
    msg = f"sp_max_mph={cfg.sp_max_mph} cannot hold {v_cruise_max_mph:.1f}mph"
    assert cfg.sp_max_mph >= needed, msg + f" (needs setpoint {needed:.1f}mph)"

  def test_floor_matches_car_minimum_set_speed(self):
    """ICBM's get_minimum_set_speed() is 20mph imperial; commanding below it wastes
    presses the cluster will refuse."""
    assert MpcConfig().sp_min_mph == 20.0

  def test_reaches_max_cruise_speed_closed_loop(self):
    """End to end: asked for openpilot's maximum, the car must actually get there."""
    cfg = MpcConfig()
    c = CruiseButtonController(self.p, cfg)
    dt = cfg.dt
    target = (145.0 * 0.621371) * MPH_TO_MS - self.p.offset
    v, a, sp = target - 2.0, 0.0, 88.0
    hist = [sp]
    for i in range(int(60 / dt)):
      st = c.update(i * dt, v, a, sp, np.full(cfg.n_steps, target), True, False)
      if st.action:
        sp = st.planned_setpoint
      hist.append(sp)
      vv, aa = rollout(self.p, np.array([sp * MPH_TO_MS]), v, a, 0.0, dt,
                       sp_hist=np.array(hist[:-1]) * MPH_TO_MS)
      v, a = float(vv[0]), float(aa[0])
    err = abs(v - target) * MS_TO_MPH
    assert err < 0.5, f"settled {err:.2f}mph short of max cruise speed"

  def test_respects_setpoint_limits(self):
    cfg = MpcConfig(sp_min_mph=60.0, sp_max_mph=62.0)
    m = ButtonMpc(self.p, cfg)
    sp_mph = m._expand(np.full((1, cfg.n_decisions), 1, dtype=int), 61.0)
    assert sp_mph.max() <= 62.0
    sp_mph = m._expand(np.full((1, cfg.n_decisions), -1, dtype=int), 61.0)
    assert sp_mph.min() >= 60.0


class TestController(OpenpilotTestCase):
  def setUp(self):
    self.p = PlantParams()

  def test_inactive_when_not_ready(self):
    c = CruiseButtonController(self.p)
    st = c.update(0.0, 30.0, 0.0, 70.0, np.full(c.cfg.n_steps, 30.0),
                  ready=False, driver_pressing=False)
    assert st.action == 0 and not st.active

  def test_yields_to_driver(self):
    c = CruiseButtonController(self.p)
    st = c.update(0.0, 30.0, 0.0, 70.0, np.full(c.cfg.n_steps, 35.0),
                  ready=True, driver_pressing=True)
    assert st.action == 0
    assert st.reason == "driver pressing"

  def test_press_rate_limited(self):
    c = CruiseButtonController(self.p)
    vd = np.full(c.cfg.n_steps, 40.0)
    first = c.update(0.0, 30.0, 0.0, 70.0, vd, True, False)
    assert first.action != 0
    nxt = c.update(0.01, 30.0, 0.0, 71.0, vd, True, False)
    assert nxt.action == 0 and nxt.reason == "press cooldown"

  def test_no_press_when_clamped(self):
    """Do not spend a press that cannot move the setpoint."""
    cfg = MpcConfig(sp_max_mph=70.0)
    c = CruiseButtonController(self.p, cfg)
    st = c.update(0.0, 40.0, 0.0, 70.0, np.full(cfg.n_steps, 60.0), True, False)
    assert st.action == 0

  def test_setpoint_history_is_one_sample_per_tick(self):
    """sp_hist feeds the plant's deadtime buffer, which indexes by time. Appending
    only on change silently stretched the buffer and biased tracking by 0.5mph."""
    c = CruiseButtonController(self.p)
    vd = np.full(c.cfg.n_steps, 30.0)
    for i in range(10):
      c.update(i * c.cfg.dt, 30.0, 0.0, 70.0, vd, ready=False, driver_pressing=False)
    # reset() clears on not-ready, so drive it ready instead
    c = CruiseButtonController(self.p)
    for i in range(10):
      c.update(i * c.cfg.dt, 30.0, 0.0, 70.0, vd, ready=True, driver_pressing=True)
    assert len(c.sp_hist) == 10

  def test_tracks_between_increments_closed_loop(self):
    """The whole point: hold a speed that falls between two mph increments."""
    cfg = MpcConfig()
    c = CruiseButtonController(self.p, cfg)
    dt = cfg.dt
    target = 74.5 * MPH_TO_MS - self.p.offset
    v, a, sp = target, 0.0, 75.0
    hist = [sp]
    errs = []
    for i in range(int(70 / dt)):
      t = i * dt
      st = c.update(t, v, a, sp, np.full(cfg.n_steps, target), True, False)
      if st.action:
        sp = st.planned_setpoint
      hist.append(sp)
      vv, aa = rollout(self.p, np.array([sp * MPH_TO_MS]), v, a, 0.0, dt,
                       sp_hist=np.array(hist[:-1]) * MPH_TO_MS)
      v, a = float(vv[0]), float(aa[0])
      if t > 20:
        errs.append((v - target) * MS_TO_MPH)
    rms = float(np.sqrt(np.mean(np.square(errs))))
    # a non-dithering controller is stuck at 0.5mph here; require clearly better
    assert rms < 0.35, f"rms={rms:.3f} mph"
