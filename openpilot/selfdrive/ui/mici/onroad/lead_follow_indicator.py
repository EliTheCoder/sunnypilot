import pyray as rl
from openpilot.common.filter_simple import FirstOrderFilter
from openpilot.selfdrive.ui.ui_state import ui_state
from openpilot.system.ui.lib.application import gui_app
from openpilot.system.ui.widgets import Widget

# Must match lead_follow_controller.py
_T_GAP_MIN = 2.0
_T_GAP_MAX = 3.0
_MIN_FOLLOW_DIST = 8.0

_PANEL_SIZE = 56
_PANEL_RADIUS = 0.25
_PANEL_SEGMENTS = 6
_PANEL_COLOR = rl.Color(0, 0, 0, 120)
_MARGIN = 16

_COLOR_TOO_CLOSE = rl.Color(255, 60, 40, 255)
_COLOR_IN_ZONE   = rl.Color(0, 255, 160, 255)
_COLOR_TOO_FAR   = rl.Color(255, 255, 255, 255)

_CHEVRON_W = 22   # half-width of chevron
_CHEVRON_H = 12   # height of each chevron arm
_CHEVRON_GAP = 8  # gap between double chevron rows
_BAR_W = 28
_BAR_H = 5
_BAR_GAP = 7


def _with_alpha(color: rl.Color, alpha: float) -> rl.Color:
  return rl.Color(color.r, color.g, color.b, int(color.a * alpha))


def _draw_chevron_up(cx: float, cy: float, color: rl.Color) -> None:
  rl.draw_triangle(
    rl.Vector2(cx, cy - _CHEVRON_H // 2),
    rl.Vector2(cx + _CHEVRON_W, cy + _CHEVRON_H // 2),
    rl.Vector2(cx - _CHEVRON_W, cy + _CHEVRON_H // 2),
    color,
  )


def _draw_chevron_down(cx: float, cy: float, color: rl.Color) -> None:
  rl.draw_triangle(
    rl.Vector2(cx - _CHEVRON_W, cy - _CHEVRON_H // 2),
    rl.Vector2(cx + _CHEVRON_W, cy - _CHEVRON_H // 2),
    rl.Vector2(cx, cy + _CHEVRON_H // 2),
    color,
  )


class LeadFollowIndicator(Widget):
  def __init__(self):
    super().__init__()
    self._alpha_filter = FirstOrderFilter(0.0, 0.15, 1 / gui_app.target_fps)
    self._lf_enabled = False
    self._d_rel_ema = 0.0
    self._v_ego = 0.0

  def _update_state(self) -> None:
    sm = ui_state.sm
    if sm.valid['longitudinalPlanSP']:
      lf = sm['longitudinalPlanSP'].leadFollow
      self._lf_enabled = lf.enabled
      self._d_rel_ema = lf.dRelEma
    if sm.valid['carState']:
      self._v_ego = sm['carState'].vEgo

    self._alpha_filter.update(1.0 if self._lf_enabled else 0.0)

  def _render(self, rect: rl.Rectangle) -> None:
    alpha = self._alpha_filter.x
    if alpha < 1e-2:
      return

    zone_min = max(self._v_ego * _T_GAP_MIN, _MIN_FOLLOW_DIST)
    zone_max = self._v_ego * _T_GAP_MAX

    if self._d_rel_ema < zone_min:
      color = _COLOR_TOO_CLOSE
      state = "close"
    elif self._d_rel_ema > zone_max:
      color = _COLOR_TOO_FAR
      state = "far"
    else:
      color = _COLOR_IN_ZONE
      state = "zone"

    panel_x = rect.x + rect.width - _PANEL_SIZE - _MARGIN
    panel_y = rect.y + _MARGIN
    panel_rect = rl.Rectangle(panel_x, panel_y, _PANEL_SIZE, _PANEL_SIZE)

    rl.draw_rectangle_rounded(panel_rect, _PANEL_RADIUS, _PANEL_SEGMENTS,
                              _with_alpha(_PANEL_COLOR, alpha))

    cx = panel_x + _PANEL_SIZE / 2
    cy = panel_y + _PANEL_SIZE / 2
    c = _with_alpha(color, alpha)

    if state == "close":
      _draw_chevron_down(cx, cy - _CHEVRON_GAP // 2, c)
      _draw_chevron_down(cx, cy + _CHEVRON_GAP // 2 + _CHEVRON_H, c)
    elif state == "far":
      _draw_chevron_up(cx, cy - _CHEVRON_H - _CHEVRON_GAP // 2, c)
      _draw_chevron_up(cx, cy + _CHEVRON_GAP // 2, c)
    else:
      bar_rect_top = rl.Rectangle(cx - _BAR_W / 2, cy - _BAR_GAP / 2 - _BAR_H, _BAR_W, _BAR_H)
      bar_rect_bot = rl.Rectangle(cx - _BAR_W / 2, cy + _BAR_GAP / 2, _BAR_W, _BAR_H)
      rl.draw_rectangle_rounded(bar_rect_top, 0.5, 4, c)
      rl.draw_rectangle_rounded(bar_rect_bot, 0.5, 4, c)
