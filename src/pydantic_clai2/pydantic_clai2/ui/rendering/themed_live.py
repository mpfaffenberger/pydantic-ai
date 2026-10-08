"""termflow's `LiveApp`, with window borders and the status bar in the session's `/theme` colours.

termflow draws both in fixed colours (an orange focus border on a near-black status bar). Here
they use CLAI's brand roles through `theme.sgr`, so a bundled palette restyles them too. The
frame-rate counter is left out: it says nothing useful to a user and crowded the key hints.
"""

from termflow.live import LiveApp, Rect, Region, Window
from termflow.tui.layout import truncate

from pydantic_clai2.ui.rendering import theme

_LIGHT = ('\u250c', '\u2510', '\u2514', '\u2518', '\u2500', '\u2502')
_HEAVY = ('\u250f', '\u2513', '\u2517', '\u251b', '\u2501', '\u2503')
_RESET = '\x1b[0m'


def draw_border(region: Region, *, title: str, focused: bool) -> None:
    """A box around `region`: heavy in the accent colour when focused, light and muted otherwise."""
    width, height = region.width, region.height
    if width < 2 or height < 2:
        return
    left_top, right_top, left_bottom, right_bottom, across, down = _HEAVY if focused else _LIGHT
    line = theme.sgr(theme.ACCENT if focused else theme.MUTED)
    region.ansi(0, 0, f'{line}{left_top}{across * (width - 2)}{right_top}{_RESET}')
    region.ansi(0, height - 1, f'{line}{left_bottom}{across * (width - 2)}{right_bottom}{_RESET}')
    for y in range(1, height - 1):
        region.ansi(0, y, f'{line}{down}{_RESET}')
        region.ansi(width - 1, y, f'{line}{down}{_RESET}')
    if title and width > 6:
        label = truncate(f' {title} ', width - 4)
        region.ansi(2, 0, f'{theme.sgr(theme.ACCENT if focused else theme.MUTED, bold=focused)}{label}{_RESET}')


class ThemedLiveApp(LiveApp):
    """A `LiveApp` whose chrome follows `/theme`; layout, focus, and widgets are termflow's."""

    def _draw_window(self, region: Region, window: Window, focused: bool) -> None:
        if not window.border:
            window.widget.draw(region, focused)
            return
        draw_border(region, title=window.title, focused=focused)
        window.widget.draw(region.sub(Rect(1, 1, region.width - 2, region.height - 2)), focused)

    def _draw_status(self, region: Region) -> None:
        extra = self.status() if self.status is not None else ''
        accent, muted = theme.sgr(theme.ACCENT, bold=True), theme.sgr(theme.MUTED)
        divider = f'  \u2502 {extra}' if extra else ''
        text = f'{accent} {self.title}{_RESET}{muted}  {self.hints}{divider}'
        region.ansi(0, 0, truncate(text, region.width) + _RESET)
