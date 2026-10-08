"""The live view's roster: one row per agent, its state, and which one is selected."""

import time
from collections.abc import Callable, Sequence

from rich.cells import cell_len
from termflow.live import Region, Widget
from termflow.tui.layout import truncate

from pydantic_clai2.ui.rendering import theme
from pydantic_clai2.ui.rendering.agent_streams import AgentStream
from pydantic_clai2.ui.rendering.tool_output import terminal_text

_SPINNER = '\u280b\u2819\u2839\u2838\u283c\u2834\u2826\u2827\u2807\u280f'
_DONE = frozenset({'done', 'ok', 'completed'})
_FAILED = frozenset({'failed', 'error', 'timeout', 'budget'})
_STOPPED = frozenset({'cancelled', 'interrupted'})
_IDLE = frozenset({'ready', ''})
ACTIVITY_WIDTH = 11


def spinner_frame(now: float) -> str:
    """The agent spinner: a braille frame, the same in the roster and the editor's agent rows."""
    return _SPINNER[int(now * 10) % len(_SPINNER)]


def state_glyph(activity: str, *, now: float) -> str:
    """A spinner frame while an agent works, otherwise a mark for how it ended."""
    if activity in _DONE:
        return f'{theme.sgr(theme.SUCCESS)}\u2713'
    if activity in _FAILED:
        return f'{theme.sgr(theme.ERROR)}\u2717'
    if activity in _STOPPED:
        return f'{theme.sgr(theme.WARNING)}\u2013'
    if activity in _IDLE:
        return ' '
    return f'{theme.sgr(theme.ACCENT)}{spinner_frame(now)}'


def window(count: int, *, selected: int, height: int) -> tuple[int, int]:
    """The first and last row index to show so `selected` stays visible in `height` rows."""
    if count <= height:
        return 0, count
    if height < 3:
        # No room for "more" markers; the selected agent alone (the region clips the rest).
        return selected, selected + 1

    def span(start: int) -> tuple[int, int]:
        # A "more" marker row takes the place of hidden agents above and below.
        end = start + height - (start > 0)
        return (start, end) if end >= count else (start, end - 1)

    # Every window that fills the height: the last one starts where the final agent reaches the bottom.
    showing = [span(start) for start in range(count - height + 2) if span(start)[0] <= selected < span(start)[1]]
    # Keep the selection near the middle, preferring the earlier window on a tie.
    return min(showing, key=lambda bounds: abs((bounds[0] + bounds[1]) / 2 - selected - 0.5))


class AgentRoster(Widget):
    """Draw the agents as rows; the view moves the selection, the roster keeps it visible."""

    def __init__(
        self,
        *,
        agents: Callable[[], Sequence[AgentStream]],
        selected: Callable[[], str],
        clock: Callable[[], float] = time.monotonic,
    ) -> None:
        """Read the agent list and the selected key on every frame."""
        self.agents = agents
        self.selected = selected
        self.clock = clock

    def rows(self, *, width: int, height: int) -> list[str]:
        """Styled rows: a marker, a state glyph, the name, and the activity, with "more" markers."""
        streams = list(self.agents())
        keys = [stream.key for stream in streams]
        selected = self.selected()
        chosen = keys.index(selected) if selected in keys else 0
        start, end = window(len(streams), selected=chosen, height=height)
        muted, reset, now = theme.sgr(theme.MUTED), '\x1b[0m', self.clock()
        markers = height >= 3
        rows = [truncate(f'{muted}  \u2026 {start} more \u2191', width) + reset] if start and markers else []
        # Marker, glyph, and two spaces, then the name; the activity gets up to ACTIVITY_WIDTH columns.
        name_width = max(6, width - 4 - ACTIVITY_WIDTH)
        for index in range(start, end):
            stream = streams[index]
            activity = ' '.join(terminal_text(stream.activity()).split())
            marker = f'{theme.sgr(theme.ACCENT, bold=True)}\u25b8' if index == chosen else ' '
            name = truncate(terminal_text(stream.title), name_width)
            # Pad by terminal columns, so wide characters keep the activity column aligned.
            name += ' ' * max(0, name_width - cell_len(name))
            row = f'{marker}{reset}{state_glyph(activity, now=now)}{reset} {name} {muted}{activity}{reset}'
            rows.append(truncate(row, width) + reset)
        if end < len(streams) and markers:
            rows.append(truncate(f'{muted}  \u2026 {len(streams) - end} more \u2193', width) + reset)
        return rows

    def draw(self, region: Region, focused: bool) -> None:
        """Paint the rows that fit."""
        for y, row in enumerate(self.rows(width=region.width, height=region.height)):
            region.ansi(0, y, row)
