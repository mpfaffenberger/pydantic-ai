"""`/forks live`: a roster of every agent and a window on the selected one, above the shell's editor.

Both are termflow `LiveApp` windows drawn over the transcript rows. The editor below keeps its
draft, history, completion, queue, and commands. Transcript output is held and replayed when
the view closes, and the view steps aside while a menu or question owns the terminal.

Tab moves focus between the roster and the agent window. With the roster focused, Up/Down
select an agent. Enter queues a follow-up for the selected agent and Alt+Enter steers it;
with the main conversation selected, both keep their usual meaning.
"""

import time
from collections.abc import Awaitable, Callable

import anyio
from termflow.live import Rect, ScreenBuffer, Window, hsplit, render_diff, vsplit

from pydantic_clai2.cli.shell_passthrough import shell_command
from pydantic_clai2.commands import is_command_input
from pydantic_clai2.ui import telemetry
from pydantic_clai2.ui.prompt.live_prompt import LivePrompt
from pydantic_clai2.ui.rendering.agent_pane import AgentPane
from pydantic_clai2.ui.rendering.agent_roster import AgentRoster
from pydantic_clai2.ui.rendering.agent_streams import AgentStream, AgentStreams
from pydantic_clai2.ui.rendering.themed_live import ThemedLiveApp
from pydantic_clai2.ui.rendering.tool_output import terminal_text

FRAME_SECONDS = 0.05
SIDE_BY_SIDE_COLUMNS = 80
"""Narrower terminals put the roster above the agent window instead of beside it."""
KEYS = 'Tab focus \u00b7 \u2191\u2193 agent \u00b7 x stop \u00b7 b background \u00b7 PgUp/PgDn scroll \u00b7 Esc close'


def roster_width(width: int) -> int:
    """About a third of the terminal, between 28 and 44 columns."""
    return max(28, min(44, width // 3))


class AgentsView:
    """Open, close, and paint the roster and agent window; route the editor's keys and drafts while open."""

    def __init__(
        self, *, agents: AgentStreams, editor: LivePrompt, clock: Callable[[], float] = time.monotonic
    ) -> None:
        """Bind the shell's agent streams and the editor whose transcript rows the view covers."""
        self.agents = agents
        self.editor = editor
        self.clock = clock
        self.open = False
        self.selected = agents.main.key
        self.roster_focused = True
        self._panes: dict[str, AgentPane] = {}
        self._size = (0, 0)
        self._previous: ScreenBuffer | None = None
        self._last = clock()
        self._wanted = anyio.Event()
        self._scope: anyio.CancelScope | None = None
        self._actions: list[Callable[[], Awaitable[str]]] = []
        self._roster = Window('agents', AgentRoster(agents=lambda: tuple(agents), selected=lambda: self.selected))
        self._window = Window('main', self._pane(agents.main))
        self.app = ThemedLiveApp(
            [self._roster, self._window],
            layout=self._layout,
            title='agents',
            hints='',
            status=lambda: KEYS,
            size=lambda: self._size,
            clock=clock,
        )

    def show(self, *, key: str) -> str:
        """Open the view (if closed) with the roster focused on agent `key`."""
        self.selected, self.roster_focused = key, True
        return self.toggle() if not self.open else 'Live agent view.'

    def close(self) -> str | None:
        """Esc closes the view; `None` when it is not open, so Esc keeps its usual meaning."""
        return self.toggle() if self.open else None

    def toggle(self) -> str:
        """Open or close the view; `serve` does the drawing."""
        self.open = not self.open
        telemetry.record('agents view', open=self.open, agents=len(self.agents))
        if not self.open:
            if self._scope is not None:
                self._scope.cancel()
            return 'Live agent view closed.'
        self._wanted.set()
        return 'Live agent view: Tab moves focus, \u2191\u2193 pick an agent, Enter queues, Alt+Enter steers.'

    async def serve(self) -> None:
        """Run for the editor's lifetime, showing the view whenever it is open."""
        while True:
            await self._wanted.wait()
            self._wanted = anyio.Event()
            if not self.open:
                continue
            with anyio.CancelScope() as scope:
                self._scope = scope
                self.editor.overlay = self
                try:
                    await self._show()
                finally:
                    self.editor.overlay = None
            self._scope = None

    async def _show(self) -> None:
        surface = self.editor.output
        while True:
            # A menu or question owns the terminal: let its output through, and cover again after.
            while self.editor.is_suspended:
                await anyio.sleep(FRAME_SECONDS)
            self._previous = None
            with surface.covered():
                while not self.editor.is_suspended:
                    await self.run_actions()
                    await self.refresh()
                    surface.paint_cover(self.frame)
                    await anyio.sleep(FRAME_SECONDS)

    async def run_actions(self) -> None:
        """Run stop requests from the keyboard, showing each one's notice in the editor footer."""
        actions, self._actions = self._actions, []
        for action in actions:
            self.editor.notice = await action()

    async def refresh(self) -> None:
        """Render new output into the selected agent's pane, so painting a frame only copies cells."""
        await self._pane(self._selected()).refresh()

    def frame(self, width: int, height: int, full: bool) -> str:
        """The cell updates that turn the last frame into this one, for `PromptSurface.paint_cover`."""
        stream = self._selected()
        title, activity = (' '.join(terminal_text(text).split()) for text in (stream.title, stream.activity()))
        self._window.title = f'{title} \u00b7 {activity}' if activity else title
        self._window.widget = self._pane(stream)
        self._roster.title = f'agents ({len(self.agents)})'
        self.app.focus(self._roster if self.roster_focused else self._window)
        # The routing hint leads, so a narrow status bar cuts the key legend first.
        self.app.hints = f'to {title}: Enter queue \u00b7 Alt+Enter steer'
        self._size = (width, height)
        now = self.clock()
        elapsed, self._last = now - self._last, now
        screen = self.app.step(elapsed)
        if full or self._previous is None or self._previous.size != screen.size:
            # Against a frame of cells that can never appear, every cell is rewritten in place,
            # without the full-screen clear that would also wipe the editor's rows.
            unseen = ScreenBuffer(width, height)
            unseen.region().fill('\x00')
            self._previous = unseen
        previous, self._previous = self._previous, screen
        return render_diff(previous, screen)

    def key(self, key: str, *, draft: str) -> bool:
        """Route the view's keys; anything else is the editor's.

        Tab moves focus on an empty draft. With the roster focused, Up/Down pick an agent and, on an
        empty draft, `x` stops it and `b` backgrounds a sub-agent. PgUp/PgDn scroll.
        """
        if key in ('tab', 'backtab') and not draft:
            self.roster_focused = not self.roster_focused
            return True
        if key in ('up', 'down') and self.roster_focused:
            keys = [stream.key for stream in self.agents]
            index = keys.index(self._selected().key) + (-1 if key == 'up' else 1)
            self.selected = keys[max(0, min(len(keys) - 1, index))]
            return True
        stream = self._selected()
        if self.roster_focused and not draft and key == 'x' and stream.stop is not None:
            self._actions.append(stream.stop)
            return True
        if self.roster_focused and not draft and key == 'b' and stream.background is not None:
            self.editor.notice = stream.background()
            return True
        if key in ('pageup', 'pagedown'):
            pane = self._pane(self._selected())
            pane.scroll(pane.page if key == 'pagedown' else -pane.page)
            return True
        return False

    def deliver(self, text: str, *, steer: bool) -> str | None:
        """Send a prompt to the selected fork or sub-agent; commands and the main agent keep their path."""
        stream = self._selected()
        if stream.send is None or is_command_input(text) or shell_command(text) is not None:
            return None
        return stream.send(text, 'asap' if steer else 'when_idle')

    def _selected(self) -> AgentStream:
        return self.agents.get(self.selected) or self.agents.main

    def _pane(self, stream: AgentStream) -> AgentPane:
        if stream.key not in self._panes:
            self._panes[stream.key] = AgentPane(stream, renderer=self.agents.renderer)
        return self._panes[stream.key]

    def _layout(self, area: Rect) -> list[Rect]:
        if area.width >= SIDE_BY_SIDE_COLUMNS:
            left = roster_width(area.width)
            return hsplit(area, left, area.width - left)
        # Stacked: the roster keeps its rows (plus a border), up to a third of the height.
        rows = min(len(self.agents) + 2, max(3, area.height // 3))
        return vsplit(area, rows, area.height - rows)
