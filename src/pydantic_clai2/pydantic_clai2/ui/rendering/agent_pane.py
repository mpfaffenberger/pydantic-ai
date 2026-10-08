"""A `LiveApp` widget showing one agent's transcript, rendered exactly like the main transcript."""

import io
from collections.abc import Callable

from rich.console import Console
from termflow.live import Region, Widget

from pydantic_clai2.ui.rendering import theme
from pydantic_clai2.ui.rendering._rendering import StreamRenderer
from pydantic_clai2.ui.rendering.agent_streams import AgentStream, SentPrompt
from pydantic_clai2.ui.rendering.tool_output import terminal_text


class AgentPane(Widget):
    """Replay the agent's events through a `StreamRenderer` at the pane's width.

    New events render incrementally; a new width replays the transcript from the start.
    `refresh` renders and `draw` paints, so the frame itself never does rendering work.
    """

    def __init__(self, stream: AgentStream, *, renderer: Callable[[Console], StreamRenderer]) -> None:
        """Show `stream`, starting at its newest output."""
        self.stream = stream
        self.top = 0
        self.follow = True
        self.page = 1
        self.width = 0
        """The width the pane was last drawn at; `refresh` renders for it."""
        self._make = renderer
        self._rendered_width = 0
        self._output = io.StringIO()
        self._console = Console(file=self._output)
        self._renderer: StreamRenderer | None = None
        self._consumed = 0
        self.lines: list[str] = []

    def scroll(self, delta: int) -> None:
        """Move the viewport; reaching the bottom again resumes following."""
        self.top = max(0, self.top + delta)
        self.follow = False

    async def refresh(self) -> None:
        """Render entries added since the last refresh, or everything again after a width change."""
        if self.width <= 0:
            return
        if self._renderer is None or self.width != self._rendered_width:
            self._output = io.StringIO()
            # A terminal console, so diffs and code keep their colours; the pane turns ANSI into cells.
            self._console = Console(file=self._output, width=self.width, force_terminal=True, color_system='truecolor')
            self._renderer = self._make(self._console)
            self._rendered_width, self._consumed = self.width, 0
        entries = self.stream.entries
        if self._consumed == len(entries):
            return
        try:
            for entry in entries[self._consumed :]:
                if isinstance(entry, SentPrompt):
                    await self._renderer.finish()
                    label = f'{entry.label}: ' if entry.label else ''
                    self._console.print(f'> {label}{terminal_text(entry.text)}', markup=False, highlight=False)
                    self._console.print()
                else:
                    await self._renderer.on_stream_event(entry)
                self._consumed += 1
        except BaseException:
            # Closing the view mid-render leaves the renderer half way through an entry; start over next time.
            self._renderer = None
            raise
        text = self._output.getvalue().replace('\r', '').rstrip('\n')
        self.lines = text.split('\n') if text else []

    def draw(self, region: Region, focused: bool) -> None:
        """Paint the visible slice, or a waiting notice before the agent has said anything."""
        if region.width <= 0 or region.height <= 0:
            return
        self.width = region.width
        total = len(self.lines)
        if not total:
            region.ansi(0, 0, f'{theme.sgr(theme.MUTED)}Waiting for output\u2026\x1b[0m')
            return
        self.page = max(1, region.height - 1)
        bottom = max(0, total - region.height)
        if self.follow or self.top >= bottom:
            self.top, self.follow = bottom, True
        for y, line in enumerate(self.lines[self.top : self.top + region.height]):
            region.ansi(0, y, line)
