"""Per-agent transcripts for `/forks live`: the main conversation, forks, and sub-agents.

A transcript is the agent's raw stream events plus the prompts sent to it. Panes replay them
through the same `StreamRenderer` as the main transcript, so tool calls, thinking, diffs, and
plugin renderers look identical. History lasts the whole session.
"""

import time
from collections.abc import Awaitable, Callable, Iterator, Sequence
from dataclasses import dataclass, field

from rich.console import Console

from pydantic_ai import AgentStreamEvent, PartEndEvent, PartStartEvent, TextPart, ThinkingPart
from pydantic_ai.messages import (
    FunctionToolCallEvent,
    FunctionToolResultEvent,
    ModelMessage,
    ModelRequest,
    ModelResponse,
    RetryPromptPart,
    ToolCallPart,
    ToolReturnPart,
    UserPromptPart,
)
from pydantic_clai2.runtime._session import SteeringPriority
from pydantic_clai2.ui.rendering._rendering import StreamRenderer

Send = Callable[[str, SteeringPriority], str]
"""Deliver a draft to one agent: `'asap'` steers it, `'when_idle'` queues a follow-up. Returns a notice."""


@dataclass(kw_only=True, frozen=True)
class SentPrompt:
    """Input the user sent an agent, echoed like the transcript's `> prompt` line."""

    text: str
    label: str = ''
    """How it was sent, such as `steer` or `queued`; empty for a turn's own prompt."""


Entry = AgentStreamEvent | SentPrompt


def history_entries(messages: Sequence[ModelMessage]) -> list[Entry]:
    """Replay a saved conversation as the events a live run would have streamed, for the same renderer."""
    entries: list[Entry] = []
    for message in messages:
        if isinstance(message, ModelRequest):
            for part in message.parts:
                if isinstance(part, UserPromptPart) and isinstance(part.content, str):
                    entries.append(SentPrompt(text=part.content))
                elif isinstance(part, (ToolReturnPart, RetryPromptPart)):
                    entries.append(FunctionToolResultEvent(part))
        elif isinstance(message, ModelResponse):  # pragma: no branch -- the only other message kind.
            for index, part in enumerate(message.parts):
                if isinstance(part, (TextPart, ThinkingPart)):
                    entries.extend([PartStartEvent(index=index, part=part), PartEndEvent(index=index, part=part)])
                elif isinstance(part, ToolCallPart):
                    entries.append(FunctionToolCallEvent(part))
    return entries


def plain_renderer(console: Console) -> StreamRenderer:
    """A renderer with default settings and no plugin renderers, until the shell supplies its own."""
    return StreamRenderer(console, stop_loading=lambda: None, smooth=False)


@dataclass(kw_only=True, eq=False)
class AgentStream:
    """One agent's transcript, plus how to reach it."""

    key: str
    title: str
    activity: Callable[[], str] = lambda: ''
    send: Send | None = None
    """`None` for the main conversation, whose queue and steering the editor already owns."""
    stop: Callable[[], Awaitable[str]] | None = None
    """Stop this agent (and a sub-agent's descendants); returns a notice. `None` when it cannot be stopped."""
    background: Callable[[], str] | None = None
    """Release a foreground sub-agent so the main run continues; returns a notice."""
    entries: list[Entry] = field(default_factory=list[Entry])
    updated: float = field(default_factory=time.monotonic)

    def prompt(self, text: str, *, label: str = '') -> None:
        """Show input the user sent this agent."""
        self.entries.append(SentPrompt(text=text, label=label))
        self.updated = time.monotonic()

    def observe(self, event: AgentStreamEvent) -> None:
        """Keep one stream event for the panes to render."""
        self.entries.append(event)
        self.updated = time.monotonic()


class AgentStreams:
    """Every agent this shell has run, in the order they started; the main conversation first."""

    def __init__(self) -> None:
        """Start with only the main conversation."""
        self.main = AgentStream(key='main', title='main')
        self._streams: dict[str, AgentStream] = {self.main.key: self.main}
        self.renderer: Callable[[Console], StreamRenderer] = plain_renderer
        """Builds a pane's renderer; the shell supplies one configured like the main transcript."""

    def add(self, stream: AgentStream) -> AgentStream:
        """Register `stream`, or return the one already registered under its key."""
        return self._streams.setdefault(stream.key, stream)

    def get(self, key: str) -> AgentStream | None:
        """The stream registered under `key`, if any."""
        return self._streams.get(key)

    def __iter__(self) -> Iterator[AgentStream]:
        """Streams in start order."""
        return iter(tuple(self._streams.values()))

    def __len__(self) -> int:
        """How many agents have a stream."""
        return len(self._streams)
