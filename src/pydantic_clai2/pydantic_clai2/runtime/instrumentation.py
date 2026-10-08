"""The process-wide Pydantic AI instrumentation default, for agents that bring no `Instrumentation` of their own.

Core reads it through `Agent.instrument_all`, so it reaches the agents CLAI and harness build inside a turn or a
command (the compaction summariser, the session namer, delegated children), not only the ones a plugin's
capabilities are passed to. An agent with its own `instrument` setting or `Instrumentation` capability keeps it.
"""

from collections.abc import Callable
from dataclasses import dataclass, field

from opentelemetry.trace import NoOpTracer, Tracer

from pydantic_ai import Agent
from pydantic_ai.models.instrumented import InstrumentationSettings


@dataclass
class _Installed:
    """What `instrument_agents` changed: the default before the first install, then each install in order."""

    before: InstrumentationSettings | bool = False
    settings: list[InstrumentationSettings] = field(default_factory=list[InstrumentationSettings])


_installed = _Installed()


def _current() -> InstrumentationSettings | bool:
    # Core offers `Agent.instrument_all` to set the default but no public way to read it back.
    return Agent._instrument_default  # pyright: ignore[reportPrivateUsage]


def instrument_agents(settings: InstrumentationSettings) -> Callable[[], None]:
    """Make `settings` the default until the returned function is called; calling it again does nothing.

    The newest install wins. Removing it brings back the previous one still installed, then the default from
    before the first install, unless something else has changed the default since.
    """
    if not _installed.settings:
        _installed.before = _current()
    _installed.settings.append(settings)
    Agent.instrument_all(settings)

    def restore() -> None:
        installed = _installed.settings
        if not any(entry is settings for entry in installed):
            return
        newest = installed[-1] is settings
        installed[:] = [entry for entry in installed if entry is not settings]
        if newest and _current() is settings:
            Agent.instrument_all(installed[-1] if installed else _installed.before)

    return restore


def default_tracer() -> Tracer:
    """The tracer an agent without its own instrumentation would use; a no-op one when the default is off."""
    current = _current()
    if current is True:
        current = InstrumentationSettings()
    return current.tracer if current else NoOpTracer()
