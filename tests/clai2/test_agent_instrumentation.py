"""The process-wide instrumentation default the `observability` plugin installs while it is loaded."""

from collections.abc import Generator

import pytest
from opentelemetry.sdk.trace import TracerProvider
from opentelemetry.trace import NoOpTracer

from pydantic_ai import Agent
from pydantic_ai.models.instrumented import InstrumentationSettings
from pydantic_clai2.runtime import instrumentation
from pydantic_clai2.runtime.instrumentation import default_tracer, instrument_agents


@pytest.fixture(autouse=True)
def uninstrumented() -> Generator[None]:
    Agent.instrument_all(False)
    yield
    Agent.instrument_all(False)


def current() -> InstrumentationSettings | bool:
    return Agent._instrument_default  # pyright: ignore[reportPrivateUsage]


def settings() -> InstrumentationSettings:
    return InstrumentationSettings(tracer_provider=TracerProvider())


def test_install_and_restore_bring_back_the_previous_default() -> None:
    user = settings()
    Agent.instrument_all(user)
    mine = settings()
    restore = instrument_agents(mine)
    assert current() is mine
    restore()
    assert current() is user
    restore()
    assert current() is user and not instrumentation._installed.settings  # pyright: ignore[reportPrivateUsage]


@pytest.mark.parametrize('newest_first', [True, False])
def test_two_installs_unwind_in_either_order(newest_first: bool) -> None:
    first, second = settings(), settings()
    restore_first, restore_second = instrument_agents(first), instrument_agents(second)
    assert current() is second
    if newest_first:
        restore_second()
        assert current() is first
        restore_first()
    else:
        restore_first()
        assert current() is second, 'an older install leaving does not change the default'
        restore_second()
    assert current() is False


def test_a_default_changed_elsewhere_is_kept() -> None:
    restore = instrument_agents(settings())
    Agent.instrument_all(True)
    restore()
    assert current() is True


def test_default_tracer_follows_the_default() -> None:
    assert isinstance(default_tracer(), NoOpTracer)
    mine = settings()
    restore = instrument_agents(mine)
    assert default_tracer() is mine.tracer
    restore()
    Agent.instrument_all(True)
    assert not isinstance(default_tracer(), NoOpTracer)
