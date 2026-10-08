"""Per-agent transcripts keep raw events; panes render them exactly like the main transcript."""

import io

from rich.console import Console
from termflow.live import Rect, ScreenBuffer

from pydantic_ai import PartDeltaEvent, PartEndEvent, PartStartEvent, TextPart, TextPartDelta, ThinkingPart
from pydantic_ai.messages import FunctionToolCallEvent, FunctionToolResultEvent, ToolCallPart, ToolReturnPart
from pydantic_ai_harness.subagents import DelegationStartEvent
from pydantic_clai2.runtime.tasks import task_row
from pydantic_clai2.ui.rendering._rendering import StreamRenderer
from pydantic_clai2.ui.rendering.agent_pane import AgentPane
from pydantic_clai2.ui.rendering.agent_streams import AgentStream, AgentStreams, SentPrompt, plain_renderer

EVENTS = [
    PartStartEvent(index=0, part=ThinkingPart(content='Let me think\n')),
    PartEndEvent(index=0, part=ThinkingPart(content='Let me think\n')),
    PartStartEvent(
        index=1, part=TextPart(content='# Title\n\nSome **bold**.\n\n```python\ndef f():\n    return 1\n```\n')
    ),
    PartEndEvent(index=1, part=TextPart(content='')),
    FunctionToolCallEvent(ToolCallPart('grep', {'pattern': 'todo'}, tool_call_id='c1')),
    FunctionToolResultEvent(ToolReturnPart('grep', 'src/a.py:1:todo', tool_call_id='c1')),
    DelegationStartEvent(
        agent_name='Explore', task='look around', truncated=False, model=None, inherits_tools=False, task_id='a' * 32
    ),
]


def renderer(console: Console) -> StreamRenderer:
    return StreamRenderer(console, stop_loading=lambda: None, smooth=False, renderers=[task_row])


def cells(lines: list[str], *, width: int) -> tuple[list[str], list[int], list[int]]:
    screen = ScreenBuffer(width, len(lines))
    region = screen.region(Rect(0, 0, width, len(lines)))
    for y, line in enumerate(lines):
        region.ansi(0, y, line)
    return screen.chars, screen.fg, screen.attrs


async def test_pane_renders_exactly_like_the_main_transcript() -> None:
    output = io.StringIO()
    console = Console(file=output, width=60, force_terminal=True, color_system='truecolor')
    main = renderer(console)
    console.print('> hello', markup=False, highlight=False)
    console.print()
    for event in EVENTS:
        await main.on_stream_event(event)
    await main.finish()
    expected = output.getvalue().rstrip('\n').split('\n')

    stream = AgentStream(key='a', title='a')
    stream.prompt('hello')
    for event in EVENTS:
        stream.observe(event)
    pane = AgentPane(stream, renderer=renderer)
    screen = ScreenBuffer(60, len(expected))
    pane.draw(screen.region(Rect(0, 0, 60, len(expected))), False)
    await pane.refresh()
    screen = ScreenBuffer(60, len(expected))
    pane.draw(screen.region(Rect(0, 0, 60, len(expected))), False)
    assert (screen.chars, screen.fg, screen.attrs) == cells(expected, width=60)
    text = [screen.row_text(y).rstrip() for y in range(len(expected))]
    assert 'Thinking Let me think' in text and '● Explore [aaaaaaaa] look around' in text


def test_streams_keep_events_and_prompts_in_order() -> None:
    stream = AgentStream(key='a', title='a')
    before = stream.updated
    stream.prompt('steer me', label='steer')
    stream.observe(EVENTS[0])
    assert stream.entries == [SentPrompt(text='steer me', label='steer'), EVENTS[0]]
    assert stream.updated >= before


def test_registry_keeps_start_order_and_first_registration() -> None:
    agents = AgentStreams()
    assert agents.renderer is plain_renderer
    fork = agents.add(AgentStream(key='fork-1', title='fork'))
    assert agents.add(AgentStream(key='fork-1', title='duplicate')) is fork
    assert [stream.key for stream in agents] == ['main', 'fork-1']
    assert agents.get('fork-1') is fork and agents.get('nope') is None and len(agents) == 2


async def draw(pane: AgentPane, *, width: int = 20, height: int = 3) -> list[str]:
    screen = ScreenBuffer(width, height)
    pane.draw(screen.region(Rect(0, 0, width, height)), False)
    await pane.refresh()
    screen = ScreenBuffer(width, height)
    pane.draw(screen.region(Rect(0, 0, width, height)), False)
    return [screen.row_text(y).rstrip() for y in range(height)]


async def test_pane_follows_new_output_until_scrolled_back() -> None:
    stream = AgentStream(key='a', title='a')
    pane = AgentPane(stream, renderer=plain_renderer)
    await pane.refresh()
    assert await draw(pane) == ['Waiting for output\u2026', '', '']
    pane.draw(ScreenBuffer(0, 0).region(Rect(0, 0, 0, 0)), True)
    for index in range(3):
        stream.prompt(f'block {index}')
    stream.observe(PartStartEvent(index=0, part=TextPart(content='streamed\n')))
    stream.observe(PartDeltaEvent(index=0, delta=TextPartDelta(content_delta='more\n')))
    stream.observe(PartEndEvent(index=0, part=TextPart(content='')))
    assert await draw(pane) == ['', 'streamed', 'more']
    assert pane.page == 2
    pane.scroll(-2)
    assert await draw(pane) == ['', '> block 2', '']
    # Nothing new: refreshing again keeps the rendered lines.
    await pane.refresh()
    pane.scroll(10)
    assert await draw(pane) == ['', 'streamed', 'more']
    assert pane.follow
    # A new width replays everything, wrapped to fit.
    assert await draw(pane, width=6, height=2) == ['ed', 'more']
    assert pane.lines[0] == '> '
