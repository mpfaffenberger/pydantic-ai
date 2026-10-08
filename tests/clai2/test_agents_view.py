"""`/forks live`: agent windows over the transcript rows, with the editor still in charge of input."""

import io
import re
from collections.abc import Callable
from pathlib import Path

import anyio
import pytest
from prompt_toolkit.application import create_app_session
from prompt_toolkit.input import create_pipe_input
from prompt_toolkit.output import DummyOutput
from rich.console import Console
from rich.text import Text
from termflow.live import Region, ScreenBuffer, Window, rgb
from termflow.live.app import STATUS_BG
from termflow.tui.completion import CompleteEvent, Document

from pydantic_ai import Agent, AgentStreamEvent, PartStartEvent, TextPart, ThinkingPart
from pydantic_ai.messages import (
    ModelMessage,
    ModelRequest,
    ModelResponse,
    NativeToolCallPart,
    RetryPromptPart,
    SystemPromptPart,
    ToolCallPart,
    ToolReturnPart,
    UserPromptPart,
)
from pydantic_ai.models.test import TestModel
from pydantic_ai_harness.subagents import DelegationTask
from pydantic_clai2 import chat
from pydantic_clai2._app import create_shell
from pydantic_clai2.config.project_settings import ProjectSettings
from pydantic_clai2.config.settings_store import SettingsStore
from pydantic_clai2.runtime._session import SteeringPriority
from pydantic_clai2.ui.prompt.agents_view import KEYS, AgentsView
from pydantic_clai2.ui.rendering import theme
from pydantic_clai2.ui.rendering._rendering import StreamRenderer
from pydantic_clai2.ui.rendering.agent_pane import AgentPane
from pydantic_clai2.ui.rendering.agent_roster import AgentRoster, window
from pydantic_clai2.ui.rendering.agent_streams import AgentStream, AgentStreams, history_entries
from pydantic_clai2.ui.rendering.themed_live import ThemedLiveApp, draw_border
from tests.clai2.surface_terminal import SurfaceTerminal
from tests.clai2.test_live_prompt import editor


def recorder(sent: list[tuple[str, SteeringPriority]]) -> Callable[[str, SteeringPriority], str]:
    def send(text: str, priority: SteeringPriority) -> str:
        sent.append((text, priority))
        return f'sent {text}'

    return send


def agents_with_fork(sent: list[tuple[str, SteeringPriority]]) -> AgentStreams:
    agents = AgentStreams()
    agents.main.prompt('main says hi')
    fork = agents.add(AgentStream(key='fork-1', title='fork #1', activity=lambda: 'thinking', send=recorder(sent)))
    fork.prompt('fork says hi')
    return agents


async def draw(view: AgentsView, *, width: int, height: int) -> list[str]:
    view.frame(width, height, True)
    await view.refresh()
    terminal = SurfaceTerminal(width=width, height=height)
    terminal.write(view.frame(width, height, True))
    return terminal.lines()


def plain(output: io.StringIO) -> str:
    return re.sub(r'\x1b(\[[0-9;?>]*[A-Za-z]|.)', '', output.getvalue())


async def until(condition: Callable[[], bool]) -> None:
    while not condition():
        await anyio.sleep(0.01)


async def test_view_covers_the_transcript_then_replays_held_output() -> None:
    terminal = SurfaceTerminal(width=80, height=24)
    async with editor(output=terminal) as (live, _, _):
        live.console.print('before the view')
        view = AgentsView(agents=agents_with_fork([]), editor=live)
        async with anyio.create_task_group() as tasks:
            tasks.start_soon(view.serve)
            assert view.toggle().startswith('Live agent view:')
            await until(lambda: any('agents (2)' in line for line in terminal.lines()))
            assert live.overlay is view
            assert not any('before the view' in line for line in terminal.lines())
            live.console.print('while covered')
            # The editor keeps its own rows below the windows.
            live.buffer.replace('draft survives')
            live.paint()
            assert any('draft survives' in line for line in terminal.lines())
            assert not any('while covered' in line for line in terminal.lines())
            assert view.toggle() == 'Live agent view closed.'
            await until(lambda: live.overlay is None)
            lines = terminal.lines()
            assert 'before the view' in lines and 'while covered' in lines
            assert not any('fork #1' in line for line in lines)
            # Reopening works, and so does closing before the view first paints.
            view.toggle()
            await until(lambda: live.overlay is view)
            view.toggle()
            await until(lambda: live.overlay is None)
            view.toggle()
            view.toggle()
            await anyio.sleep(0)
            assert live.overlay is None and not view.open
            tasks.cancel_scope.cancel()


async def test_closing_while_a_menu_owns_the_terminal_restores_on_return() -> None:
    terminal = SurfaceTerminal(width=80, height=24)
    async with editor(output=terminal) as (live, _, _):
        live.console.print('kept')
        view = AgentsView(agents=agents_with_fork([]), editor=live)
        async with anyio.create_task_group() as tasks:
            tasks.start_soon(view.serve)
            view.toggle()
            await until(lambda: any('fork #1' in line for line in terminal.lines()))
            async with live.suspended():
                view.toggle()
                await until(lambda: live.overlay is None)
            tasks.cancel_scope.cancel()
        assert 'kept' in terminal.lines()
        assert not any('fork #1' in line for line in terminal.lines())


async def test_roster_focus_selection_scrolling_and_draft_routing() -> None:
    sent: list[tuple[str, SteeringPriority]] = []
    async with editor() as (live, _, _):
        view = AgentsView(agents=agents_with_fork(sent), editor=live)
        live.overlay = view
        # Main selected: drafts keep their usual queue and steering.
        live.buffer.replace('main prompt')
        live.feed('enter')
        assert live.queued_messages == ('main prompt',)
        # The roster starts focused: Up stays on the first agent, Down selects the next.
        live.feed('up')
        assert view.selected == 'main'
        live.feed('down')
        live.feed('down')
        assert view.selected == 'fork-1'
        live.buffer.replace('queue this')
        live.feed('enter')
        assert sent == [('queue this', 'when_idle')] and live.notice == 'sent queue this'
        assert live.buffer.text == ''
        live.buffer.replace('steer this')
        live.feed('alt-enter')
        assert sent[-1] == ('steer this', 'asap') and live.buffer.text == ''
        # Commands and shell lines are the shell's, whichever agent is selected.
        live.buffer.replace('/help')
        live.feed('enter')
        live.buffer.replace('!ls')
        live.feed('enter')
        assert live.queued_messages == ('main prompt', '/help', '!ls')
        live.buffer.replace('/help')
        live.feed('alt-enter')
        assert live.buffer.text == '/help'
        live.buffer.replace('')
        # Tab gives the arrows back to the editor: Up recalls a queued prompt, which Enter rewrites.
        live.feed('tab')
        assert not view.roster_focused
        live.feed('up')
        live.feed('alt-enter')
        live.buffer.replace('main prompt, edited')
        live.feed('enter')
        assert live.queued_messages[-1] == 'main prompt, edited' and len(sent) == 2
        assert view.selected == 'fork-1'
        fork = view.agents.get('fork-1')
        assert fork is not None
        for index in range(20):
            fork.prompt(f'line {index}')
        assert any('line 19' in row for row in await draw(view, width=100, height=12))
        assert view.key('pageup', draft='')
        assert not any('line 19' in row for row in await draw(view, width=100, height=12))
        assert view.key('pagedown', draft='')
        assert any('line 19' in row for row in await draw(view, width=100, height=12))
        # A typed draft keeps Tab for completion; other keys are the editor's.
        assert not view.key('tab', draft='typed') and not view.key('x', draft='')
        live.feed('backtab')
        assert view.roster_focused
        assert view.deliver('anything', steer=True) == 'sent anything'


async def test_roster_beside_or_above_the_selected_agent() -> None:
    agents = AgentStreams()
    agents.main.activity = lambda: 'ready'
    # A still clock: frames take no time, and the layout never touches the editor.
    view = AgentsView(agents=agents, editor=None, clock=lambda: 0.0)  # type: ignore[arg-type]
    for index, activity in enumerate(('tool: grep', 'done', 'failed', 'cancelled')):
        stream = agents.add(AgentStream(key=f'task-{index}', title=f'task {index}', activity=lambda a=activity: a))
        stream.prompt(f'hi {index}')
    wide = await draw(view, width=100, height=10)
    assert wide[0].startswith('\u250f\u2501 agents (5)') and '\u2500 main \u00b7 ready' in wide[0]
    assert wide[1].startswith('\u2503\u25b8  main') and 'ready' in wide[1]
    assert '\u2713 task 1' in wide[3] and '\u2717 task 2' in wide[4] and '\u2013 task 3' in wide[5]
    assert KEYS[:20] in wide[-1] and 'to main: Enter queue' in wide[-1]
    view.selected = 'task-0'
    view.roster_focused = False
    narrow = await draw(view, width=60, height=14)
    assert narrow[0].startswith('\u250c\u2500 agents (5)')
    window = next(index for index, line in enumerate(narrow) if 'task 0 \u00b7 tool: grep' in line)
    assert narrow[window].startswith('\u250f\u2501') and '> hi 0' in narrow[window + 1]
    view.selected = 'gone'
    assert 'main \u00b7 ready' in ''.join(await draw(view, width=100, height=10))


def test_roster_window_keeps_the_selection_visible() -> None:
    assert window(5, selected=4, height=5) == (0, 5)
    assert window(20, selected=0, height=10) == (0, 9)
    assert window(20, selected=9, height=10) == (5, 13)
    assert window(20, selected=19, height=10) == (11, 20)
    assert window(20, selected=7, height=2) == (7, 8)
    streams = [AgentStream(key=f'a{index}', title=f'agent {index}') for index in range(20)]
    roster = AgentRoster(agents=lambda: streams, selected=lambda: 'a9', clock=lambda: 0.0)
    rows = [Text.from_ansi(row).plain for row in roster.rows(width=30, height=10)]
    assert rows[0] == '  \u2026 5 more \u2191' and rows[-1] == '  \u2026 7 more \u2193'
    assert rows[5].startswith('\u25b8  agent 9')
    tight = [Text.from_ansi(row).plain for row in roster.rows(width=30, height=2)]
    assert len(tight) == 1 and tight[0].startswith('\u25b8')
    lost = AgentRoster(agents=lambda: streams, selected=lambda: 'gone', clock=lambda: 0.0)
    assert Text.from_ansi(lost.rows(width=30, height=3)[0]).plain.startswith('\u25b8')


async def test_view_steps_aside_while_a_question_owns_the_terminal() -> None:
    terminal = SurfaceTerminal(width=80, height=24)
    async with editor(output=terminal) as (live, _, _):
        view = AgentsView(agents=agents_with_fork([]), editor=live)
        async with anyio.create_task_group() as tasks:
            tasks.start_soon(view.serve)
            view.toggle()
            await until(lambda: any('agents (2)' in line for line in terminal.lines()))
            live.console.print('held while covered')
            async with live.suspended():
                await until(lambda: 'held while covered' in terminal.lines())
                live.console.print('Which file?')
                # Released, the terminal is in cooked mode, where a newline also returns the carriage.
                assert any(line.strip() == 'Which file?' for line in terminal.lines())
                await anyio.sleep(0.12)
                assert not any('agents (2)' in line for line in terminal.lines())
            await until(lambda: any('agents (2)' in line for line in terminal.lines()))
            view.toggle()
            await until(lambda: live.overlay is None)
            tasks.cancel_scope.cancel()
        lines = [line.strip() for line in terminal.lines()]
        assert lines.index('held while covered') < lines.index('Which file?')


async def test_a_pane_interrupted_mid_render_renders_again() -> None:
    renders = 0
    started, release = anyio.Event(), anyio.Event()

    class Slow(StreamRenderer):
        async def on_stream_event(self, event: AgentStreamEvent) -> None:
            nonlocal renders
            renders += 1
            if renders == 1:
                started.set()
                await release.wait()
            await super().on_stream_event(event)

    stream = AgentStream(key='a', title='a')
    stream.observe(PartStartEvent(index=0, part=TextPart(content='hello\n')))
    pane = AgentPane(stream, renderer=lambda console: Slow(console, stop_loading=lambda: None, smooth=False))
    pane.width = 20
    async with anyio.create_task_group() as tasks:
        tasks.start_soon(pane.refresh)
        await started.wait()
        tasks.cancel_scope.cancel()
    await pane.refresh()
    # The interrupted entry is rendered again from a fresh renderer, once, in full.
    assert renders == 2 and pane.lines == ['hello']


async def test_forks_live_opens_from_the_shell_and_the_chord_closes_it(tmp_path: Path) -> None:
    output, done = io.StringIO(), anyio.Event()

    async def run() -> None:
        await chat(
            Agent(TestModel(custom_output_text='hi there')),
            deps=None,
            console=Console(file=output, force_terminal=True, width=100, height=24),
            store=SettingsStore(tmp_path / 'config.db'),
        )
        done.set()

    with create_pipe_input() as pipe, create_app_session(input=pipe, output=DummyOutput()), anyio.fail_after(10):
        async with anyio.create_task_group() as tasks:
            tasks.start_soon(run)
            pipe.send_text('hello\n')
            # Streamed text is interleaved with footer repaints; the footer reports the finished turn.
            await until(lambda: 'output tokens' in plain(output))
            pipe.send_text('/forks live\n')
            await until(lambda: 'agents (1)' in plain(output))
            start = len(plain(output))
            pipe.send_text('\x18\x01')
            await until(lambda: 'Live agent view closed.' in plain(output)[start:])
            pipe.send_text('/exit\n')
            await done.wait()
    text = plain(output)
    frame = text[text.index('agents (1)') :]
    assert '> hello' in frame and 'hi there' in frame


async def test_forks_live_needs_a_terminal_and_completes(tmp_path: Path) -> None:
    shell = create_shell(
        Agent(TestModel()),
        deps=None,
        plugins=(),
        usage_limits=None,
        console=Console(file=io.StringIO()),
        settings=None,
        store=SettingsStore(tmp_path / 'config.db'),
        builtin_plugins=(),
        project=ProjectSettings(),
        headless=True,
    )
    assert await shell.commands.execute_async('/forks live') == 'The live agent view needs an interactive terminal.'
    assert shell.toggle_agents() == ''
    completions = shell.commands.get_completions(Document('/forks ', 7), CompleteEvent(text_inserted=True))
    assert [completion.text for completion in completions] == ['live']
    with pytest.raises(ValueError, match=r'Usage: /forks \[live\]'):
        await shell.commands.execute_async('/forks extra args')


def test_unchanged_frames_send_almost_nothing_and_never_clear() -> None:
    view = AgentsView(agents=AgentStreams(), editor=None, clock=lambda: 0.0)  # type: ignore[arg-type]
    first = view.frame(100, 10, False)
    second = view.frame(100, 10, False)
    assert '\x1b[2J' not in first + second
    assert len(second) < len(first) // 10


def test_roster_pads_wide_titles_by_columns_and_escapes_titles() -> None:
    streams = [
        AgentStream(key='a', title='\u63a2\u7d22\u8005', activity=lambda: 'done'),
        AgentStream(key='b', title='plain', activity=lambda: 'done'),
    ]
    roster = AgentRoster(agents=lambda: streams, selected=lambda: 'a', clock=lambda: 0.0)
    wide, plain = (Text.from_ansi(row) for row in roster.rows(width=30, height=5))
    assert wide.cell_len == plain.cell_len and wide.plain.endswith('done') and plain.plain.endswith('done')
    many = [AgentStream(key=f'k{index}', title=f'k{index}') for index in range(9)]
    narrow = AgentRoster(agents=lambda: many, selected=lambda: 'k4', clock=lambda: 0.0)
    assert all(Text.from_ansi(row).cell_len <= 8 for row in narrow.rows(width=8, height=4))


async def test_window_title_escapes_control_characters() -> None:
    agents = AgentStreams()
    agents.add(AgentStream(key='x', title='bad\x1b[2Jtitle', activity=lambda: 'tool:\nrun'))
    view = AgentsView(agents=agents, editor=None, clock=lambda: 0.0)  # type: ignore[arg-type]
    view.selected = 'x'
    top = (await draw(view, width=100, height=6))[0]
    assert 'bad\\x1b[2Jtitle \u00b7 tool: run' in top


def test_chrome_follows_the_theme_not_termflows_orange(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv('COLORTERM', 'truecolor')
    view = AgentsView(agents=agents_with_fork([]), editor=None, clock=lambda: 0.0)  # type: ignore[arg-type]
    frames: dict[str, ScreenBuffer] = {}
    for name in ('default', 'catppuccin_latte'):
        with theme.use(lambda name=name: name):
            accent = theme.sgr(theme.ACCENT)
            view.frame(100, 8, True)
            frames[name] = view.app.step(0)
            expected = ScreenBuffer(1, 1)
            expected.region().ansi(0, 0, accent + '\u250f')
            # The focused roster's corner is drawn in this theme's accent, not termflow's fixed orange.
            assert frames[name].fg[0] == expected.fg[0] != rgb(255, 140, 40)
            assert STATUS_BG not in frames[name].bg
    assert frames['default'].fg[0] != frames['catppuccin_latte'].fg[0]
    status = frames['default'].row_text(7)
    assert status.startswith(' agents  to main') and 'fps' not in status
    assert frames['default'].row_text(0).startswith('\u250f\u2501 agents (2)')
    assert '\u250c\u2500 main' in frames['default'].row_text(0)
    tiny = ScreenBuffer(1, 1)
    draw_border(tiny.region(), title='x', focused=True)
    assert tiny.row_text(0) == ' '


async def test_x_stops_and_b_backgrounds_the_selected_agent_from_the_roster() -> None:
    stopped: list[str] = []

    async def stop() -> str:
        stopped.append('fork')
        return 'Cancelling fork #1...'

    async with editor() as (live, _, _):
        agents = agents_with_fork([])
        fork = agents.get('fork-1')
        assert fork is not None
        fork.stop, fork.background = stop, lambda: 'moved'
        view = AgentsView(agents=agents, editor=live)
        live.overlay = view
        # Main has neither action, so the letters are typed.
        live.feed('x')
        assert live.buffer.text == 'x'
        live.buffer.replace('')
        assert view.show(key='fork-1').startswith('Live agent view:') and view.selected == 'fork-1'
        assert view.show(key='fork-1') == 'Live agent view.'
        live.feed('b')
        assert live.notice == 'moved' and live.buffer.text == ''
        live.feed('x')
        assert stopped == []
        await view.run_actions()
        assert stopped == ['fork'] and live.notice == 'Cancelling fork #1...'
        # Typing, or the output focused, keeps the letters for the draft.
        live.buffer.replace('e')
        live.feed('x')
        live.feed('tab')
        assert live.buffer.text == 'ex' and view.roster_focused
        live.buffer.replace('')
        live.feed('tab')
        live.feed('b')
        assert live.buffer.text == 'b'
        view.toggle()


def test_saved_history_replays_as_stream_events() -> None:
    messages: list[ModelMessage] = [
        ModelRequest(parts=[UserPromptPart('go'), UserPromptPart(['image']), SystemPromptPart('rules')]),
        ModelResponse(
            parts=[
                ThinkingPart('hmm'),
                TextPart('answer'),
                ToolCallPart('grep', {}, tool_call_id='c'),
                NativeToolCallPart('web_search', {}, tool_call_id='w'),
            ]
        ),
        ModelRequest(parts=[ToolReturnPart('grep', 'ok', tool_call_id='c'), RetryPromptPart('bad', tool_call_id='d')]),
    ]
    kinds = [type(entry).__name__ for entry in history_entries(messages)]
    assert kinds == [
        'SentPrompt',
        'PartStartEvent',
        'PartEndEvent',
        'PartStartEvent',
        'PartEndEvent',
        'FunctionToolCallEvent',
        'FunctionToolResultEvent',
        'FunctionToolResultEvent',
    ]


async def test_bare_tasks_opens_the_view_on_the_first_task(tmp_path: Path) -> None:
    from tests.clai2.test_forks import Model, shell_for

    shell = shell_for(tmp_path, Model(), io.StringIO())
    async with editor() as (live, _, _):
        shell.agents_view = AgentsView(agents=shell.agents, editor=live)
        assert await shell.tasks_command([]) == 'No delegated tasks in this conversation yet.'
        assert shell.agents_view.open and shell.agents_view.selected == 'main'
        record = DelegationTask(
            id='a' * 32, agent_name='Explore', prompt='look', conversation_id=shell.session.summary.id
        )
        record.status, record.outcome = 'finished', 'ok'
        # A nested child saved first does not win: `/tasks` opens on the first top-level task.
        child = DelegationTask(
            id='b' * 32, agent_name='Plan', prompt='p', conversation_id=record.conversation_id, parent_id=record.id
        )
        shell.tasks.owner.records[child.id] = child
        shell.tasks.owner.records[record.id] = record
        assert await shell.tasks_command([]) == 'Live agent view.'
        assert shell.agents_view.selected == f'task-{record.id}'
        assert shell.toggle_agents() == 'Live agent view closed.'
    await shell.forks.close()


def test_borderless_windows_and_untitled_or_narrow_borders() -> None:
    drawn: list[bool] = []

    class Probe(AgentRoster):
        def draw(self, region: Region, focused: bool) -> None:
            drawn.append(region.width == 10)

    probe = Probe(agents=lambda: (), selected=lambda: '')
    app = ThemedLiveApp([Window('bare', probe, border=False)], layout=lambda area: [area], size=lambda: (10, 3))
    app.step(0)
    assert drawn == [True]
    for width, title in ((20, ''), (6, 'long title')):
        screen = ScreenBuffer(width, 3)
        draw_border(screen.region(), title=title, focused=False)
        assert screen.row_text(0) == '\u250c' + '\u2500' * (width - 2) + '\u2510'


async def test_escape_closes_the_view_before_interrupting_a_turn() -> None:
    async with editor() as (live, _, _):
        view = AgentsView(agents=agents_with_fork([]), editor=live)
        live.overlay = view
        view.toggle()
        # A suggestion list takes Esc first, then the view, then a running turn.
        live.buffer.replace('/he')
        live.refresh_completions()
        await until(lambda: '/help' in ''.join(live.frame()))
        live.feed('escape')
        assert view.open
        live.buffer.replace('')
        live.feed('escape')
        assert not view.open and live.notice == 'Live agent view closed.'
        live.feed('escape')
        started = anyio.Event()

        async def turn() -> None:
            started.set()
            await anyio.sleep_forever()

        async with anyio.create_task_group() as tasks:

            async def run() -> None:
                assert not await live.interrupts.run(turn())

            tasks.start_soon(run)
            await started.wait()
            view.toggle()
            live.feed('escape')
            assert not view.open and live.interrupts.active
            live.feed('escape')
