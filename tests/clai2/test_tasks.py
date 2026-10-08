"""Task rendering, live inspection, and stock-agent integration."""

import asyncio
import io
import time
from collections.abc import AsyncIterable, AsyncIterator
from pathlib import Path

import pytest
from rich.console import Console

from pydantic_ai import Agent, AgentStreamEvent, RunContext
from pydantic_ai.messages import (
    ModelMessage,
    ModelRequest,
    ModelResponse,
    PartDeltaEvent,
    PartEndEvent,
    PartStartEvent,
    TextPart,
    TextPartDelta,
    ToolCallPart,
    UserPromptPart,
)
from pydantic_ai.models.function import AgentInfo, DeltaToolCall, DeltaToolCalls, FunctionModel
from pydantic_ai_harness.coder import Coder
from pydantic_ai_harness.subagents import (
    DelegationEndEvent,
    DelegationReports,
    DelegationStartEvent,
    DelegationTask,
    DelegationTaskEvent,
    SubAgent,
    SubAgents,
)
from pydantic_clai2._app import create_stock_agent
from pydantic_clai2.runtime._session import Session
from pydantic_clai2.runtime.sandbox_calls import DelegationToolCallEvent
from pydantic_clai2.runtime.tasks import TaskPresentation, Tasks, task_row, task_tree
from pydantic_clai2.ui.rendering.agent_streams import SentPrompt


def task(*, task_id: str = 'a' * 32, parent_id: str | None = None) -> DelegationTask:
    return DelegationTask(
        id=task_id, parent_id=parent_id, agent_name='worker', prompt='inspect', conversation_id='root'
    )


def test_panel_tree_and_completion_retention() -> None:
    ui = Tasks(console=Console(file=io.StringIO()), conversation_id=lambda: 'root', directory=None)
    parent, child = task(), task(task_id='b' * 32, parent_id='a' * 32)
    ui.owner.records = {parent.id: parent, child.id: child}
    assert [(depth, record.id) for depth, record in task_tree(ui.records())] == [(0, parent.id), (1, child.id)]
    assert '1 descendants' in '\n'.join(ui.rows('*'))
    child.status, child.outcome, child.finished_at = 'finished', 'ok', time.time()
    assert '[bbbbbbbb]' not in '\n'.join(ui.rows('*'))
    child.outcome = 'failed'
    assert '[bbbbbbbb]' in '\n'.join(ui.rows('*'))
    child.finished_at -= 31
    assert '[bbbbbbbb]' not in '\n'.join(ui.rows('*'))


@pytest.mark.parametrize('name', ['Explore', 'Plan', 'general-purpose'])
async def test_stock_managed_specialists_and_general_purpose(tmp_path: Path, name: str) -> None:
    prompts: list[str] = []

    def respond(messages: list[ModelMessage], info: AgentInfo) -> ModelResponse:
        request = messages[0]
        assert isinstance(request, ModelRequest)
        prompt = next(part.content for part in request.parts if isinstance(part, UserPromptPart))
        assert isinstance(prompt, str)
        if len(messages) == 1:
            prompts.append(prompt)
            names = {tool.name for tool in info.function_tools}
            if prompt == 'parent':
                assert 'delegate_task' in names
                return ModelResponse(parts=[ToolCallPart('delegate_task', {'agent_name': name, 'task': 'child'})])
            if name in ('Explore', 'Plan'):
                assert names == {'read_file', 'list_directory', 'search_files', 'find_files', 'file_info'}
                assert not {'shell', 'run_code', 'write_file', 'edit_file', 'delegate_task'} & names
            else:
                assert {'shell', 'write_file', 'delegate_task'} <= names
        return ModelResponse(parts=[TextPart('evidence')])

    async def stream(messages: list[ModelMessage], info: AgentInfo) -> AsyncIterator[str | DeltaToolCalls]:
        for index, part in enumerate(respond(messages, info).parts):
            if isinstance(part, TextPart):
                yield part.content
            else:
                assert isinstance(part, ToolCallPart)
                yield {index: DeltaToolCall(name=part.tool_name, json_args=part.args_as_json_str())}

    ui = Tasks(console=Console(file=io.StringIO()), conversation_id=lambda: 'root', directory=None)
    session = Session(
        create_stock_agent(FunctionModel(function=respond, stream_function=stream)),
        deps=None,
        workspace=tmp_path,
        plugins=[Coder(repo_context=False)],
    )
    session.delegations = ui.owner
    async with ui.owner.opened():
        result = await session.prompt('parent')
        assert result.output == 'evidence'
        (record,) = ui.owner.records.values()
        assert record.outcome == 'ok', record.output
        assert record.resumable == (name == 'general-purpose')
        assert prompts == ['parent', 'child']


async def test_partial_stream_and_lifecycle_routing() -> None:
    output = io.StringIO()
    console = Console(file=output)
    ui = Tasks(console=console, conversation_id=lambda: 'root', directory=None)
    record = task()
    ui.owner.records[record.id] = record
    start = PartStartEvent(index=0, part=TextPart('first'))
    await ui.observe(DelegationTaskEvent(task=record, event=start))
    await ui.observe(DelegationTaskEvent(task=record, event=PartDeltaEvent(index=0, delta=TextPartDelta(' second'))))
    assert record.messages == []
    await ui.observe(DelegationTaskEvent(task=record, event=PartEndEvent(index=0, part=TextPart('first second'))))
    assert task_row(start) is None
    event = DelegationStartEvent(agent_name='self', task='hello', truncated=False, inherits_tools=True, model=None)
    await ui.observe(DelegationTaskEvent(task=record, event=event))
    assert 'general-purpose' in output.getvalue()
    received: list[object] = []

    async def sink(event: object) -> None:
        received.append(event)

    ui.sink = sink
    await ui.observe(DelegationTaskEvent(task=record, event=event))
    assert received == [event]
    record.conversation_id = 'other'
    await ui.observe(DelegationTaskEvent(task=record, event=event))
    assert received == [event]
    record.conversation_id = 'root'
    record.status = 'finished'
    await ui.observe(DelegationTaskEvent(task=record))
    assert ui.rows('*') == ()
    assert 'No foreground' in ui.promote()
    assert ui.resolve('aaaa') is record
    with pytest.raises(ValueError, match='ambiguous'):
        ui.resolve('z')
    record.status = 'running'
    record.backgroundable = False
    assert 'workspace' in ui.promote()


async def test_presentation_classifies_capability_owned_tool() -> None:
    def respond(messages: list[ModelMessage], info: AgentInfo) -> ModelResponse:
        if len(messages) == 1:
            return ModelResponse(parts=[ToolCallPart('renamed_delegate', {'agent_name': 'worker', 'task': 'go'})])
        return ModelResponse(parts=[TextPart('done')])

    async def stream(messages: list[ModelMessage], info: AgentInfo) -> AsyncIterator[str | DeltaToolCalls]:
        for part in respond(messages, info).parts:
            if isinstance(part, TextPart):
                yield part.content
            else:
                assert isinstance(part, ToolCallPart)
                yield {0: DeltaToolCall(name=part.tool_name, json_args=part.args_as_json_str())}

    events: list[AgentStreamEvent] = []

    async def receive(ctx: RunContext[object], stream: AsyncIterable[AgentStreamEvent]) -> None:
        async for event in stream:
            events.append(event)

    from pydantic_ai.models.test import TestModel

    child = Agent(TestModel(custom_output_text='child'), deps_type=object, name='worker')
    parent = Agent(
        FunctionModel(function=respond, stream_function=stream),
        deps_type=object,
        capabilities=[
            SubAgents(agents=[SubAgent(child)], tool_name='renamed_delegate', agent_folders=None),
            TaskPresentation(),
        ],
    )
    await parent.run('go', event_stream_handler=receive)
    assert any(isinstance(event, DelegationToolCallEvent) for event in events)


async def test_promote_all_foreground_siblings() -> None:
    import asyncio

    ui = Tasks(console=Console(file=io.StringIO()), conversation_id=lambda: 'root', directory=None)
    started = asyncio.Event()

    async def child(record: DelegationTask) -> str:
        started.set()
        await asyncio.Event().wait()
        return 'unreachable'  # pragma: no cover

    async with ui.owner.opened():
        pending = asyncio.create_task(
            ui.owner.delegate(
                agent_name='worker',
                prompt='',
                conversation_id='root',
                model=None,
                background=False,
                resume=None,
                run=child,
            )
        )
        await started.wait()
        assert '1 task(s) moved' in ui.promote()
        assert 'not a result' in await pending


@pytest.mark.parametrize('success', [True, False])
def test_terminal_row_outcomes(success: bool) -> None:
    event = DelegationEndEvent(
        agent_name='worker',
        outcome='ok' if success else 'failed',
        output='untrusted',
        truncated=False,
        usage=None,
        duration_seconds=1.0,
    )
    row = task_row(event)
    if success:
        assert row is None
    else:
        assert row is not None
        assert 'Ctrl+X Ctrl+A to watch' in row.plain and 'untrusted' not in row.plain


async def test_presentation_keeps_ordinary_tool_events() -> None:
    from pydantic_ai.messages import FunctionToolCallEvent
    from pydantic_ai.models.test import TestModel

    def ordinary() -> str:
        return 'tool result'

    agent = Agent(TestModel(), deps_type=object, tools=[ordinary], capabilities=[TaskPresentation()])
    events: list[AgentStreamEvent] = []

    async def receive(ctx: RunContext[object], stream: AsyncIterable[AgentStreamEvent]) -> None:
        async for event in stream:
            events.append(event)

    await agent.run('go', event_stream_handler=receive)
    assert any(isinstance(event, FunctionToolCallEvent) for event in events)
    assert not any(isinstance(event, DelegationToolCallEvent) for event in events)


async def test_task_commands_and_plugin_lifetime(tmp_path: Path) -> None:
    from prompt_toolkit.history import InMemoryHistory

    from pydantic_clai2.ui.prompt.live_prompt import LivePrompt
    from tests.clai2.test_forks import Model, shell_for

    shell = shell_for(tmp_path, Model(), io.StringIO())
    assert shell.tasks.owner.directory == tmp_path / 'sessions.db.tasks'
    record = task()
    record.conversation_id = shell.session.summary.id
    shell.tasks.owner.records[record.id] = record
    assert shell.plugins_busy('/plugins list')
    assert not shell.plugins_busy('/tasks')
    with pytest.raises(ValueError, match='Usage'):
        await shell.tasks_command(['bad'])
    async with shell.tasks.owner.opened():
        assert 'Stopping' in await shell.tasks_command(['stop', 'aaaa'])
        assert record.user_stopped
        assert not shell.plugins_busy('/plugins list')
        assert 'may now be resumed' in await shell.tasks_command(['resume', 'aaaa'])
        assert not record.user_stopped
        assert 'already finished' in await shell.tasks_command(['background', 'aaaa'])
        shell.editor = LivePrompt(
            history=InMemoryHistory(),
            console=shell.console,
            commands=shell.commands,
            images=shell.images,
            interrupts=shell.interrupts,
            toolbar=lambda: [],
        )
        assert 'Resume requested' in await shell.tasks_command(['resume', 'aaaa'])
        assert record.id in shell.editor.queued_messages[0]
    await shell.forks.close()


async def test_main_user_interrupt_marks_foreground_child_stopped(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    import asyncio
    import json
    import signal

    from pydantic_clai2 import chat
    from pydantic_clai2.config.settings_store import SettingsStore
    from tests.clai2.test_app_edges import inputs

    inputs(monkeypatch, ['parent', '/exit'])

    async def stream(messages: list[ModelMessage], info: AgentInfo) -> AsyncIterator[str | DeltaToolCalls]:
        request = messages[0]
        assert isinstance(request, ModelRequest)
        prompt = next(part.content for part in request.parts if isinstance(part, UserPromptPart))
        if prompt == 'child':
            signal.raise_signal(signal.SIGINT)
            await asyncio.Event().wait()
        elif len(messages) == 1:
            yield {0: DeltaToolCall(name='delegate_task', json_args='{"agent_name":"self","task":"child"}')}
        else:
            yield 'parent'  # pragma: lax no cover

    await chat(
        create_stock_agent(FunctionModel(stream_function=stream)),
        deps=None,
        plugins=[Coder(repo_context=False)],
        builtin_plugins=(),
        store=SettingsStore(tmp_path / 'config.db'),
        console=Console(file=io.StringIO()),
    )
    files = list((tmp_path / 'sessions.db.tasks').glob('*.json'))
    assert len(files) == 1
    record = json.loads(files[0].read_text())
    assert record['user_stopped'] is True
    assert record['outcome'] == 'cancelled'


async def test_managed_code_mode_delegation_has_one_typed_row(tmp_path: Path) -> None:
    pytest.importorskip('pydantic_monty')
    from pydantic_clai2.runtime.sandbox_calls import DelegationCallStartedEvent, SandboxCallOrder
    from pydantic_clai2.runtime.speculation import SpeculationCounters
    from pydantic_clai2.runtime.speculative_mode import speculative_capabilities
    from pydantic_clai2.ui.rendering._rendering import StreamRenderer
    from tests.clai2.test_speculative_mode import streamed

    def respond(messages: list[ModelMessage], info: AgentInfo) -> ModelResponse:
        request = messages[0]
        assert isinstance(request, ModelRequest)
        prompt = next(part.content for part in request.parts if isinstance(part, UserPromptPart))
        if prompt == 'parent' and len(messages) == 1:
            return ModelResponse(
                parts=[ToolCallPart('run_code', {'code': 'await delegate_task(agent_name="self", task="child")'})]
            )
        return ModelResponse(parts=[TextPart('evidence')])

    output = io.StringIO()
    console = Console(file=output)
    ui = Tasks(console=console, conversation_id=lambda: 'root', directory=None)
    renderer = StreamRenderer(console, stop_loading=lambda: None, renderers=[task_row])
    ui.sink = renderer.on_stream_event
    seen: list[AgentStreamEvent] = []

    async def receive(event: AgentStreamEvent) -> None:
        seen.append(event)
        await renderer.on_stream_event(event)

    coder = Coder[None](repo_context=False)
    session = Session(
        create_stock_agent(streamed(respond)),
        deps=None,
        workspace=tmp_path,
        plugins=[coder, *speculative_capabilities(SpeculationCounters(), (coder,))],
        on_stream_event=receive,
    )
    ui.conversation_id = lambda: session.summary.id
    session.delegations = ui.owner
    async with ui.owner.opened():
        assert (await session.prompt('parent')).output == 'evidence'
    await renderer.finish()
    starts = [event for event in seen if isinstance(event, DelegationCallStartedEvent)]
    assert len(starts) == 1
    assert SandboxCallOrder().tool_events(starts[0]) == []
    assert '● delegate_task' not in output.getvalue()
    assert output.getvalue().count('general-purpose [') == 1


@pytest.mark.parametrize('timing', ['idle', 'active', 'restored'])
async def test_background_completion_continues_parent_without_user_input(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, timing: str
) -> None:
    import asyncio

    import anyio
    from prompt_toolkit.application import create_app_session
    from prompt_toolkit.input import create_pipe_input
    from prompt_toolkit.output import DummyOutput

    from pydantic_ai.messages import SystemPromptPart
    from pydantic_ai_harness.subagents import DelegationTaskEvent
    from pydantic_clai2._app import create_shell
    from pydantic_clai2.config.project_settings import ProjectSettings
    from pydantic_clai2.config.settings_store import SettingsStore
    from pydantic_clai2.ui.prompt.live_prompt import LivePrompt

    child_started, release, idle, continued, finished = (asyncio.Event() for _ in range(5))
    captured: list[list[ModelMessage]] = []
    output = io.StringIO()
    reads = 0
    original_read = LivePrompt.read

    async def read(live: LivePrompt) -> str:
        nonlocal reads
        reads += 1
        if reads == 2:
            idle.set()
        return await original_read(live)

    monkeypatch.setattr(LivePrompt, 'read', read)

    async def stream(messages: list[ModelMessage], info: AgentInfo) -> AsyncIterator[str | DeltaToolCalls]:
        request = messages[0]
        assert isinstance(request, ModelRequest)
        first = next(part.content for part in request.parts if isinstance(part, UserPromptPart))
        if first == 'child':
            child_started.set()
            await release.wait()
            yield 'Admiral Fluff commands the cheese fleet.'
        elif any(
            isinstance(part, (SystemPromptPart, UserPromptPart))
            and isinstance(part.content, str)
            and 'Automated subagent task report' in part.content
            for message in messages
            if isinstance(message, ModelRequest)
            for part in message.parts
        ):
            captured.append(messages)
            yield 'Pirate hamster reporting for duty.'
            continued.set()
        elif len(messages) == 1:
            yield {
                0: DeltaToolCall(
                    name='delegate_task', json_args='{"agent_name":"self","task":"child","background":true}'
                )
            }
        else:
            if timing == 'active':
                await release.wait()
            yield 'Background work launched.'

    shell = create_shell(
        create_stock_agent(FunctionModel(stream_function=stream)),
        deps=None,
        plugins=[Coder(repo_context=False)],
        usage_limits=None,
        console=Console(file=output, force_terminal=True, width=100, height=24),
        settings=None,
        store=SettingsStore(tmp_path / 'config.db'),
        builtin_plugins=(),
        project=ProjectSettings(),
        headless=True,
    )

    original_observer = shell.tasks.owner.observer

    async def observe(progress: DelegationTaskEvent) -> None:
        assert original_observer is not None
        await original_observer(progress)
        if progress.task.status == 'finished':
            finished.set()

    shell.tasks.owner.observer = observe
    if timing == 'restored':
        with create_pipe_input() as pipe, create_app_session(input=pipe, output=DummyOutput()), anyio.fail_after(10):
            async with anyio.create_task_group() as group:
                group.start_soon(shell.run)
                pipe.send_text('launch\n')
                await idle.wait()
                shell.tasks.wake = None
                release.set()
                await finished.wait()
                assert shell.editor is not None
                shell.editor.submit('/exit')
        assert not captured
        shell = create_shell(
            create_stock_agent(FunctionModel(stream_function=stream)),
            deps=None,
            plugins=[Coder(repo_context=False)],
            usage_limits=None,
            console=Console(file=output, force_terminal=True, width=100, height=24),
            settings=None,
            store=SettingsStore(tmp_path / 'config.db'),
            builtin_plugins=(),
            project=ProjectSettings(),
            headless=True,
            summary=shell.session.summary,
            message_history=shell.session.messages,
        )
    with create_pipe_input() as pipe, create_app_session(input=pipe, output=DummyOutput()), anyio.fail_after(10):
        async with anyio.create_task_group() as group:
            group.start_soon(shell.run)
            if timing != 'restored':
                pipe.send_text('launch\n')
                await child_started.wait()
                if timing == 'idle':
                    await idle.wait()
                assert shell.editor is not None
                shell.editor.buffer.replace('keep this unfinished draft')
                release.set()
            await continued.wait()
            assert shell.editor is not None
            assert shell.editor.buffer.text == ('' if timing == 'restored' else 'keep this unfinished draft')
            shell.editor.submit('/exit')
    assert len(captured) == 1
    prompts = [
        part.content
        for message in shell.session.messages
        if isinstance(message, ModelRequest)
        for part in message.parts
        if isinstance(part, UserPromptPart)
    ]
    assert prompts == ['launch']
    (record,) = shell.tasks.owner.records.values()
    assert record.delivered
    last = shell.session.messages[-1]
    assert isinstance(last, ModelResponse)
    assert any(
        isinstance(part, TextPart) and part.content == 'Pirate hamster reporting for duty.' for part in last.parts
    )
    assert shell.tasks.wake is None


async def test_consumed_background_report_does_not_start_another_turn(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    from prompt_toolkit.application import create_app_session
    from prompt_toolkit.input import create_pipe_input
    from prompt_toolkit.output import DummyOutput

    from pydantic_clai2.ui.prompt.live_prompt import LivePrompt, PromptWakeup
    from tests.clai2.test_forks import Model as ForkModel, shell_for

    shell = shell_for(tmp_path, ForkModel(), io.StringIO())
    shell.console = Console(file=io.StringIO(), force_terminal=True, width=100, height=24)
    reads = 0

    async def read(live: LivePrompt) -> str:
        nonlocal reads
        reads += 1
        if reads == 1:
            raise PromptWakeup
        raise EOFError

    monkeypatch.setattr(LivePrompt, 'read', read)
    with create_pipe_input() as pipe, create_app_session(input=pipe, output=DummyOutput()):
        assert await shell.run() == 'eof'
    assert reads == 2
    assert shell.session.messages == []


async def test_live_view_streams_and_steers_a_running_task() -> None:
    ui = Tasks(console=Console(file=io.StringIO()), conversation_id=lambda: 'root', directory=None)
    started, release = asyncio.Event(), asyncio.Event()
    seen: list[str] = []

    async def child_stream(messages: list[ModelMessage], info: AgentInfo) -> AsyncIterator[str]:
        request = messages[-1]
        assert isinstance(request, ModelRequest)
        seen.append(' '.join(str(part.content) for part in request.parts if isinstance(part, UserPromptPart)))
        if not started.is_set():
            started.set()
            await release.wait()
        yield 'child done'

    async def run(record: DelegationTask) -> str:
        child = Agent(FunctionModel(stream_function=child_stream))
        capability = DelegationReports(ui.owner, conversation_id='root', task_id=record.id)
        return (await child.run(record.prompt, capabilities=[capability])).output

    async with ui.owner.opened():
        await ui.owner.delegate(
            agent_name='self',
            prompt='inspect',
            conversation_id='root',
            model=None,
            background=True,
            resume=None,
            run=run,
        )
        await started.wait()
        (record,) = ui.owner.records.values()
        await ui.observe(DelegationTaskEvent(task=record, event=PartStartEvent(index=0, part=TextPart('partial'))))
        stream = ui.agents.get(f'task-{record.id}')
        assert stream is not None
        assert (stream.title, stream.activity()) == (f'general-purpose {record.id[:8]}', 'responding')
        assert isinstance(stream.entries[-1], PartStartEvent)
        assert ui.send(record.id, 'look here', 'asap') == f'Steering sent to task {record.id[:8]}.'
        assert ui.send(record.id, 'then wrap up', 'when_idle') == f'Queued for task {record.id[:8]}.'
        release.set()
        while record.status != 'finished':
            await asyncio.sleep(0)
        assert ui.send(record.id, 'late', 'asap').startswith(f'Task {record.id[:8]} is not running.')
        assert stream.activity() == 'ok'
        assert stream.entries[-2:] == [
            SentPrompt(text='look here', label='steer'),
            SentPrompt(text='then wrap up', label='queued'),
        ]
    assert seen == ['inspect', 'look here then wrap up']
    elsewhere = DelegationTask(id='c' * 32, agent_name='worker', prompt='p', conversation_id='other')
    await ui.observe(DelegationTaskEvent(task=elsewhere, event=PartStartEvent(index=0, part=TextPart('x'))))
    assert ui.agents.get(f'task-{elsewhere.id}') is None


async def test_live_view_lists_saved_tasks_and_stops_or_backgrounds_them(tmp_path: Path) -> None:
    from tests.clai2.test_forks import Model, shell_for

    shell = shell_for(tmp_path, Model(), io.StringIO())
    assert 'needs an interactive terminal' in await shell.tasks_command([])
    saved = task()
    saved.conversation_id = shell.session.summary.id
    saved.status, saved.outcome = 'finished', 'ok'
    saved.messages = [
        ModelRequest(parts=[UserPromptPart('inspect')]),
        ModelResponse(parts=[TextPart('looked around'), ToolCallPart('grep', {'pattern': 'x'}, tool_call_id='c')]),
    ]
    elsewhere = task(task_id='b' * 32)
    elsewhere.conversation_id = 'other'
    shell.tasks.owner.records = {saved.id: saved, elsewhere.id: elsewhere}
    shell.tasks.restore()
    shell.tasks.restore()
    stream = shell.agents.get(f'task-{saved.id}')
    assert stream is not None and shell.agents.get(f'task-{elsewhere.id}') is None
    # The saved history already starts with the prompt, so it shows once.
    assert [entry for entry in stream.entries if isinstance(entry, SentPrompt)] == [SentPrompt(text='inspect')]
    assert stream.activity() == 'ok'
    assert stream.stop is not None and await stream.stop() == 'Task aaaaaaaa already finished.'
    empty = task(task_id='c' * 32)
    empty.conversation_id = shell.session.summary.id
    shell.tasks.owner.records[empty.id] = empty
    shell.tasks.restore()
    blank = shell.agents.get(f'task-{empty.id}')
    assert blank is not None and blank.entries == [SentPrompt(text='inspect')]
    assert any(isinstance(entry, PartStartEvent) for entry in stream.entries)
    assert stream.background is not None and stream.background() == 'Task aaaaaaaa already finished.'
    saved.status, saved.backgroundable = 'running', False
    assert 'workspace' in stream.background()
    async with shell.tasks.owner.opened():
        assert await stream.stop() == 'Stopping task aaaaaaaa and its descendants.'
    assert saved.user_stopped
    await shell.forks.close()
