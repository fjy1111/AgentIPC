from agentipc.runtime.context import RunContext, RunMode


class PoisonComponent:
    def __getattr__(self, name: str) -> object:
        raise AssertionError(f"RunContext construction accessed component attribute {name!r}")


def _components() -> dict[str, PoisonComponent]:
    return {
        "config": PoisonComponent(),
        "registry": PoisonComponent(),
        "state_hub": PoisonComponent(),
        "artifact_store": PoisonComponent(),
        "memory_service": PoisonComponent(),
        "metrics": PoisonComponent(),
        "trace_logger": PoisonComponent(),
        "provider_bundle": PoisonComponent(),
    }


def test_run_mode_values_are_fixed() -> None:
    assert RunMode.TEXT.value == "text"
    assert RunMode.STRUCTURED.value == "structured"
    assert list(RunMode) == [RunMode.TEXT, RunMode.STRUCTURED]


def test_structured_context_preserves_dependency_identity_and_flags() -> None:
    components = _components()

    ctx = RunContext(
        trace_id="trace-structured",
        task_id="task-structured",
        mode=RunMode.STRUCTURED,
        **components,
        use_state=True,
        use_memory=False,
        use_sandbox=False,
    )

    assert ctx.trace_id == "trace-structured"
    assert ctx.task_id == "task-structured"
    assert ctx.mode is RunMode.STRUCTURED
    for name, component in components.items():
        assert getattr(ctx, name) is component
    assert ctx.use_state is True
    assert ctx.use_memory is False
    assert ctx.use_sandbox is False


def test_text_context_preserves_dependency_identity_and_default_flags() -> None:
    components = _components()

    ctx = RunContext(
        trace_id="trace-text",
        task_id="task-text",
        mode=RunMode.TEXT,
        **components,
    )

    assert ctx.trace_id == "trace-text"
    assert ctx.task_id == "task-text"
    assert ctx.mode is RunMode.TEXT
    for name, component in components.items():
        assert getattr(ctx, name) is component
    assert ctx.use_state is False
    assert ctx.use_memory is False
    assert ctx.use_sandbox is False
