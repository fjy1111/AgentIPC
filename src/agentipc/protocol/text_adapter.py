import json

from agentipc.protocol.envelope import AgentEnvelope


_REFERENCE_ERROR = (
    "reference materialization is not supported by the minimal TextAdapter"
)


def _json_text(value: object) -> str:
    return json.dumps(
        value,
        ensure_ascii=False,
        sort_keys=True,
        separators=(",", ":"),
    )


def render(
    envelope: AgentEnvelope,
    resolver: object | None = None,
) -> str:
    """Render a reference-free AgentEnvelope as stable human-readable text."""
    del resolver

    if envelope.state_refs or envelope.artifact_refs or envelope.memory_refs:
        raise ValueError(_REFERENCE_ERROR)

    data = envelope.model_dump(mode="json")
    action = envelope.action.value if envelope.action is not None else "null"
    capability = envelope.capability if envelope.capability is not None else "null"

    lines = [
        f"Protocol version: {envelope.version}",
        f"Message ID: {envelope.message_id}",
        f"Trace ID: {envelope.trace_id}",
        f"Task ID: {envelope.task_id}",
        f"Step ID: {envelope.step_id}",
        f"From: {envelope.sender}",
        f"To: {envelope.receiver}",
        f"Message type: {envelope.message_type.value}",
        f"Action: {action}",
        f"Capability: {capability}",
        f"Arguments: {_json_text(data['args'])}",
        f"Result: {_json_text(data['result'])}",
        f"Status: {envelope.status.value}",
        f"Created at: {envelope.created_at}",
        f"Metrics: {_json_text(data['metrics'])}",
    ]
    return "\n".join(lines)