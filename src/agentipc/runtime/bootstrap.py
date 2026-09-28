from agentipc.evaluation.trace import TraceLogger
from agentipc.protocol.builders import build_ack, build_hello, build_register
from agentipc.protocol.capability import AgentCapability
from agentipc.protocol.envelope import AgentEnvelope
from agentipc.protocol.registry import CapabilityRegistry
from agentipc.runtime.agent_registry import AgentRegistry


_REQUIRED_AGENT_IDS = (
    "planner",
    "retriever",
    "executor",
    "summarizer",
)

_RUNTIME_ID = "runtime"


def bootstrap_capabilities(
    *,
    agent_registry: AgentRegistry,
    capability_registry: CapabilityRegistry,
) -> None:
    if not isinstance(agent_registry, AgentRegistry):
        raise TypeError("agent_registry must be an AgentRegistry")
    if not isinstance(capability_registry, CapabilityRegistry):
        raise TypeError("capability_registry must be a CapabilityRegistry")

    agents = [
        agent_registry.get(agent_id)
        for agent_id in _REQUIRED_AGENT_IDS
    ]

    for agent in agents:
        capability_registry.register(
            AgentCapability(
                agent_id=agent.agent_id,
                capabilities=list(agent.capabilities),
            )
        )


def run_handshake(
    *,
    agent_registry: AgentRegistry,
    capability_registry: CapabilityRegistry,
    trace_logger: TraceLogger,
    trace_id: str,
    task_id: str,
) -> list[AgentEnvelope]:
    if not isinstance(agent_registry, AgentRegistry):
        raise TypeError("agent_registry must be an AgentRegistry")
    if not isinstance(capability_registry, CapabilityRegistry):
        raise TypeError("capability_registry must be a CapabilityRegistry")
    if not isinstance(trace_logger, TraceLogger):
        raise TypeError("trace_logger must be a TraceLogger")
    if type(trace_id) is not str:
        raise TypeError("trace_id must be a str")
    if trace_id == "":
        raise ValueError("trace_id must be a non-empty str")
    if type(task_id) is not str:
        raise TypeError("task_id must be a str")
    if task_id == "":
        raise ValueError("task_id must be a non-empty str")

    bootstrap_capabilities(
        agent_registry=agent_registry,
        capability_registry=capability_registry,
    )

    messages: list[AgentEnvelope] = []
    for agent_id in _REQUIRED_AGENT_IDS:
        capability = capability_registry.get(agent_id)
        if capability is None:
            raise RuntimeError(
                f"capability bootstrap did not register agent: {agent_id!r}"
            )

        hello = build_hello(
            sender=agent_id,
            receiver=_RUNTIME_ID,
            trace_id=trace_id,
            task_id=task_id,
            step_id=f"handshake-{agent_id}-hello",
        )
        trace_logger.log_envelope(hello)
        messages.append(hello)

        register = build_register(
            sender=agent_id,
            receiver=_RUNTIME_ID,
            trace_id=trace_id,
            task_id=task_id,
            step_id=f"handshake-{agent_id}-register",
            capability=capability,
        )
        trace_logger.log_envelope(register)
        messages.append(register)

        ack = build_ack(
            sender=_RUNTIME_ID,
            receiver=agent_id,
            trace_id=trace_id,
            task_id=task_id,
            step_id=f"handshake-{agent_id}-ack",
            ack_message_id=register.message_id,
        )
        trace_logger.log_envelope(ack)
        messages.append(ack)

    return messages
