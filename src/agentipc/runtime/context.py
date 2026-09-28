from __future__ import annotations

from dataclasses import dataclass
from enum import Enum
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from agentipc.artifacts.store import ArtifactStore
    from agentipc.config import AgentIPCConfig
    from agentipc.evaluation.metrics import MetricsCollector
    from agentipc.evaluation.trace import TraceLogger
    from agentipc.memory.service import MemoryService
    from agentipc.protocol.registry import CapabilityRegistry
    from agentipc.providers.factory import ProviderBundle
    from agentipc.state.hub import StateHub


class RunMode(str, Enum):
    TEXT = "text"
    STRUCTURED = "structured"


@dataclass(slots=True)
class RunContext:
    trace_id: str
    task_id: str
    mode: RunMode

    config: AgentIPCConfig
    registry: CapabilityRegistry

    state_hub: StateHub
    artifact_store: ArtifactStore
    memory_service: MemoryService

    metrics: MetricsCollector
    trace_logger: TraceLogger

    provider_bundle: ProviderBundle

    use_state: bool = False
    use_memory: bool = False
    use_sandbox: bool = False
