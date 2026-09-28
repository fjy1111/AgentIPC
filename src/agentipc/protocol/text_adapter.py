from __future__ import annotations

import base64
import json
from typing import TYPE_CHECKING

import numpy as np
from pydantic import BaseModel

from agentipc.protocol.envelope import AgentEnvelope

if TYPE_CHECKING:
    from agentipc.runtime.reference_resolver import ReferenceResolver


_REFERENCE_ERROR = "reference materialization is not supported without a resolver"


def _json_text(value: object) -> str:
    return json.dumps(
        value,
        ensure_ascii=False,
        sort_keys=True,
        separators=(",", ":"),
    )


def _text_safe(value: object) -> object:
    if isinstance(value, np.ndarray):
        if np.iscomplexobj(value):
            return {
                "dtype": value.dtype.str,
                "shape": list(value.shape),
                "real": value.real.tolist(),
                "imag": value.imag.tolist(),
            }
        return {
            "dtype": value.dtype.str,
            "shape": list(value.shape),
            "values": value.tolist(),
        }

    if isinstance(value, bytes):
        try:
            text = value.decode("utf-8")
        except UnicodeDecodeError:
            return {
                "encoding": "base64",
                "data": base64.b64encode(value).decode("ascii"),
            }
        return {
            "encoding": "utf-8",
            "text": text,
        }

    if isinstance(value, BaseModel):
        return value.model_dump(mode="json")

    return value


def _materialize_refs(
    refs: list[object],
    *,
    resolve,
) -> list[dict[str, object]]:
    materialized: list[dict[str, object]] = []
    for ref in refs:
        resolved = resolve(ref)
        materialized.append(
            {
                "ref": ref.model_dump(mode="json"),
                "materialized": _text_safe(resolved),
            }
        )
    return materialized


def render(
    envelope: AgentEnvelope,
    resolver: ReferenceResolver | None = None,
) -> str:
    """Render an AgentEnvelope as stable human-readable text."""
    has_refs = bool(
        envelope.state_refs or envelope.artifact_refs or envelope.memory_refs
    )

    resolve = None
    if has_refs:
        if resolver is None:
            raise ValueError(_REFERENCE_ERROR)
        resolve = getattr(resolver, "resolve", None)
        if not callable(resolve):
            raise TypeError("resolver must provide a callable resolve() method")

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

    if envelope.state_refs:
        lines.append(
            "State references: "
            + _json_text(
                _materialize_refs(
                    list(envelope.state_refs),
                    resolve=resolve,
                )
            )
        )
    if envelope.artifact_refs:
        lines.append(
            "Artifact references: "
            + _json_text(
                _materialize_refs(
                    list(envelope.artifact_refs),
                    resolve=resolve,
                )
            )
        )
    if envelope.memory_refs:
        lines.append(
            "Memory references: "
            + _json_text(
                _materialize_refs(
                    list(envelope.memory_refs),
                    resolve=resolve,
                )
            )
        )

    return "\n".join(lines)
