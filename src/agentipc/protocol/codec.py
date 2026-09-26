import json

from agentipc.protocol.envelope import AgentEnvelope


def encode(envelope: AgentEnvelope) -> bytes:
    """Encode an AgentEnvelope as deterministic, compact UTF-8 JSON bytes."""
    data = envelope.model_dump(mode="json")
    text = json.dumps(
        data,
        ensure_ascii=False,
        sort_keys=True,
        separators=(",", ":"),
    )
    return text.encode("utf-8")


def decode(payload: bytes) -> AgentEnvelope:
    """Decode UTF-8 JSON bytes and validate them as an AgentEnvelope."""
    data = json.loads(payload.decode("utf-8"))
    return AgentEnvelope.model_validate(data)