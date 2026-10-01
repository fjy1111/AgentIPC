from agentipc.evaluation.metrics import MetricsCollector
from agentipc.evaluation.text_counter import TextCounter
from agentipc.protocol.envelope import AgentEnvelope
from agentipc.protocol.text_adapter import render
from agentipc.runtime.context import RunContext
from agentipc.runtime.router import Router


class TextTransport:
    def __init__(
        self,
        *,
        text_counter: TextCounter | None = None,
        resolver: object | None = None,
    ) -> None:
        if text_counter is None:
            text_counter = TextCounter()
        elif not isinstance(text_counter, TextCounter):
            raise TypeError("text_counter must be a TextCounter")

        self._text_counter = text_counter
        self._resolver = resolver

    def dispatch(
        self,
        envelope: AgentEnvelope,
        *,
        ctx: RunContext,
        router: Router,
    ) -> AgentEnvelope:
        if not isinstance(envelope, AgentEnvelope):
            raise TypeError("envelope must be an AgentEnvelope")
        if not isinstance(router, Router):
            raise TypeError("router must be a Router")

        self._record_text_envelope(envelope, ctx)
        response = router.dispatch(envelope, ctx)
        if not isinstance(response, AgentEnvelope):
            raise ValueError("stage response must be an AgentEnvelope")
        self._record_text_envelope(response, ctx)
        return response

    def _record_text_envelope(
        self,
        envelope: AgentEnvelope,
        ctx: RunContext,
    ) -> str:
        rendered = render(
            envelope,
            resolver=self._resolver,
        )

        if isinstance(ctx.metrics, MetricsCollector):
            count = self._text_counter.count(rendered)
            payload = rendered.encode("utf-8")
            ctx.metrics.increment("message_count")
            ctx.metrics.increment("text_chars", count.text_chars)
            ctx.metrics.increment("text_tokens", count.text_tokens)
            ctx.metrics.increment("wire_chars", count.text_chars)
            ctx.metrics.increment("wire_tokens", count.text_tokens)
            ctx.metrics.increment("wire_bytes", len(payload))
            if envelope.state_refs:
                ctx.metrics.increment("state_transfer_count", len(envelope.state_refs))
                ctx.metrics.increment(
                    "state_bytes",
                    sum(ref.nbytes for ref in envelope.state_refs),
                )
            if envelope.artifact_refs:
                ctx.metrics.increment("artifact_ref_count", len(envelope.artifact_refs))
                ctx.metrics.increment(
                    "artifact_payload_bytes",
                    sum(ref.size_bytes for ref in envelope.artifact_refs),
                )

        ctx.trace_logger.log_envelope(envelope)
        return rendered
