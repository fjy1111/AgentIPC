from __future__ import annotations
from agentipc.evaluation.experiment import EXPERIMENT_D
from agentipc.evaluation.runner import run_single
from agentipc.experiments.formal.aggregation import flatten_record, aggregate_records
from agentipc.scenarios.knowledge_loader import load_knowledge_documents

def run_e2(*, factory, root, repeat):
    documents = load_knowledge_documents(root/'scenarios/knowledge_chain/knowledge')
    if not documents:
        raise ValueError("knowledge corpus is empty")
    document = documents[0]
    task = f"Summarize the operational guidance in {document.title}: {document.body}"
    rows=[]
    for i in range(repeat):
        ctx, agents=factory(EXPERIMENT_D,0,i,i,task)
        rows.append(flatten_record(run_single(experiment=EXPERIMENT_D,task=task,ctx=ctx,agent_registry=agents),i))
    one = rows
    n = len(one)
    mean = lambda key: sum(float(r.get(key, 0)) for r in one) / n
    return rows, {"E2": {"mean_latency_ms": mean("latency_ms"), "mean_tokens": mean("llm_total_tokens"), "mean_protocol_bytes": mean("protocol_bytes"), "mean_state_bytes": mean("state_bytes"), "mean_tool_calls": mean("tool_call_count"), "memory_effective_rate": sum(bool(r.get("memory_effective")) for r in one) / n}}
