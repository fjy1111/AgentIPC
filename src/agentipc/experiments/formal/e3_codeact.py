from __future__ import annotations
from agentipc.evaluation.experiment import EXPERIMENT_D
from agentipc.evaluation.runner import run_single
from agentipc.experiments.formal.aggregation import flatten_record
from agentipc.scenarios.models import load_codeact_tasks

def run_e3(*, factory, root, repeat):
    tasks=load_codeact_tasks(root/'scenarios/codeact_chain/tasks.json'); task=tasks[0].description
    rows=[]
    for i in range(repeat):
        ctx, agents=factory(EXPERIMENT_D,0,i,i,task)
        rows.append(flatten_record(run_single(experiment=EXPERIMENT_D,task=task,ctx=ctx,agent_registry=agents),i))
    def summarize(items):
        n = len(items)
        return {
            "tool_call_count": sum(float(r.get("tool_call_count", 0)) for r in items) / n,
            "repeated_tool_call_count": sum(float(r.get("repeated_tool_call_count", 0)) for r in items) / n,
            "memory_used": sum(float(r.get("memory_used", 0)) for r in items) / n,
            "memory_effective": sum(float(r.get("memory_effective", 0)) for r in items) / n,
            "mean_latency_ms": sum(float(r.get("latency_ms", 0)) for r in items) / n,
            "mean_tokens": sum(float(r.get("llm_total_tokens", 0)) for r in items) / n,
        }
    baseline = rows[:1]
    reuse = rows[1:] or rows[:1]
    return rows, {"baseline": summarize(baseline), "memory_reuse": summarize(reuse)}
