from __future__ import annotations
from agentipc.evaluation.experiment import EXPERIMENT_D
from agentipc.evaluation.runner import run_single
from agentipc.experiments.formal.aggregation import flatten_record, aggregate_records
from agentipc.scenarios.models import load_codeact_tasks

def run_e3(*, factory, root, repeat):
    tasks=load_codeact_tasks(root/'scenarios/codeact_chain/tasks.json'); task=tasks[0].description
    rows=[]
    for i in range(repeat):
        ctx, agents=factory(EXPERIMENT_D,0,i,i,task)
        rows.append(flatten_record(run_single(experiment=EXPERIMENT_D,task=task,ctx=ctx,agent_registry=agents),i))
    return rows, aggregate_records(rows)
