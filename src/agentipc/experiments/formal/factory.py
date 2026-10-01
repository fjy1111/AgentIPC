from __future__ import annotations
from pathlib import Path
import os
from agentipc.agents.planner import PlannerAgent
from agentipc.agents.retriever import RetrieverAgent
from agentipc.agents.summarizer import SummarizerAgent
from agentipc.artifacts.store import ArtifactStore
from agentipc.config import AgentIPCConfig
from agentipc.evaluation.metrics import MetricsCollector
from agentipc.evaluation.trace import TraceLogger
from agentipc.memory.service import MemoryService
from agentipc.memory.sqlite_store import SQLiteMemoryStore
from agentipc.memory.vector_index import VectorIndex
from agentipc.protocol.registry import CapabilityRegistry
from agentipc.providers.factory import create_provider_bundle
from agentipc.runtime.agent_registry import AgentRegistry
from agentipc.runtime.context import RunContext, RunMode
from agentipc.state.hub import StateHub

def build_factory(*, root: str | Path, provider: str = "openai"):
    root=Path(root); cfg=AgentIPCConfig(llm_provider=provider, embedding_provider=("hash" if provider=="mock" else provider), random_seed=42)
    tasks=["Diagnose a DNS resolution failure and summarize the remediation steps."]
    def factory(experiment, task_index, run_index, seed, task):
        run_root=root/f"run-{experiment.name.value}-{task_index}-{run_index}"; run_root.mkdir(parents=True,exist_ok=True)
        local_cfg=cfg.model_copy(update={"random_seed":seed, "state_root":run_root/"state", "memory_root":run_root/"memory", "artifact_root":run_root/"artifacts"})
        bundle=create_provider_bundle(local_cfg, llm_options={} if provider=="mock" else {"model":"qwen-plus","api_key":os.environ.get("DASHSCOPE_API_KEY",""),"base_url":os.environ.get("AGENTIPC_BAILIAN_BASE_URL","")}, embedding_options={} if provider=="mock" else {"model":"text-embedding-v3","dim":1024,"api_key":os.environ.get("DASHSCOPE_API_KEY",""),"base_url":os.environ.get("AGENTIPC_BAILIAN_BASE_URL","")})
        store=SQLiteMemoryStore(local_cfg.memory_root); memory=MemoryService(store,bundle.embedding,VectorIndex(bundle.embedding.dim))
        hub=StateHub(transport="shm"); agents=AgentRegistry(); agents.register(PlannerAgent()); agents.register(RetrieverAgent(knowledge=[])); agents.register(SummarizerAgent())
        from agentipc.agents.executor import ExecutorAgent
        agents.register(ExecutorAgent())
        ctx=RunContext(trace_id=f"formal-{experiment.name.value}-{run_index}",task_id=f"formal-task-{task_index}-{run_index}",mode=experiment.mode,config=local_cfg,registry=CapabilityRegistry(),state_hub=hub,artifact_store=ArtifactStore(local_cfg.artifact_root),memory_service=memory,metrics=MetricsCollector(),trace_logger=TraceLogger(run_root/"trace.jsonl"),provider_bundle=bundle,use_state=experiment.use_state,use_memory=experiment.use_memory,use_sandbox=False)
        return ctx,agents
    return tasks,factory
