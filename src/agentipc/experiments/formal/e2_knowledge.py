from __future__ import annotations
from pathlib import Path
from agentipc.config import AgentIPCConfig
from agentipc.experiments.real_bailian.config import load_real_bailian_config
from agentipc.experiments.real_bailian.runner import run_knowledge_mini_calibration, build_recording_provider_bundle

def run_e2(*, root: Path, result_dir: Path, repeat: int):
    secret=load_real_bailian_config(); bundle=build_recording_provider_bundle(secret)
    summary, raw=run_knowledge_mini_calibration(repo_root=root,result_dir=result_dir,config=AgentIPCConfig(llm_provider="openai",embedding_provider="openai",random_seed=42),provider_bundle=bundle,secret_config=secret)
    return raw, summary
