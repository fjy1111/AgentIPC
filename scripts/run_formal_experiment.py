from __future__ import annotations
import argparse, json, platform, sys
from datetime import datetime, timezone
from pathlib import Path
from agentipc.experiments.formal import render_report, run_e1

def main():
 p=argparse.ArgumentParser(); p.add_argument("experiment",choices=["e1","e2","e3","e4"]); p.add_argument("--repeat",type=int,default=3); p.add_argument("--provider",choices=["openai","mock"],default="openai"); p.add_argument("--input",type=Path); a=p.parse_args()
 root=Path("results/formal")/("e1-abcd-"+datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")); root.mkdir(parents=True,exist_ok=False)
 if a.input:
  from agentipc.experiments.formal.aggregation import aggregate_records
  rows=[json.loads(x) for x in a.input.read_text(encoding="utf-8").splitlines() if x.strip()]; summary=aggregate_records(rows)
 elif a.experiment=="e4":
  from agentipc.experiments.formal.e4_shm import run_e4
  rows=run_e4(repeat=a.repeat); summary={"inproc":{},"shm":{}}
  for r in rows: summary[r["experiment"]]={"mean_latency_ms":sum(x["latency_ms"] for x in rows if x["experiment"]==r["experiment"])/a.repeat,"mean_state_bytes":r["state_bytes"]}
 else:
  from agentipc.experiments.formal.factory import build_factory
  tasks,factory=build_factory(root=root/"work",provider=a.provider); rows,summary=(run_e1(tasks=tasks,repeat=a.repeat,run_factory=factory) if a.experiment=="e1" else (__import__("agentipc.experiments.formal.e2_knowledge",fromlist=["run_e2"]).run_e2(factory=factory,root=Path("."),repeat=a.repeat) if a.experiment=="e2" else __import__("agentipc.experiments.formal.e3_codeact",fromlist=["run_e3"]).run_e3(factory=factory,root=Path("."),repeat=a.repeat)))
 (root/"environment.json").write_text(json.dumps({"python":sys.version,"platform":platform.platform(),"repeat":a.repeat,"provider":a.provider},indent=2),encoding="utf-8")
 (root/"raw.jsonl").write_text("\n".join(json.dumps(x,ensure_ascii=False) for x in rows)+"\n",encoding="utf-8")
 (root/"summary.json").write_text(json.dumps(summary,indent=2),encoding="utf-8")
 (root/"report.md").write_text(render_report(summary),encoding="utf-8"); print(root)
if __name__=="__main__": main()
