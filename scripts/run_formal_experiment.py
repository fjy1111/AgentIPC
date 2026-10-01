from __future__ import annotations
import argparse, json, platform, sys
from datetime import datetime, timezone
from pathlib import Path
from agentipc.experiments.formal import render_report, run_e1

def main():
 p=argparse.ArgumentParser(); p.add_argument("experiment",choices=["e1","e2","e3","e4"]); p.add_argument("--repeat",type=int,default=3); p.add_argument("--provider",choices=["openai","mock"],default="mock"); p.add_argument("--input",type=Path); a=p.parse_args()
 labels={"e1":"e1-abcd","e2":"e2-knowledge","e3":"e3-codeact","e4":"e4-shm"}
 root=Path("results/formal")/(labels[a.experiment]+"-"+datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")); root.mkdir(parents=True,exist_ok=False)
 if a.input:
  from agentipc.experiments.formal.aggregation import aggregate_records
  rows=[json.loads(x) for x in a.input.read_text(encoding="utf-8").splitlines() if x.strip()]; summary=aggregate_records(rows)
 elif a.experiment=="e4":
  from agentipc.experiments.formal.e4_shm import run_e4
  rows=run_e4(repeat=a.repeat); summary={"inproc":{},"shm":{}}
  for r in rows: summary[r["experiment"]]={"mean_latency_ms":sum(x["latency_ms"] for x in rows if x["experiment"]==r["experiment"])/a.repeat,"mean_state_bytes":r["state_bytes"]}
 else:
  from agentipc.experiments.formal.factory import build_factory
  if a.experiment=="e1":
   tasks,factory=build_factory(root=root/"work",provider=a.provider); rows,summary=run_e1(tasks=tasks,repeat=a.repeat,run_factory=factory)
  elif a.experiment=="e2":
   from agentipc.experiments.formal.e2_knowledge import run_e2; rows,summary=run_e2(root=Path("."),result_dir=root,repeat=a.repeat)
  else:
   from agentipc.experiments.formal.e3_codeact import run_e3; rows,summary=run_e3(root=Path("."),result_dir=root,repeat=a.repeat)
 (root/"environment.json").write_text(json.dumps({"python":sys.version,"platform":platform.platform(),"repeat":a.repeat,"provider":a.provider},indent=2),encoding="utf-8")
 (root/"raw.jsonl").write_text("\n".join(json.dumps(x,ensure_ascii=False) for x in rows)+"\n",encoding="utf-8")
 (root/"summary.json").write_text(json.dumps(summary,indent=2),encoding="utf-8")
 (root/"report.md").write_text(render_report(summary),encoding="utf-8"); print(root)
if __name__=="__main__": main()
