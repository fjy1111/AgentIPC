# AgentIPC Real Bailian Calibration Report

Overall calibration: **PASS**

## Metric semantics

`runtime_memory_effective_strict` / `runtime_memory_harmful_strict` are the existing Runtime strict metrics based on exact historical-vs-final answer string equality.
`validated_memory_effective` / `validated_memory_harmful` are experiment-layer ground-truth validated metrics based on deterministic task evaluators. The two definitions are intentionally kept separate; the validated metric never overwrites the Runtime metric.

`text_tokens` is a communication-side token estimate from AgentIPC TextCounter. It is not Bailian billing usage. Real API usage is reported only by `llm_prompt_tokens`, `llm_completion_tokens`, and `llm_total_tokens`.

## Calibration checks

| Check | Result | Key evidence |
|---|---|---|
| C1 LLM provider | PASS | prompt=33, completion=312, latency_ms=5981.239171997004 |
| C1 embedding | PASS | shape=[2, 1024], dtype=float32, finite=True |
| C2 POSIX SHM | PASS | transport=shm, shape=[64], nbytes=256, released=True |
| C3 Knowledge R2/R8 | PASS | rounds=2 |
| C4 CodeAct R2/R8 | PASS | rounds=2 |
| Secret audit | PASS | files_scanned=5 |

## Provider usage

| Metric | Value |
|---|---:|
| LLM calls | 9 |
| LLM prompt tokens | 2236 |
| LLM completion tokens | 6088 |
| LLM total tokens | 8324 |
| Missing LLM usage records | 0 |
| Embedding calls | 11 |
| Embedding input items | 20 |

## Knowledge memory semantics

| Round | Eval | memory_used | strict effective/harmful | validated effective/harmful | tool calls | discrepancy |
|---:|---|---:|---|---|---:|---|
| 2 | PASS | 0 | 0/0 | 0/0 | 0 | False |
| 8 | PASS | 1 | 1/0 | 1/0 | 0 | False |

## CodeAct memory semantics

| Round | Eval | memory_used | strict effective/harmful | validated effective/harmful | tool calls | discrepancy |
|---:|---|---:|---|---|---:|---|
| 2 | PASS | 0 | 0/0 | 0/0 | 1 | False |
| 8 | PASS | 1 | 1/0 | 1/0 | 0 | False |

The calibration is a small-cost gate only. It does not run the full formal R002 experiment matrix.
