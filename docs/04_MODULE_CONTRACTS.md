# 04 — Module Contracts

本文件定义 MVP 公共契约。实现者不得擅自修改字段含义或公共方法签名。

## 1. Protocol

### 1.1 ProtocolVersion

MVP 固定：

```text
agentipc/0.1
```

### 1.2 MessageType

```text
HELLO
REGISTER
DISCOVER
REQUEST
RESULT
ACK
ERROR
```

### 1.3 ActionType

```text
PLAN
RETRIEVE
EXECUTE
SUMMARIZE
MEMORY_QUERY
MEMORY_WRITE
```

### 1.4 MessageStatus

```text
PENDING
OK
ERROR
TIMEOUT
SKIPPED
```

### 1.5 AgentEnvelope

公共字段：

```python
class AgentEnvelope(BaseModel):
    version: str
    message_id: str
    trace_id: str
    task_id: str
    step_id: str

    sender: str
    receiver: str
    message_type: MessageType
    action: ActionType | None
    capability: str | None

    args: dict[str, Any]
    result: dict[str, Any] | None

    state_refs: list[StateRef]
    artifact_refs: list[ArtifactRef]
    memory_refs: list[MemoryRef]

    status: MessageStatus
    created_at: float
    metrics: dict[str, Any]
```

要求：

- `args/result` 只承载小型结构化内容。
- 大对象必须使用 refs。
- `message_id` 唯一。
- `trace_id/task_id` 在一次任务内稳定。

### 1.6 ProtocolCodec

```python
encode(envelope: AgentEnvelope) -> bytes

decode(payload: bytes) -> AgentEnvelope
```

MVP 默认 JSON UTF-8。是否增加 msgpack 属于后续增强，不在 MVP 核心。

### 1.7 TextAdapter

```python
render(envelope: AgentEnvelope, resolver: ReferenceResolver) -> str
```

用于 text baseline。必须展开与 structured 路径语义等价的信息。

## 2. Capability Registry

### AgentCapability

```python
class AgentCapability(BaseModel):
    agent_id: str
    capabilities: list[str]
    protocol_versions: list[str]
    metadata: dict[str, Any]
```

### Registry API

```python
register(capability: AgentCapability) -> None
get(agent_id: str) -> AgentCapability | None
discover(required: str) -> list[AgentCapability]
supports(agent_id: str, capability: str) -> bool
```

## 3. State

### StateRef

```python
class StateRef(BaseModel):
    uri: str
    kind: str
    shape: list[int]
    dtype: str
    nbytes: int
    checksum: str
    transport: str
    summary: str
```

URI 示例：

```text
shm://agentipc/<name>
inproc://agentipc/<id>
```

### StateHub API

```python
put_array(array: np.ndarray, *, kind: str, summary: str) -> StateRef
resolve_array(ref: StateRef) -> np.ndarray
exists(ref: StateRef) -> bool
release(ref: StateRef) -> None
close() -> None
```

要求：

- `resolve_array` 返回的数据必须可用于真实计算。
- checksum 必须可以验证数据未损坏。
- shared memory 不可用时允许降级 `inproc`，但 metrics 必须记录 transport。

## 4. Artifact

### ArtifactRef

```python
class ArtifactRef(BaseModel):
    uri: str
    sha256: str
    media_type: str
    size_bytes: int
    summary: str
```

URI 示例：

```text
artifact://sha256/<digest>
```

### ArtifactStore API

```python
put_bytes(data: bytes, *, media_type: str, summary: str) -> ArtifactRef
put_json(value: Any, *, summary: str) -> ArtifactRef
get_bytes(ref: ArtifactRef) -> bytes
get_json(ref: ArtifactRef) -> Any
exists(ref: ArtifactRef) -> bool
```

## 5. Memory

### MemoryType

```text
evidence
experience
result
```

### MemoryRecord

```python
class MemoryRecord(BaseModel):
    memory_id: str
    source_agent: str
    created_at: float
    task_topic: str
    summary: str

    memory_type: MemoryType
    tags: list[str]
    keywords: list[str]
    embedding: list[float] | None
    payload: dict[str, Any]

    reuse_count: int
    success_count: int
    failure_count: int
    last_accessed_at: float | None
```

### MemoryRef

```python
class MemoryRef(BaseModel):
    memory_id: str
    score: float
    match_type: str
    summary: str
```

### MemoryService API

```python
write(record: MemoryRecord) -> MemoryRecord
get(memory_id: str) -> MemoryRecord | None
retrieve(
    query: str,
    *,
    tags: list[str] | None = None,
    keywords: list[str] | None = None,
    top_k: int = 5,
) -> list[MemoryRef]
mark_used(memory_id: str, *, effective: bool | None = None) -> None
```

MVP hybrid score 建议固定：

```text
0.60 * semantic + 0.25 * keyword + 0.15 * tag
```

若后续调整，必须记录实验配置。

## 6. Providers

### LLMProvider

```python
class LLMProvider(Protocol):
    def complete(
        self,
        messages: list[dict[str, str]],
        *,
        temperature: float = 0.0,
    ) -> LLMResponse: ...
```

### LLMResponse

```python
class LLMResponse(BaseModel):
    text: str
    prompt_tokens: int | None
    completion_tokens: int | None
    latency_ms: float
    raw: dict[str, Any] | None
```

### EmbeddingProvider

```python
class EmbeddingProvider(Protocol):
    @property
    def dim(self) -> int: ...

    def embed(self, texts: list[str]) -> np.ndarray: ...
```

返回 shape 必须是 `(n, dim)`，dtype 推荐 `float32`。

## 7. Agent

### BaseAgent

```python
class BaseAgent(ABC):
    agent_id: str
    capabilities: list[str]

    @abstractmethod
    def handle(self, envelope: AgentEnvelope, ctx: RunContext) -> AgentEnvelope:
        ...
```

四个 Agent 的 `handle` 不得直接访问其他 Agent 对象，只通过 Runtime / Context 提供的基础设施。

## 8. Runtime

### RunMode

```text
text
structured
```

其他功能通过 feature flags 表示：

```python
use_state: bool
use_memory: bool
use_sandbox: bool
```

### RunContext

必须至少提供：

```python
trace_id
task_id
mode
config
registry
state_hub
artifact_store
memory_service
metrics
trace_logger
provider_bundle
```

### RunResult

```python
class RunResult(BaseModel):
    task_id: str
    success: bool
    answer: str
    error: dict[str, Any] | None
    metrics: dict[str, Any]
    trace_path: str | None
```

## 9. Sandbox

### SandboxResult

```python
class SandboxResult(BaseModel):
    exit_code: int
    stdout: str
    stderr: str
    timed_out: bool
    duration_ms: float
```

### PythonSandbox

```python
run(code: str, *, timeout_sec: float) -> SandboxResult
```

MVP 目标是“受限执行”，不是安全边界证明。必须在文档中明确局限。

## 10. Metrics

核心字段固定：

```text
message_count
text_chars
text_tokens
protocol_bytes
state_transfer_count
state_bytes
artifact_ref_count
memory_retrieved
memory_used
memory_effective
memory_harmful
tool_call_count
repeated_tool_call_count
latency_ms
success
```

正式报告不得随意改名。
