## vLLM v0 架构下 best_of 的离线推理支持

本文聚焦 vLLM v0 架构在“离线推理（offline inference）”场景中对 best_of 的支持机制，给出完整数据流动路径与调用接口说明。说明基于当前仓库中仍可考证的参数与接口定义，并补充 v0 典型实现行为。需要注意：vLLM v1 已不再支持 best_of；本文仅讨论 v0 语义与离线链路。

### 结论概览
- best_of 仅在 v0 支持；v1 会直接报错拒绝。
- 在 v0 中，best_of=K 表示对同一请求并行采样 K 条候选，再按得分选回前 n 条（n≤K）作为最终输出。
- 参数层会将 best_of“编译”为等价的并行采样（把 n 置为 best_of，保存原始 n 到内部字段），引擎侧像处理 n>1 一样进行并行解码；完成后再按原始 n 截取结果。
- best_of>1 会显著增加显存/缓存占用。v0 通过 CPU 交换区（swap_space）缓解内存压力。

---

## 语义与参数行为

- best_of 的语义（仅 v0 支持）：生成 K 条候选，从中返回前 n 条。参数层明确标注仅 v0 支持：
```145:151:/workspace/vllm/sampling_params.py
best_of: int | None = None
"""Number of output sequences that are generated from the prompt. From
these `best_of` sequences, the top `n` sequences are returned. `best_of`
must be greater than or equal to `n`. By default, `best_of` is set to `n`.
Warning, this is only supported in V0."""
```

- 参数重写逻辑（v0 用于驱动并行采样）：若设置了 best_of，则将 `n` 改写为 `best_of`，并用内部 `_real_n` 记录原始 `n`，用于收尾阶段按原始 n 截取返回：
```340:356:/workspace/vllm/sampling_params.py
# how we deal with `best_of`:
# if `best_of` is not set, we default to `n`;
# if `best_of` is set, we set `n` to `best_of`,
# and set `_real_n` to the original `n`.
# when we return the result, we will check
# if we need to return `n` or `_real_n` results
if self.best_of:
    if self.best_of < self.n:
        raise ValueError(
            f"best_of must be greater than or equal to n, "
            f"got n={self.n} and best_of={self.best_of}."
        )
    if not self._real_n:
        self._real_n = self.n
        self.n = self.best_of
```

- 与流式输出的约束：当 best_of 与 n 不相等时，不允许 DELTA（仅增量）输出，这在 v0/服务端层面常转化为“仅返回最终结果”：
```506:509:/workspace/vllm/sampling_params.py
if self.best_of != self._real_n and self.output_kind == (
    RequestOutputKind.DELTA
):
    raise ValueError("best_of must equal n to use output_kind=DELTA")
```

- v1 不再支持 best_of（用以强调本文限于 v0）：
```144:146:/workspace/vllm/v1/engine/processor.py
# Best of not yet supported.
if params.best_of is not None and params.best_of > 1:
    raise ValueError("vLLM V1 does not yet support best_of.")
```

---

## 离线推理（v0）数据流动路径

下述描述对应 v0 架构的离线链路，体现 best_of 如何“降解”为并行采样与最终筛选：

1) 用户侧发起请求（离线 API）
- 通过 `vllm.LLM.generate(prompts, sampling_params)` 调用，传入 `SamplingParams(best_of=K, n=M, ...)`。
- 建议将 `output_kind` 视为最终输出（FINAL_ONLY），避免中间流式与 best_of 的冲突（见上文约束）。

2) 参数预处理（Processor / SamplingParams）
- `SamplingParams.__post_init__` 将 `n ← best_of`，并保存原始 `n` 到 `_real_n`（见上文代码）。
- 之后引擎在内部仅看见“并行采样 n>1”的需求。

3) 请求入队与扇出（LLMEngine.add_request）
- v0 的行为等价于：当 `n>1` 时，将一个请求扇出为 n 个“子请求/子序列”，共享同一提示但独立解码状态；每个子请求对应一条候选。
- 作为参照，当前代码库 v1 对“n>1 并行采样”的扇出仍保留，逻辑与 v0 类似（注意：v1 不支持 best_of，但扇出机制说明了并行采样如何实现）：
```273:287:/workspace/vllm/v1/engine/llm_engine.py
# Fan out child requests (for n>1).
parent_req = ParentRequest(request_id, params)
for idx in range(n):
    request_id, params = parent_req.get_child_info(idx)
    child_request = request if idx == n - 1 else copy(request)
    child_request.request_id = request_id
    child_request.sampling_params = params

    # Make a new RequestState and queue.
    self.output_processor.add_request(
        child_request, prompt_text, parent_req, idx
    )
    # Add the request to EngineCore.
    self.engine_core.add_request(child_request)
```
- 在 v0 中，best_of 改写为 `n=best_of` 之后，正是以上“并行采样扇出”路径生效。

4) 调度与执行（Scheduler/Executor/Workers）
- 预填充（prefill）与解码（decode）对每个子请求并行/批量进行，产生 K 条候选序列。
- 每条候选独立维护其 KV Cache、停止条件、累计对数概率等统计；多候选带来显存与缓存压力。

5) 结果回收与最终筛选（OutputProcessor / Aggregation）
- 当原始请求达到终止条件（K 条候选均完成、或有足够候选满足 n），引擎按得分对 K 条候选进行排序，返回其中前 M 条（M 为 `_real_n`，即用户原始 n）。
- v0 中的“得分”通常使用累计对数概率（可结合长度惩罚等策略，如果启用）进行排序，行为与主流 best-of 实践一致。

6) 返回用户
- 以 `RequestOutput` 形式返回 n 条最终候选（文本、token_ids、logprobs 等按请求参数决定）。

---

## 资源与内存管理（best_of 的影响）

- best_of>1 会使同一提示持有多份并行解码状态与 KV Cache，显著拉高显存与显存碎片压力。
- v0 提供 CPU 交换区以缓解：
```148:154:/workspace/vllm/entrypoints/llm.py
swap_space: The size (GiB) of CPU memory per GPU to use as swap space.
This can be used for temporarily storing the states of the requests
when their `best_of` sampling parameters are larger than 1. If all
requests will have `best_of=1`, you can safely set this to 0.
Noting that `best_of` is only supported in V0. Otherwise, too small
```
- 调优建议：
  - 合理设置 `swap_space` 与 `gpu_memory_utilization`，防止 OOM。
  - 控制同时并发的“高 best_of”请求数量；可按场景分批。

---

## API 调用接口（离线，v0 视角）

- 构造引擎（离线）
  - `vllm.LLM(model=..., gpu_memory_utilization=..., swap_space=...)`
  - 用于离线批推理；内部选择 v0/v1（注意：仓库当前 v1 默认，不再支持 best_of）。

- 采样参数（关键字段）
  - `SamplingParams(n, best_of, temperature, top_p, top_k, min_p, max_tokens, ...)`
  - best_of 语义与改写逻辑见上文代码引用。

- 生成接口
  - `LLM.generate(prompts, sampling_params)`
  - 返回 `list[RequestOutput]`；离线场景通常仅需要最终结果（FINAL_ONLY）。

- 流式与 best_of（服务端参考）
  - OpenAI 兼容服务端在 `n != best_of` 时不进行流式，强调“best_of 仅 v0 支持”：
```239:246:/workspace/vllm/entrypoints/openai/serving_completion.py
# Similar to the OpenAI API, when n != best_of, we do not stream the
# results. Noting that best_of is only supported in V0. In addition,
# we do not stream the results when use beam search.
stream = (
    request.stream
    and (request.best_of is None or request.n == request.best_of)
    and not request.use_beam_search
)
```

### 简要示例（离线使用，v0 语义）
```python
from vllm import LLM, SamplingParams

llm = LLM(
    model="your/model",
    gpu_memory_utilization=0.9,
    swap_space=8,  # best_of>1 时建议开启
)

params = SamplingParams(
    n=1,           # 希望最终返回的条数
    best_of=5,     # 并行采样 5 条，择优返回 n 条
    temperature=0.8,
    top_p=0.95,
    max_tokens=128,
)

outputs = llm.generate([
    "Explain best_of in vLLM v0.",
], sampling_params=params)

for out in outputs:
    for cand in out.outputs:
        print(cand.text)
```

---

## 与 Beam Search 的区分
- best_of：多次独立随机采样，事后按分数“择优”返回前 n 条；易并行、但代价是多路解码与更高内存占用。
- beam_search：系统性扩展/裁剪候选，通常计算上更“共享”，搜索行为与可控性强；但不等价于随机多样性采样。
- 两者不可混用为流式（参考服务端 gating）；在离线 v0 下，使用 best_of 请以最终结果为主。

---

## 实践建议
- 小试最佳化：先以较小 best_of（如 2~4）评估收益与内存成本，再扩大。
- 配置层面：为高并发/长生成场景预留足够 `swap_space` 与 KV Cache 空间。
- 质量与多样性：结合 `temperature/top_p/min_p` 调参，best_of 的收益通常与采样多样性有关。

---

## 参考代码索引
- SamplingParams 对 best_of 的定义与改写：见上文三处引用。
- v1 对 best_of 的禁用（用于界定 v0 范围）：见上文 v1/engine/processor 引用。
- 并行采样扇出（n>1 的实现思路；v0 与 v1 一致）：见上文 v1/engine/llm_engine 引用。
- 服务端关于流式 gating（n != best_of 不流式）：见上文 serving_completion 引用。
- CPU 交换区（swap_space）用于缓解 best_of>1 的内存压力：见上文 LLM 参数文档引用。

---

## 进一步细化：逐跳数据与对象

- 输入对象（送入引擎核）
  - `EngineCoreRequest`（请求入队后的规范化结构，携带文/多模态、采样参数、LoRA 等）：
```44:73:/workspace/vllm/v1/engine/__init__.py
class EngineCoreRequest(
    msgspec.Struct,
    array_like=True,  # type: ignore[call-arg]
    omit_defaults=True,  # type: ignore[call-arg]
    gc=False,
):  # type: ignore[call-arg]
    request_id: str
    prompt_token_ids: list[int] | None
    mm_features: list[MultiModalFeatureSpec] | None
    sampling_params: SamplingParams | None
    pooling_params: PoolingParams | None
    eos_token_id: int | None
    arrival_time: float
    lora_request: LoRARequest | None
    cache_salt: str | None
    data_parallel_rank: int | None
    prompt_embeds: torch.Tensor | None = None
```

- 执行输出（从引擎核返回）
  - `EngineCoreOutput`（每步新 token、logprobs、停止原因、缓存命中数等）：
```102:127:/workspace/vllm/v1/engine/__init__.py
class EngineCoreOutput(
    msgspec.Struct,
    array_like=True,  # type: ignore[call-arg]
    omit_defaults=True,  # type: ignore[call-arg]
    gc=False,
):  # type: ignore[call-arg]
    request_id: str
    new_token_ids: list[int]

    new_logprobs: LogprobsLists | None = None
    new_prompt_logprobs_tensors: LogprobsTensors | None = None

    pooling_output: torch.Tensor | None = None

    finish_reason: FinishReason | None = None
    stop_reason: int | str | None = None
    events: list[EngineCoreEvent] | None = None
    kv_transfer_params: dict[str, Any] | None = None

    trace_headers: Mapping[str, str] | None = None
    # The number of tokens with prefix cache hits.
    num_cached_tokens: int = 0
```

- 聚合输出（返还给用户）
  - `RequestOutput`（一次请求的输出集合）与 `CompletionOutput`（单条候选）：
```84:121:/workspace/vllm/outputs.py
class RequestOutput:
    """The output data of a completion request to the LLM.
    ...
    outputs: list[CompletionOutput],
    finished: bool,
    ...
    num_cached_tokens: int | None = None
```
```22:49:/workspace/vllm/outputs.py
@dataclass
class CompletionOutput:
    index: int
    text: str
    token_ids: GenericSequence[int]
    cumulative_logprob: float | None
    logprobs: SampleLogprobs | None
    finish_reason: str | None = None
    stop_reason: int | str | None = None
```

### 并行采样扇出与种子策略（n/best_of 如何映射为 K 路子请求）

- 父请求与子请求（`ParentRequest`）在 v0 中用于管理 `n>1` 的并行采样（best_of 在参数层已改写为 n）：
```12:21:/workspace/vllm/v1/engine/parallel_sampling.py
class ParentRequest:
    """Info, state & processing for parallel sampling request.
    ...
    request_id: str
    sampling_params: SamplingParams
```
- 子请求采样参数与种子（若设置了 `seed`，每个子请求会偏移种子避免重复）：
```47:76:/workspace/vllm/v1/engine/parallel_sampling.py
def _get_child_sampling_params(...):
    child_sampling_params = copy(self.sampling_params)
    child_sampling_params.n = 1
    if seed is None:
        self.cached_child_sampling_params = child_sampling_params
    else:
        child_sampling_params.seed = seed + index
```
- 子请求 ID 分配与登记：
```78:90:/workspace/vllm/v1/engine/parallel_sampling.py
def get_child_info(self, index: int) -> tuple[str, SamplingParams]:
    child_req_id = f"{index}_{self.request_id}"
    self.child_requests.add(child_req_id)
    return child_req_id, self._get_child_sampling_params(index)
```

### 聚合与返回（FINAL_ONLY 与 DELTA）

- 非流式（FINAL_ONLY）时，父请求收集全部子请求的最终输出，随后一次性返回：
```95:112:/workspace/vllm/v1/engine/parallel_sampling.py
def get_outputs(...):
    if self.sampling_params.output_kind != RequestOutputKind.FINAL_ONLY:
        outputs = [completion_output]
    else:
        self.output_aggregator[completion_output.index] = completion_output
        outputs = [] if self.child_requests else self.output_aggregator
    finished = not self.child_requests
    return self.request_id, outputs, finished
```
- `RequestState.make_request_output()` 将父请求聚合后的若干 `CompletionOutput` 封装到 `RequestOutput`：
```205:216:/workspace/vllm/v1/engine/output_processor.py
if self.parent_req is None:
    outputs = [output]
else:
    request_id, outputs, finished = self.parent_req.get_outputs(
        request_id, output
    )
    if not outputs:
        return None
return self._new_request_output(request_id, outputs, finished, kv_transfer_params)
```
- 注：v0 的 best_of 在收尾会基于每条候选的 `cumulative_logprob` 等分数择优截取 `_real_n` 条；当前仓库 v1 路径不再实现该选择（因 v1 已禁用 best_of），此处仅保留并行采样与聚合示意。

### 逐步时序（离线 best_of）
1. `LLM.generate` 接收 `SamplingParams(best_of=K, n=M, ...)`。
2. `SamplingParams.__post_init__`：`n ← best_of`，保存 `_real_n ← 原始 n`；验证约束（`best_of≥n`、`DELTA` 兼容性等）。
3. `Processor.process_inputs`：构造 `EngineCoreRequest`（含 token/embeds、多模态、LoRA、eos、采样参数）。
4. `LLMEngine.add_request`：若 `n>1`，构造 `ParentRequest`，扇出 K 个子请求（种子偏移），分别入队执行。
5. 引擎核执行：批内并行 prefill/decode；每子请求由 `LogprobsProcessor` 累计 `cumulative_logprob`。
6. `OutputProcessor` 聚合：FINAL_ONLY 模式下父请求收集所有子输出；v0 再按分数选回 `_real_n`，产生 `RequestOutput(outputs=...)`。
7. 用户读取：获得 `RequestOutput`，其内 `outputs` 即最终 n 条文本与 token、logprobs 等。

---

## 完整 API 面板（离线 best_of 相关）

- 引擎构造（离线）
  - `LLM(model: str, *, gpu_memory_utilization: float=0.9, swap_space: float=4, kv_cache_memory_bytes: int|None=None, seed: int|None=None, ...)`
  - 关键：`swap_space` 建议在 `best_of>1` 时设置更大，以缓解显存压力。

- 采样参数（与 best_of 强相关）
  - `SamplingParams(n: int=1, best_of: int|None=None, seed: int|None=None, temperature: float=1.0, top_p: float=1.0, top_k: int=0, min_p: float=0.0, max_tokens: int|None=16, output_kind: RequestOutputKind=FINAL_ONLY, logprobs: int|None=None, prompt_logprobs: int|None=None, ignore_eos: bool=False, stop: list[str]|str|None=None, stop_token_ids: list[int]|None=None, ...)`
  - 约束：
    - `best_of≥n`；若 `output_kind=DELTA`，要求 `best_of==n`。
    - `temperature==0` 时为贪心采样，不支持 `n>1`。
    - `logprobs/prompt_logprobs` 上限受模型与配置限制。

- 生成接口
  - `LLM.generate(prompts: PromptType|Sequence[PromptType], sampling_params: SamplingParams|Sequence[SamplingParams]|None=None, use_tqdm: bool|Callable=True, lora_request: LoRARequest|Sequence[LoRARequest]|None=None, priority: list[int]|None=None) -> list[RequestOutput]`
  - 返回：与输入 `prompts` 同长度的 `RequestOutput` 列表；每个 `RequestOutput.outputs` 含 n 条 `CompletionOutput`。

- 结果结构（只列与 best_of 选择常用字段）
  - `CompletionOutput.text`、`token_ids`、`cumulative_logprob`、`logprobs`、`finish_reason`、`stop_reason`。
  - `RequestOutput.prompt_token_ids`、`prompt_logprobs`、`outputs`、`num_cached_tokens`。

---

## 典型注意事项 / 故障定位
- 开启流式（DELTA）且 `best_of!=n` 将触发参数校验错误（仅允许 `best_of==n` 时 DELTA）。
- OOM/显存碎片：降低并发、减小 `best_of`、增大 `swap_space`、调低 `gpu_memory_utilization`。
- 质量不佳：适当增大 `temperature/top_p/min_p` 保证多样性，结合 `best_of` 才能带来显著收益。

