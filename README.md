# glm-5.3-flash-spark

Serve [GLM-5.3-Flash](https://huggingface.co/zai-org/GLM-5.3-Flash) (NVIDIA's NVFP4 checkpoint) on four or two DGX Sparks connected through a switch. It runs on vLLM v0.31.0 with DFlash2 speculative decoding.

This is a hybrid of three community Spark recipes:
- [Kindling's glm-5.3-flash-gx10](https://github.com/kindlingai/glm-5.3-flash-gx10)
- [knapcio's GLM-5.3-Flash-4x-DGX-Spark-TP4](https://github.com/knapcio/GLM-5.3-Flash-4x-DGX-Spark-TP4)
- [MiaAI-Lab's](https://github.com/MiaAI-Lab) sm_121 and TensorFold work

Everything they measured as a win was ported onto one stock vLLM release. Most of it is a vLLM patch, and every patch names its source, license and upstream status.

> **Status: untested as a whole.** No piece of this repo has run on GB10 yet.
> - Each change comes from a recipe that measured it on 4× or 2× GB10 (the numbers below are theirs, not ours).
> - The patches were ported and checked statically: they apply in order to pristine v0.31.0, compile, and CPU unit tests pass where the code has them.
> - Run the [gates](#gates) before trusting it with real traffic.

## What you need

- **Nodes:** 4 DGX Sparks for TP=4 (the default), or 2 for TP=2.
- **Network:** ConnectX-7 into a RoCE switch; the design target is a MikroTik CRS812.
  - Each node gets a static IPv4 and MTU 9000.
  - Give each of the two PCIe roots (`rocep1s0f0`, `roceP2p1s0f0`) its own subnet, so NCCL can use both.
- **On each node:** Docker with the NVIDIA runtime, passwordless ssh from wherever you run `spark.sh`, and `/dev/infiniband`.
- **Weights,** at the same path on every node:

```bash
hf download nvidia/GLM-5.3-Flash-NVFP4 --revision da920bb0b9f4a06727223a349e55468e38352348 --local-dir /models/GLM-5.3-Flash-NVFP4
```

```bash
hf download incoai/GLM-5.3-Flash-DFlash2 --revision bf582e4eacc1810f76656d1811693ff6c6737d2a --local-dir /models/GLM-5.3-Flash-DFlash2
```

The drafter is CC BY-NC-ND 4.0, so read its model card before commercial use. GLM-5.3-Flash and the NVIDIA quantization are MIT. `canada-quant/GLM-5.3-Flash-DFlash2-G` is a self-trained Apache-2.0 alternative whose author claims slightly higher acceptance at k=7 (via eshorb on the forum); it is untested here.

### Host preparation

These come from Kindling's notes and from incident reports on NVIDIA's DGX Spark forum. `./spark.sh check` prints most of them for each node.

- **Swap:** keep `vm.swappiness` at 10–60, not 0 or 1. Several GB10s wedged under NVRM out-of-memory at 0–1 (eshorb, tonyd615, Sparkdown_Format).
- **Watermarks:** set `vm.watermark_scale_factor=100`. Out of unified memory, a GB10 tends to livelock (pingable, ssh dead, power button only) rather than OOM-kill (Sparkdown_Format).
- **earlyoom:** DGX OS's `earlyoom` (2% threshold) kills the vLLM worker on the head node, which shows up as "died unexpectedly". Disable it on serving nodes or lower its threshold (jcagle).
- **Headless:** run `multi-user.target`. `spark.sh` already sets `--ulimit core=0`.
- **Clocks:** a node can come back stuck at 500–800 MHz after a crash. That slows every rank, about 25% at TP=4. A reboot may not clear it; pulling AC power for 60 s recovered 24–88% (tonyd, nvidiaspark1). Check `clocks.sm` under load.
- **Kernel, driver and firmware:**
  - Keep them identical on every node; syncing them once gave +140% prefill and +37% decode (kimbona.dy).
  - One known-good set is kernel 6.17.0-1031-nvidia, driver 580.173.02 and ConnectX-7 firmware 28.45.4028, which gave about 109 Gb/s per rail (fernando.qi).
  - Kernel 7.0.0-1019 breaks RDMA memory registration above about 90 GB resident on the GPU. This recipe sits near 107 GiB, so TP=4 prefill roughly halves. Boot it with `kho=off` (`nvidia-spark-grub-kho`) or `cma=128M`, or use 6.17.0-1032 (mlau1, ForsakenSilver).
  - DGX OS 7.6 with driver 580.178.04 has freeze reports.
- **IOMMU:** boot with `iommu.passthrough=1`, NVIDIA's recommended setting. It took a ConnectX-7 function from 13 to 112 Gb/s and cut 32k TTFT by 27% (leafy-inn.0z; Neill, NVIDIA).
- **NCCL:** the image's `nvidia-nccl-cu13` must stay 2.30.x. A FlashInfer nightly once downgraded it to 2.29.7, which breaks the fabric (tonyd615). `build` prints the version.
- **Page cache:** model files in page cache can block CUDA allocations (NVRM `NV_ERR_NO_MEMORY`, tonyd615). `DROP_CACHES=1` drops caches before boot.
- **Long uptimes:** the free-memory floor sank about 1.5 GiB a day on one TP=2 server (fernando.qi), so plan periodic restarts.

## Quick start

```bash
cp spark.env.example spark.env
```

Edit `NODES`, `FABRIC_IPS` and the paths in `spark.env`, then:

```bash
./spark.sh check
```

```bash
./spark.sh build
```

```bash
./spark.sh serve
```

```bash
python3 gates/smoke.py --url http://spark1:8000
```

- **`check`** prints, per node:
  - the kernel and its `iommu.passthrough` / `kho` / `cma` boot flags;
  - swappiness, earlyoom, GPU clocks and the driver;
  - each RDMA device's netdev, MTU and link speed.
- **`build`** streams this directory to every node and builds the image there.
- **`serve`** starts one container per node, last rank first, and waits for `/health`. The first boot JIT-compiles kernels, so allow up to an hour; later boots reuse `CACHE_DIR`.
- **Other commands:** `stop`, `status` and `logs [rank]`.

**Two replicas at TP=2.** Run two separate servers rather than `--data-parallel-size 2`, because vLLM spreads a MoE model's experts across all DP×TP ranks. For the second server, use another env file with `TP=2`, `NODE_OFFSET=2`, and a different `PORT`, `MASTER_PORT` and `NAME`. TP=4 is still the better use of four nodes: Kindling measured 194 against 2×83 tok/s aggregate at 8 streams.

### Optional: lossless8 weights

[`tools/lossless8.py`](tools/lossless8.py) rewrites NVIDIA's BF16 attention, KDA and shared-expert projections onto the 8-bit grids they were trained on:
- 128×128 block FP8 for MLA and shared experts, which gives the same codes as Z.ai's FP8 release;
- MXFP8 for KDA.

About 90% of elements come back bit-exact, and a tensor that fails the error gate stays BF16. Decode weight bytes per token per rank fall from 5.23 to 3.52 GiB at TP=4 (−33%), and from 9.91 to 6.67 GiB at TP=2. The scheme is knapcio's, and knapcio measured −7.5 ms per decode step at TP=4.

```bash
python3 tools/lossless8.py /models/GLM-5.3-Flash-NVFP4 /models/GLM-5.3-Flash-NVFP4-l8 --dry-run
```

Drop `--dry-run` to write it. That takes about 148 GiB of new disk; untouched shards are hard-linked. Then point `MODEL_DIR` at the new directory. `spark.sh` already routes these layers to Marlin W8A16.

Despite the name it isn't fully lossless: about 10% of elements move. Forum reports of online FP8 / NVFP4 requantization of the same layers measured +0.003 nat/token, and about 1% NLL. So run the KL gate before serving it.

## The recipe

These are the `spark.sh` defaults. Kindling's numbers are on 4× GB10 with the same switch class; knapcio's are on 4× GB10 over RoCE.

| Setting | Value (TP=4 / TP=2) | Why | Source |
|---|---|---|---|
| Image | `vllm/vllm-openai:v0.31.0`, CUDA 13.0, pinned digest | CUDA 13 builds `12.0f` family kernels, which run on sm_121; the `-cu129` image builds 12.0a only and doesn't | vLLM v0.31.0 release |
| FlashInfer | 0.7.0.post1, as pinned by vLLM, plus 4 patches | b12x's move into FlashInfer (#5767) isn't in any wheel yet and wasn't qualified on SM121 | our review of flashinfer#5767 |
| Checkpoint | `nvidia/GLM-5.3-Flash-NVFP4` @ `da920bb` | <ul><li>Its dense layers 0–2 are NVFP4: about 3% fewer decode bytes than RedHat's.</li><li>Router bias, `A_log` and `dt_bias` are kept in F32; RedHat downcasts them and router near-ties flip.</li><li>The MTP layer is in `ignore`.</li><li>99–103% of BF16 on NVIDIA's evals.</li><li>The forum's best NVFP4 scores: 96.0 hardmode and 0.502 BPB.</li><li>vllm#54150 (the fused gate/up GEMM dequantizes both halves with the gate's scale) is a no-op here because all 12,096 gate/up scale pairs match.</li></ul> | our comparison; NVIDIA model card; ajvazan, p-pugstaller, jetspark, jahnclawdmonet (forum) |
| Chat template | Z.ai's template @ `eb9eb208` with one change: thinking off becomes `Reasoning Effort: Low`, whatever effort is set | The model never saw an empty `<think></think>`; long outputs corrupted 3–8 of 8 times with thinking off | Kindling (finding and change) |
| Parallelism | TP=4 / TP=2, one server per replica | TP=4 is 1.7× faster per stream and at prefill, with 524k against 160k context | Kindling TP table |
| Speculation | DFlash2, `k=7`, `disable_eagle_block_drop` | All three recipes use these weights at k=7. Decode at one request (code / prose / structured), stock + DFlash2 → patched: 91.3 / 38.7 / 121.9 → 117.3 / 70.5 / 164.3 tok/s | Kindling, knapcio, MiaAI |
| MoE | `--moe-backend marlin` (W4A16) | W4A4 kernels clip activations to static input scales: 13–21% MoE output error, 62% past the calibrated amax. Marlin ignores the input scales. "Code +40%, JSON +50%" against the official recipe | knapcio, Kindling |
| Other quantized linears | `linear_backend_per_quant`: NVFP4 dense, block FP8 and MXFP8 → Marlin | Same reasoning: no activation quantization until a KL gate says otherwise | knapcio (lossless8), our policy |
| Sparse MLA | SM90 backend on FlashInfer FA2, MLA layers only (`backend_per_kind`) | Dense MHA prefill and `fp8_e4m3` KV. Stock picks the SM120 backend, which forces `fp8_ds_mla` and has no dense prefill | MiaAI (sm_121 patch), Kindling, knapcio |
| Sparse prefill | Triton sparse-MLA kernel when SM90 runs on SM12x | 3.3× faster sparse prefill than FlashInfer's | knapcio |
| KV cache | `fp8_e4m3`, 26 GiB / 8 GiB per rank, pinned with `--kv-cache-memory-bytes`; the drafter's KV in its own small pool (patch) | 4.4M / 1.1M token pool. 28 GiB OOM-killed a worker. Unpinned, identical boots sized the pool 1.66M or 2.21M tokens because unified memory reported a negative CUDA-graph size, and the big one swapped the host (−20 to −40% on every timing) | Kindling, tonyliu312 and miken (KV pin) |
| Context | 524,288 / 163,840 | needle 12/12 to 507k at TP=4 | Kindling |
| Concurrency | 64 / 16 sequences | Needs RecoverSSM (patched in): otherwise each request holds 1+k KDA states and TP=2 fits 3 | Kindling |
| Batched tokens | 16,384 / 8,192 | | Kindling |
| Block size | 2304 (vLLM raises it to 4608 at TP=2) | The smallest block whose MLA page holds a rank's KDA state, and its kpool tiles fit DeepGEMM's 64-entry pages. A TP=2 rank holds twice the KDA heads, so its state needs twice the block. Prefix caching uses `--mamba-cache-mode align` | Kindling, knapcio, tonyd615 |
| Long prefills | `--long-prefill-token-threshold 2304`, `--prefill-schedule-interval 8` | A 12-token request behind a 120k prefill: 78–90 s → 4.8 s. Decode under a 32k prefill: 1.3 → 7.3 tok/s | Kindling; knapcio (cadence, after jnardiello's E27) |
| Memory | `--gpu-memory-utilization 0.88`, `VLLM_GLM53_MEM_FRACTION=0.92`, `expandable_segments` | 0.90 boots, then wedges the box hours later; the fraction makes a runaway allocation fail one request, not the node | Kindling |
| Indexer top-k | `per_row` plus the deterministic top-k patch | `persistent_topk` breaks past about 3.4M KV tokens; arbitrary tie-breaks made identical requests drift by up to 9.8 nats | Kindling |
| Loading | `--safetensors-load-strategy eager` | 511 s against 690 s for lazy. One forum report had TP=2 nodes wedge during an eager load at about 89 GiB per rank; if that happens, add `--safetensors-load-strategy lazy` to `EXTRA_ARGS` | Kindling; xander.no (forum) |
| Fabric | both PCIe roots in `NCCL_IB_HCA`, RoCE v2, GID unpinned, exact `*_SOCKET_IFNAME`, `NCCL_MAX_NCHANNELS=8` | <ul><li>One root tops out near 110 Gb/s; both reach 180–191 Gb/s, +11% prefill.</li><li>The GID moves across reboots.</li><li>Gloo binds 127.0.0.1 without the interface name.</li><li>8 channels: 4,905 against 4,769 tok/s at 32k.</li></ul> | Kindling; MiaAI (try 4 channels at TP=2) |
| Host process | shm busy-wait 0.002 s instead of 1 s, `ulimit -c 0` | Faster decode and about 20 °C cooler; a core dump holds the NVIDIA RM lock | nacyot (spin wait, via Kindling), Kindling |
| Caches | Triton / Inductor / TileLang / CUDA / FlashInfer JIT caches persisted; FlashInfer **autotune** cache wiped each boot | JIT caches cut boot 271 → 128 s. A persisted autotune cache deadlocks the next TP>1 boot on v0.31.0 (fixed on main by vllm#57635) | knapcio; Kindling |

## Patches

`patches/vllm/series` and `patches/flashinfer/series` are applied in order at image build time. To bisect, comment a line out and rebuild. Each `.patch` is a `git format-patch` file whose message has the full source, license, upstream status and claimed gain.

**vLLM v0.31.0**

| Patch | What it does | Source (license) | Upstream |
|---|---|---|---|
| Mamba state index on prefix hits | Seeds the hybrid state index with `mamba_block_size` | vllm#55601 (Apache-2.0) | merged after v0.31.0 |
| Mamba align chunking | <ul><li>Splits align-mode chunks on the 2304-token KDA block, not the drafter's 1152. Otherwise a prefix hit resumes KDA state half a block early.</li><li>DFlash no longer drops the last cached block on a hit. The equivalent MiaAI fix took hits from 85% to 99% of a 32k prompt.</li><li>Caps a non-final align chunk at one KDA block per request per step, the same cap `--long-prefill-token-threshold 2304` already sets.</li></ul> | vllm#54076 by wickist (Apache-2.0); bugs found by knapcio and miken | open |
| PDL off on SM12x | Programmatic dependent launch races in the KDA-state and mHC kernels on GB10 | MiaAI-Lab (MIT) | none |
| Argmax clamp | Tile-local argmax can't return a padded-vocab id | vllm#50843 by alexbi29 (Apache-2.0) | open |
| shm busy-wait knob | `VLLM_SHM_BROADCAST_BUSY_LOOP_S` | vllm#52814 by tobymao (Apache-2.0); GB10 finding by nacyot | open |
| Deterministic kpool top-k | -1-initialized buffers, pool id bounds check, ties to the lowest pool id (`VLLM_GLM5NEXT_DETERMINISTIC_TOPK`) | MiaAI-Lab (MIT); approach from Kindling, re-implemented | none |
| DFlash aux hidden states | `SupportsEagle3` for Glm5Next, through mHC | vllm#56983 by Dovis01 (Apache-2.0) | open |
| DFlash drafter KV groups | The drafter's sliding-window layers get their own KV groups, so DFlash2 starts at all | vllm#55423 by zixi-qi (Apache-2.0), adapted | closed in favor of #56983 / #55219 |
| SM90 sparse MLA on SM12x | Runs the SM90 backend through FlashInfer FA2 on capability 12, listed after SM120 | MiaAI-Lab (MIT) | none |
| SM90 fp8 plan + indexer workspace | | vllm#55222 by drakosha (Apache-2.0) | merged after v0.31.0 |
| Triton sparse MLA (opt-in) | `VLLM_TRITON_SPARSE_MLA=1` sends every row through Triton | Kindling | none |
| Triton sparse-MLA prefill | On by default when SM90 runs on SM12x; `VLLM_TRITON_SPARSE_MLA_PREFILL=0` turns it off | knapcio (MIT/Apache-2.0); idea from mmastrac and chuck-ads | none |
| lossless8 loader (2 patches) | Lets ModelOpt mixed precision quantize the KDA and MLA projections, and skips the FP8 dequant when the target is quantized | knapcio (MIT) | none |
| RecoverSSM for KDA | One checkpoint state per request; accepted tokens are replayed bit-exactly. TP=2 concurrency 3 → 16 (+44%), TP=4 +28% aggregate | Kindling, after vLLM's Kimi-K3 RecoverSSM (Apache-2.0) | none for GLM |
| mHC BF16 weight reads | The fused decode kernel reads exact BF16 copies, with a bitwise probe that fails closed. 11.9 → 8.2 µs per call | Kindling; gate idea from knapcio | none |
| KDA conv `out_group` | Prefill writes dense q, k, v. Prefill +2–3% | Kindling | none |
| Sequence-parallel prefill | SP over plain TP at ≥1024 tokens (`VLLM_GLM53_SP_PREFILL`). Prefill +17% at TP=4 | Kindling (stock collectives only) | none |
| Indexer head gate | BF16 operands, FP32 output: −69 µs per indexer layer | Kindling | none |
| GLM-5.3 reasoning always parsed | | vllm#56994 by Shijin Zhang (Apache-2.0) | merged after v0.31.0 |
| GLM tool markup without tools | Tool-call markup stays content when the request has no tools | knapcio (MIT), re-implemented | none |
| Prefill interval without DP | `--prefill-schedule-interval` works on a single engine | knapcio (MIT), after jnardiello's E27 | none |
| Worker memory cap | `VLLM_GLM53_MEM_FRACTION` | Kindling | none |
| Separate drafter KV pool | <ul><li>The drafter's five sliding-window layers get their own uncached pool of 256-token blocks, sized for their 2048-token windows.</li><li>Without it every 2304-token block carries a drafter page: 30% of the KV at TP=4, 46% at TP=2.</li><li>By the patch's arithmetic at 32 sequences: target blocks 1,416 → 1,991 at TP=4 (+41%) and 335 → 592 at TP=2 (+77%).</li><li>After a prefix hit the drafter starts fresh inside its window, so acceptance can dip for up to 2048 tokens; output is unchanged.</li><li>Needs Model Runner V2. `VLLM_GLM5NEXT_DRAFT_POOL=0` turns it off.</li></ul> | Kindling (re-implemented natively); vllm#55423 by zixi-qi for the idea | none |

**FlashInfer 0.7.0.post1**

| Patch | What it does | Source (license) |
|---|---|---|
| `0001` | Allows FP8 KV for FA2 batch MLA on SM12x | MiaAI-Lab (MIT) |
| `0002` | Skips the per-`plan()` snapshot clone when the source is the reserved buffer (about 136 MB per step) | Kindling idea, made safer |
| `0003` | Caps the FP8 MLA KV tile at 32 so it fits SM12x shared memory | MiaAI-Lab (MIT) |
| `0004` | Gives the patched module its own JIT URI, so a shared cache can't serve an unpatched build | ours |

**Already in v0.31.0, so not carried:** the FlashKDA fp32 state (#58846), kpool fixes (#57477, #58454, #53906, #58704), router dedup (#55736), vocab-parallel argmax (#34049), NoPE fp8_ds_mla (#55277), fused DFlash2 conv (#55960), DFlash async scheduling (#58065), GDN metadata reuse (#58762), drafter EP fix (#56930), the expert-mapping speedup (#58720), and verify-shape CUDA-graph capture (#50488, #58400).

## A/B switches

| Switch | Default | Try | Gate on |
|---|---|---|---|
| `ATTN_BACKEND` | `FLASHINFER_MLA_SPARSE_SM90` | `FLASHINFER_MLA_SPARSE_SM120`, the stock path; it converts KV to `fp8_ds_mla` itself | KL, TTFT, decode |
| `MOE_BACKEND` | `marlin` | `flashinfer_cutlass` (W4A4); `b12x` with `VLLM_B12X_MOE_FP4_FORCE_A16=1` (build with `--build-arg B12X_VERSION=1.5.0`). One forum rig saw b12x corrupt output at 4–6 concurrent streams (stu.miller) | KL first at several concurrencies, then prefill and decode |
| `VLLM_TRITON_SPARSE_MLA_PREFILL_CFG` | `64,4,1,1` (tuned at TP=4) | `64,8,1,1` or `32,8,2,1` at TP=2 | TTFT at 32k / 128k |
| `NCCL_MAX_NCHANNELS` | 8 | 4 at TP=2 | all-reduce / all-gather latency |
| `MAX_BATCHED` | 16384 / 8192 | 4096 at TP=2. On one TP=2 server that cut short-request TTFT behind a long prefill from 8.1 to 5.7 s, at a cost of 6% solo TTFT and 11% peak aggregate (fernando.qi) | TTFT under mixed load |
| `SPEC_K` | 7 | 0 (no speculation), 3, 5 | Aggregate tok/s at your concurrency, on real prompts: 20-token prompts overstate it about 5× (fernando.qi). DFlash2 k=7 wins code (acceptance 0.68–0.71) and structured output but not prose (0.21–0.25), per miken. A TP=2 rig stayed flat at about 45 tok/s from 4 to 16 streams (jcagle); RecoverSSM should lift that |
| `SPEC_TABLE` | unset (k=7 at every batch size) | `[[1,1,7],[2,2,5],[3,64,3]]`, knapcio's table, which lowers k as concurrency rises. Forum reports: k=4 gave +8.6–20% at TP=2, and picking k by concurrency gave 1.7–1.9× at 16 streams (voktolom, jetspark). Those were without RecoverSSM, and our RecoverSSM port hasn't run with a varying k yet | aggregate tok/s at 1 / 4 / 16 streams, plus the RecoverSSM gate |
| Prefix-cache granularity | hits on 2304-token blocks (4608 at TP=2) | `--prefix-match-unit 128 --prefix-cache-retention-interval 9216` in `EXTRA_ARGS`. The finer unit replayed about 82 tokens per agent turn instead of about half a block (florianbrede). Retained checkpoints kept 99.94% hits on 5×250k when unrelated long prompts arrived (florianbrede, another model), but each one holds a KDA state block while its request runs. The interval must be a multiple of the block; 9216 works at both TPs | hit rate and warm TTFT on agent replays; KV pool use |
| `--long-prefill-token-threshold` at TP=2 | 2304 | 4608, the TP=2 block, in `EXTRA_ARGS`: +16% long prefill on one GLM rig (stuart.trusty), at the cost of short-request latency behind a long prefill | prefill tok/s and short-request TTFT |
| Reasoning effort | `max` (thinking on), `low` (thinking off) | `high` per request, or as the server default: `EXTRA_ARGS='--default-chat-template-kwargs {"reasoning_effort":"high"}'` in `spark.env` (`EXTRA_ARGS` splits on spaces, so no spaces inside the JSON). Thinking off still renders `Low`. Forum results: on an 88-question hard set `high` scored the same as `max` with 31% faster median turns and 13% fewer tokens (eshorb). On TEB, `max` took about twice the wall time of `low` and scored lower, 85 against 89 (jetspark) | your own eval |
| `VLLM_GLM53_RECOVERSSM`, `VLLM_GLM53_MHC_BF16W`, `VLLM_GLM53_SP_PREFILL`, `VLLM_GLM5NEXT_DETERMINISTIC_TOPK`, `VLLM_GLM5NEXT_DRAFT_POOL` | on | `0` | to isolate a regression |

Pass env through `EXTRA_ENV` and flags through `EXTRA_ARGS` in `spark.env`. A flag in `EXTRA_ARGS` overrides the same flag set earlier by `spark.sh`.

**Clients:** leave `repetition_penalty` at 1.0. vLLM v0.31.0 applies it to prompt tokens as well as output, and 1.2 produced word salad in long chats (mclenithan).

## Gates

**`gates/smoke.py`** (stdlib only) checks:
- arithmetic;
- a tool call;
- greedy determinism over three runs, printing decode tok/s;
- a prefix-cache hit (warm TTFT under half of cold).

Greedy output is not bit-reproducible across *boots* on any of these stacks, because the autotuner picks different kernels. So also gate quality on KL against a reference: knapcio's `bench/kld_probe.py` and `compare_kld_strict.py`, including an A/A run for the noise floor. Gate every MoE / linear backend change and the lossless8 checkpoint that way.

**To validate on hardware, at TP=4 and TP=2:**
1. Boot with DFlash2 and check the draft acceptance rate. The boot log should have no w1/w3 `weight_scale_2` mismatch warning.
2. Check that the `_fp8tilecap` FlashInfer module JIT-builds without a shared-memory error.
3. KL of SM90 against SM120.
4. Triton prefill against `=0`.
5. RecoverSSM on against `=0`, which should be bit-identical.
6. SP prefill gives matching greedy output at 1k / 8k / 32k.
7. The log says mHC "reads 90 of 90 weights in BF16".
8. Prefix-cache hit rate on a repeated 20k prompt.
9. A long-context decode sweep (35k / 226k / 500k).
10. The boot log's target and draft block counts, with the drafter pool on and off.
11. Repeat a 24k prefill about ten times after mixed traffic. On another stack, DFlash2 with the hybrid cache made every other run 20–86% slower until a restart (fernando.qi).

In production, watch a real completion (for example `gates/smoke.py` on a timer), not `/health`. `/health` can stay 200 while the engine is deadlocked (vllm#37729, jamieo1337).

## Not included yet

These were measured by the source recipes or reported on the forum but not ported. The gains are the sources' own, mostly at TP=2 on other stacks.

| Change | Claimed gain | Source |
|---|---|---|
| NVFP4 KV on the sparse-MLA path (Triton gather into the FP8 kernel) | 288 against 656 bytes per token | Sparkdown_Format (forum) |
| 528-byte NoPE MLA layout, FP8 indexer cache, DCP2 cache sharding | about 4.7M tokens at 1M context on two Sparks | emihuang, coolbho3k (forum) |
| MXFP8 DFlash2 drafter | faster decode, acceptance barely changed | emihuang (forum); Local Inference Lab's gated MXFP8 drafter |
| Display-reserve KV (headless, driver 580.159) | +1.75 GiB KV per rank | coolbho3k's allocator, measured by jcagle (forum) |
| RoCEnante one-shot all-reduce / all-gather (now in FlashInfer, API changed) | about 17 µs against 41–56 µs per small collective | b12x (Luke Alonso, Jason Cook), knapcio's port |
| megamoe W4A16 decode / fused W4A4 prefill | decode +7–8% | Kindling |
| KDA stash, L2 prefetch, knapcio's decode set | −0.9 / −0.5 / −1.9 ms per step | knapcio |
| pf3 MoE, SP FP8 gather, arxbig reduce-scatter | +3–4%, +2%, +2.5–3% | knapcio, Kindling |
| DFlash2-coupled levers (draft truncation, coupled Gumbel, cert head, adaptive k) | see the sources | knapcio, Kindling, MiaAI |

## Credits

This recipe is mostly other people's work. Thanks to:

- **[Z.ai](https://huggingface.co/zai-org/GLM-5.3-Flash)** for GLM-5.3-Flash, and **NVIDIA** for the [NVFP4 checkpoint](https://huggingface.co/nvidia/GLM-5.3-Flash-NVFP4) and Model Optimizer.
- **[Inco AI](https://huggingface.co/incoai/GLM-5.3-Flash-DFlash2)** for the DFlash2 drafter.
- **[Kindling AI / Matt Mastracci](https://github.com/kindlingai/glm-5.3-flash-gx10)**:
  - RecoverSSM for GLM, SP prefill, mHC BF16, KDA `out_group`, the indexer head gate, the Triton sparse MLA, the memory cap, the FlashInfer `plan()` finding, the thinking-off template finding;
  - most of the KV, memory, fabric and host settings, and the TP=2 vs TP=4 measurements.
- **[knapcio](https://github.com/knapcio/GLM-5.3-Flash-4x-DGX-Spark-TP4)**:
  - lossless8, the Triton sparse-MLA prefill, Marlin W4A16 MoE and its KL case against W4A4, the mamba alignment bug, the parser fix, the prefill cadence, the JIT-cache boot work;
  - the KL tooling.
- **[MiaAI-Lab](https://github.com/MiaAI-Lab)**: the sm_121 FA2 patches (MIT), PDL off, the indexer bounds fixes, NCCL channel numbers for two nodes, and TensorFold's copy-draft and fabric work.
- **vLLM contributors**, whose PRs are carried here: Dovis01 (#56983), zixi-qi (#55423, #34049), wickist (#54076), alexbi29 (#50843), tobymao (#52814), drakosha (#55222), Shijin Zhang (#56994), plus ivanium, mmastrac, JaredforReal and ZJY0516 for the GLM-5.3 and kpool work it builds on.
- **[FlashInfer](https://github.com/flashinfer-ai/flashinfer)** and **[b12x](https://github.com/local-inference-lab/b12x)** (Luke Alonso, Jason Cook, Local Inference Lab).
- **tonyd2wild** (the first sm_121 DFlash2 image, NoPE/kpool and prefix-cache findings), **jnardiello** (prefill-cadence experiments), **Alex Ellis** (the NVFP4 + Marlin 4-Spark recipe and RigMark), **nacyot** (the spin-wait finding), **tonyliu312** (the KV pin), **chuck-ads** (the RDMA MTU and sparse-MLA kernel ideas).
- **Members of NVIDIA's [DGX Spark forum](https://forums.developer.nvidia.com/c/accelerated-computing/dgx-spark-gb10/dgx-spark-gb10/721)**, mostly in the [GLM-5.3-Flash thread](https://forums.developer.nvidia.com/t/glm-5-3-flash-320b-total-parameters-18b-active/381350):
  - Wpnx330: the image ENTRYPOINT is already `vllm serve`, and `fp8` KV auto-selects `fp8_ds_mla`;
  - tonyd615: the sm_121 bug list (block 2304, NCCL downgrade, page cache);
  - miken: KV pinning, the dropped prefix block, DFlash2 acceptance by prompt type;
  - jcagle: earlyoom, the DFlash2 concurrency ceiling, display-reserve KV;
  - fernando.qi: batched-token A/B, memory aging, fabric versions, the alternating-prefill bug;
  - Sparkdown_Format: the livelock and NVFP4 KV;
  - eshorb: swappiness 60 while serving;
  - emihuang and coolbho3k: compact KV layouts;
  - jahnclawdmonet: align-mode chunk clipping;
  - tonyd and nvidiaspark1: the post-crash clock drop and the AC power-cycle fix;
  - mlau1 and ForsakenSilver: the kernel 7.0 RDMA registration bug;
  - leafy-inn.0z and Neill (NVIDIA): `iommu.passthrough=1`;
  - kimbona.dy: node symmetry;
  - voktolom, jetspark and florianbrede: speculation depth, prefix-cache retention and match unit;
  - mclenithan: repetition penalty;
  - stuart.trusty: scheduler tuning;
  - xander.no: eager loading at TP=2;
  - jamieo1337: the `/health` deadlock;
  - brian361 and lewald_jens: TP=4 spec-decode and B12X stack numbers.

[NOTICE](NOTICE) lists the third-party code included here and its licenses.

## License

Apache-2.0 ([LICENSE](LICENSE)). Some patches carry MIT-licensed code from MiaAI-Lab and knapcio; their notices are in [LICENSES/](LICENSES/) and [NOTICE](NOTICE). Model weights are not distributed here and keep their own licenses: GLM-5.3-Flash and NVIDIA's NVFP4 checkpoint are MIT, and DFlash2 is CC BY-NC-ND 4.0.
