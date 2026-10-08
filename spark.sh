#!/usr/bin/env bash
# GLM-5.3-Flash on DGX Spark over ssh: build | check | serve | stop | logs [rank] | status
set -euo pipefail
cd "$(dirname "$0")"
[[ -f spark.env ]] || { echo "copy spark.env.example to spark.env and edit it" >&2; exit 1; }
source spark.env

TP=${TP:-4}
case $TP in
  # Concurrency assumes RecoverSSM (one KDA state per request, patched in, on by default).
  # With VLLM_GLM53_RECOVERSSM=0 each request holds 1+k states: drop to 32 / 3.
  4) : "${KV_CACHE_BYTES:=$((26 << 30))}" "${MAX_MODEL_LEN:=524288}" "${MAX_NUM_SEQS:=64}" "${MAX_BATCHED:=16384}" ;;
  2) : "${KV_CACHE_BYTES:=$((8 << 30))}" "${MAX_MODEL_LEN:=163840}" "${MAX_NUM_SEQS:=16}" "${MAX_BATCHED:=8192}" ;;
  *) echo "TP must be 2 or 4" >&2; exit 1 ;;
esac

read -r -a all_hosts <<< "$NODES"
read -r -a all_ips <<< "$FABRIC_IPS"
off=${NODE_OFFSET:-0}
HOSTS=("${all_hosts[@]:$off:$TP}")
IPS=("${all_ips[@]:$off:$TP}")
(( ${#HOSTS[@]} == TP && ${#IPS[@]} == TP )) || { echo "need $TP NODES and FABRIC_IPS from offset $off" >&2; exit 1; }
CTN=${NAME:-glm53}-$off

# Run a command on a host with every argument quoted for the remote shell.
rrun() { local h=$1; shift; ssh "$h" "$(printf '%q ' "$@")"; }

# SPEC_K=0 serves without speculation (for the k sweep: 0, 3, 5, 7).
spec_json() {
  # SPEC_TABLE picks k by running batch size, e.g. [[1,1,7],[2,2,5],[3,64,3]] (knapcio's).
  local table=""; [[ -n "${SPEC_TABLE:-}" ]] && table=",\"num_speculative_tokens_per_batch_size\":${SPEC_TABLE}"
  echo "{\"method\":\"dflash\",\"model\":\"/draft\",\"num_speculative_tokens\":${SPEC_K:-7},\"disable_eagle_block_drop\":true${table}}"
}

run_rank() {
  local r=$1 h=${HOSTS[$1]}
  local env=(
    -e VLLM_HOST_IP="${IPS[$r]}"
    -e NCCL_IB_HCA="=$NCCL_IB_HCA" -e NCCL_SOCKET_IFNAME="=$FABRIC_IF" -e GLOO_SOCKET_IFNAME="$FABRIC_IF"
    -e NCCL_IB_DISABLE=0 -e NCCL_IB_ROCE_VERSION_NUM=2 -e NCCL_IB_ADDR_FAMILY=AF_INET
    -e NCCL_MAX_NCHANNELS="${NCCL_MAX_NCHANNELS:-8}"
    -e PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
    # A runaway allocation fails one request instead of freezing the node.
    -e VLLM_GLM53_MEM_FRACTION=0.92
    -e VLLM_SHM_BROADCAST_BUSY_LOOP_S=0.002
    # The FlashInfer autotune cache stays inside the container: a persisted one
    # deadlocks the next TP>1 boot on v0.31.0 (fixed on main by vllm#57635).
    -e VLLM_FLASHINFER_AUTOTUNE_CACHE_DIR=/tmp/fi-autotune
    -e FLASHINFER_WORKSPACE_BASE=/cache/flashinfer -e VLLM_CACHE_ROOT=/cache/vllm
    -e TRITON_CACHE_DIR=/cache/triton -e TORCHINDUCTOR_CACHE_DIR=/cache/inductor
    -e TILELANG_CACHE_DIR=/cache/tilelang -e CUDA_CACHE_PATH=/cache/nv
    -e HF_HUB_OFFLINE=1 -e VLLM_ENGINE_READY_TIMEOUT_S=3600
  )
  [[ -n "${NCCL_IB_GID_INDEX:-}" ]] && env+=(-e NCCL_IB_GID_INDEX="$NCCL_IB_GID_INDEX")
  for kv in ${EXTRA_ENV:-}; do env+=(-e "$kv"); done

  # The image ENTRYPOINT is already `vllm serve`.
  local args=(
    /model --served-model-name "$SERVED_NAME"
    --tensor-parallel-size "$TP" --nnodes "$TP" --node-rank "$r"
    --master-addr "${IPS[0]}" --master-port "$MASTER_PORT" --distributed-executor-backend mp
    # MLA layers only; the drafter's plain attention layers keep automatic selection.
    --attention-config "{\"backend_per_kind\":{\"mla_attention\":\"${ATTN_BACKEND:-FLASHINFER_MLA_SPARSE_SM90}\"}}"
    --kv-cache-dtype "${KV_CACHE_DTYPE:-fp8_e4m3}" --kv-cache-memory-bytes "$KV_CACHE_BYTES"
    --max-model-len "$MAX_MODEL_LEN" --max-num-seqs "$MAX_NUM_SEQS" --max-num-batched-tokens "$MAX_BATCHED"
    --block-size 2304 --enable-prefix-caching --mamba-cache-mode align
    --long-prefill-token-threshold 2304 --prefill-schedule-interval 8 --gpu-memory-utilization 0.88
    --sparse-indexer-topk-backend per_row --moe-backend "${MOE_BACKEND:-marlin}"
    # Weight-only (BF16 activations) for every non-expert quantized linear: NVFP4 dense,
    # and the block-FP8 / MXFP8 layers a lossless8 checkpoint adds. No-op if absent.
    --kernel-config '{"linear_backend_per_quant":{"nvfp4_w4a4":"marlin","fp8_block_w8a8":"marlin","mxfp8":"marlin"}}'
    --safetensors-load-strategy eager
    --chat-template /opt/glm53/chat-template.jinja
    --enable-auto-tool-choice --tool-call-parser glm47 --reasoning-parser glm45
    --limit-mm-per-prompt '{"image":16,"video":0}'
    --host 0.0.0.0 --port "$PORT"
  )
  (( ${SPEC_K:-7} > 0 )) && args+=(--speculative-config "$(spec_json)")
  (( r > 0 )) && args+=(--headless)
  # shellcheck disable=SC2206
  args+=(${EXTRA_ARGS:-})

  rrun "$h" docker run -d --name "$CTN-r$r" --restart no \
    --gpus all --network host --ipc host --device /dev/infiniband \
    --cap-add IPC_LOCK --ulimit memlock=-1 --ulimit core=0 --ulimit nofile=1048576:1048576 \
    -v "$MODEL_DIR:/model:ro" -v "$DRAFT_DIR:/draft:ro" -v "$CACHE_DIR:/cache" \
    "${env[@]}" "$IMAGE" "${args[@]}"
}

cmd=${1:-}
case $cmd in
  build)
    for h in "${HOSTS[@]}"; do
      tar -c Dockerfile chat-template.jinja patches | ssh "$h" docker build -t "$IMAGE" - &
    done
    wait ;;
  check)
    for h in "${HOSTS[@]}"; do
      # Forum reports: swappiness 0-1 has wedged GB10 under NVRM OOM; DGX OS earlyoom kills
      # the worker; a node stuck at 500-800 MHz slows every rank; kernel 7.0.0-1019 breaks
      # RDMA registration past ~90 GB unless kho=off or cma=128M; iommu.passthrough=1 is
      # NVIDIA's recommended setting. All nodes should match on kernel and driver.
      rrun "$h" sh -c 'echo "$(hostname) kernel=$(uname -r) $(grep -o -e "iommu.passthrough=[01]" -e "kho=[a-z]*" -e "cma=[0-9A-Za-z]*" /proc/cmdline | tr "\n" " ")swappiness=$(cat /proc/sys/vm/swappiness) earlyoom=$(systemctl is-active earlyoom 2>/dev/null) $(nvidia-smi --query-gpu=clocks.sm,clocks.max.sm,driver_version --format=csv,noheader)"'
      for dev in ${NCCL_IB_HCA//,/ }; do
        nd=$(rrun "$h" sh -c "ls /sys/class/infiniband/$dev/device/net 2>/dev/null | head -1") || nd=""
        [[ -n $nd ]] || { echo "$h: $dev has no netdev" >&2; continue; }
        mtu=$(rrun "$h" cat "/sys/class/net/$nd/mtu")
        speed=$(rrun "$h" cat "/sys/class/net/$nd/speed" 2>/dev/null || echo "?")
        echo "$h $dev $nd mtu=$mtu speed=${speed}Mb/s"
        [[ $mtu == 9000 ]] || echo "  WARNING: MTU should be 9000" >&2
      done
    done ;;
  serve)
    if [[ ${DROP_CACHES:-0} == 1 ]]; then
      for h in "${HOSTS[@]}"; do
        ssh "$h" "sudo sh -c 'sync; echo 3 > /proc/sys/vm/drop_caches; echo 1 > /proc/sys/vm/compact_memory'"
      done
    fi
    for h in "${HOSTS[@]}"; do rrun "$h" mkdir -p "$CACHE_DIR"; done
    for ((r = TP - 1; r >= 0; r--)); do run_rank "$r"; done
    echo "waiting for http://${HOSTS[0]}:$PORT/health (first boot compiles kernels; allow up to an hour)"
    for _ in $(seq 720); do
      rrun "${HOSTS[0]}" curl -sf "http://localhost:$PORT/health" > /dev/null 2>&1 && { echo ready; exit 0; }
      sleep 5
    done
    echo "not healthy after an hour; see: $0 logs 0" >&2; exit 1 ;;
  stop)
    for r in "${!HOSTS[@]}"; do rrun "${HOSTS[$r]}" docker rm -f "$CTN-r$r" || true; done ;;
  logs)
    r=${2:-0}; rrun "${HOSTS[$r]}" docker logs -f --tail 200 "$CTN-r$r" ;;
  status)
    for r in "${!HOSTS[@]}"; do
      echo "== ${HOSTS[$r]} $(rrun "${HOSTS[$r]}" docker ps -a --filter "name=^$CTN-r$r\$" --format '{{.Status}}')"
    done ;;
  *) echo "usage: $0 build|check|serve|stop|logs [rank]|status" >&2; exit 1 ;;
esac
