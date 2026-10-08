# GLM-5.3-Flash on DGX Spark: vLLM v0.31.0 plus the patches in patches/*/series.
# Default CUDA 13.0 image (multi-arch index). The -cu129 image builds SM12x kernels
# for 12.0a only, which does not run on GB10 (sm_121).
ARG VLLM_IMAGE=vllm/vllm-openai:v0.31.0@sha256:c1c9f6fd5c109ba7f0546a59f5b2f15fb87f64c77782e90a27b648b42a8e67c3
FROM ${VLLM_IMAGE}

# Optional standalone b12x for the --moe-backend b12x / --linear-backend b12x A/B.
# 1.5.0 is the code FlashInfer merged (flashinfer#5767) and pins CUTLASS DSL 4.7.1,
# same as vLLM 0.31.0. Empty = not installed.
ARG B12X_VERSION=

RUN apt-get update && apt-get install -y --no-install-recommends patch \
    && rm -rf /var/lib/apt/lists/* \
    && if [ -n "$B12X_VERSION" ]; then uv pip install --system "b12x==$B12X_VERSION"; fi

COPY patches /opt/glm53/patches
COPY chat-template.jinja /opt/glm53/chat-template.jinja

# Patches are Python/header diffs against the installed packages (paths under vllm/
# and flashinfer/), applied in series order. To bisect, comment a line out of a
# series file and rebuild.
RUN set -eu; \
    site=$(python3 -c 'import os, vllm; print(os.path.dirname(os.path.dirname(vllm.__file__)))'); \
    cd "$site"; \
    for s in vllm flashinfer; do \
      f=/opt/glm53/patches/$s/series; \
      [ -f "$f" ] || continue; \
      for p in $(grep -v -e '^#' -e '^$' "$f"); do \
        echo "applying $s/$p"; \
        patch -p1 --forward --no-backup-if-mismatch < "/opt/glm53/patches/$s/$p"; \
      done; \
    done; \
    python3 -m compileall -q vllm flashinfer > /dev/null; \
    # An extra pip install can silently move these (NCCL must stay 2.30.x on the fabric).
    uv pip list --system 2>/dev/null | grep -i -E '^(vllm|torch|flashinfer|nvidia-nccl|nvidia-cutlass-dsl|b12x)' || true
