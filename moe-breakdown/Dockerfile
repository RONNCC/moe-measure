# moe-breakdown -- GPU-ready container for profiling MoE models.
#
# Build:
#   docker build -t moe-breakdown .
#
# Quick demo (no GPU required):
#   docker run --rm moe-breakdown --backend synthetic
#
# Profile a HuggingFace MoE model (needs NVIDIA GPU + nvidia-docker):
#   docker run --rm --gpus all moe-breakdown \
#       --backend transformers --model mistralai/Mixtral-8x7B-Instruct-v0.1 \
#       --tokens 32 --passes 2 --out /runs/mixtral
#
# Profile a running vLLM server (any host):
#   docker run --rm --network host moe-breakdown \
#       --backend vllm --model mistralai/Mixtral-8x7B-Instruct-v0.1 \
#       --base-url http://host.docker.internal:8000 --out /runs/vllm

FROM nvidia/cuda:12.3.1-cudnn8-runtime-ubuntu22.04

ARG DEBIAN_FRONTEND=noninteractive
ENV PYTHONUNBUFFERED=1 \
    PYTHONDONTWRITEBYTECODE=1 \
    PIP_NO_CACHE_DIR=1

RUN apt-get update && apt-get install -y --no-install-recommends \
        python3.11 python3.11-venv python3-pip \
        git curl ca-certificates \
    && rm -rf /var/lib/apt/lists/*

RUN python3.11 -m pip install --upgrade pip

WORKDIR /opt/moe-breakdown
COPY pyproject.toml README.md ./
COPY src ./src
COPY scripts ./scripts
COPY configs ./configs

RUN python3.11 -m pip install -e ".[hf]"

# Default: dump artifacts to /runs; mount -v /host/path:/runs
WORKDIR /work
VOLUME ["/runs"]
ENTRYPOINT ["python3.11", "/opt/moe-breakdown/scripts/run_breakdown.py"]
CMD ["--backend", "synthetic", "--out", "/runs"]
