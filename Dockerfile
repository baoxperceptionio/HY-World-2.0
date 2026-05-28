# syntax=docker/dockerfile:1.7
ARG BASE_IMAGE=hyworld2:cuda12.8-gh200
FROM ${BASE_IMAGE}

ARG DEBIAN_FRONTEND=noninteractive
ARG TORCH_CUDA_ARCH_LIST=9.0
ARG MAX_JOBS=8

ENV PYTHONUNBUFFERED=1 \
    PIP_NO_CACHE_DIR=1 \
    PIP_DISABLE_PIP_VERSION_CHECK=1 \
    CUDA_HOME=/usr/local/cuda \
    FORCE_CUDA=1 \
    TORCH_CUDA_ARCH_LIST=${TORCH_CUDA_ARCH_LIST} \
    MAX_JOBS=${MAX_JOBS} \
    HF_HOME=/workspace/.cache/huggingface \
    HF_HUB_CACHE=/workspace/.cache/huggingface/hub \
    TORCH_HOME=/workspace/.cache/torch \
    MPLCONFIGDIR=/workspace/.cache/matplotlib \
    WANDB_DIR=/workspace/.cache/wandb \
    PYTHONPATH=/workspace/HY-World-2.0:/workspace/HY-World-2.0/hyworld2:/workspace/HY-World-2.0/hyworld2/worldgen:/workspace/HY-World-2.0/hyworld2/panogen \
    NVIDIA_VISIBLE_DEVICES=all \
    NVIDIA_DRIVER_CAPABILITIES=compute,utility,graphics \
    OPENBLAS_NUM_THREADS=1 \
    OMP_NUM_THREADS=8

RUN rm -rf /workspace/HY-World-2.0 \
    && mkdir -p /workspace/HY-World-2.0

WORKDIR /workspace/HY-World-2.0

# Keep this layer below the prebuilt FlashAttention 3 and ONNX runtime base.
# Source code is mounted at runtime by docker-compose; nothing from the repo is
# copied into the image.
RUN apt-get update && apt-get install -y --no-install-recommends \
        libglm-dev \
    && rm -rf /var/lib/apt/lists/*

RUN python -m pip install --upgrade pip setuptools wheel packaging \
    && python -m pip install \
        "fastapi" \
        "nanobind" \
        "pybind11" \
        "pytest" \
        "python-multipart" \
        "tensorboard" \
        "transformers[accelerate,tiktoken]==5.2.0" \
        "uvicorn[standard]" \
    && python -m pip install --no-build-isolation \
        "git+https://github.com/nerfstudio-project/nerfview@4538024fe0d15fd1a0e4d760f3695fc44ca72787" \
        "git+https://github.com/rahul-goel/fused-ssim@328dc9836f513d00c4b5bc38fe30478b4435cbb5" \
        "git+https://github.com/nianticlabs/spz.git@v3.0.0"

RUN python - <<'PY'
import importlib

mods = [
    "diffusers", "fastapi", "flash_attn_interface", "fused_ssim",
    "moge", "nanobind", "nerfview", "onnxruntime", "pybind11", "python_multipart",
    "spz", "tensorboard", "transformers", "uvicorn",
]
for mod in mods:
    importlib.import_module(mod)

from transformers import Sam3Model, Sam3Processor, Sam3VideoModel, Sam3VideoProcessor
import onnxruntime as ort

print("SAM3 imports OK", Sam3Model, Sam3Processor, Sam3VideoModel, Sam3VideoProcessor)
print("onnxruntime providers", ort.get_available_providers())
print("HY-World mounted-code runtime imports OK")
PY

CMD ["bash"]
