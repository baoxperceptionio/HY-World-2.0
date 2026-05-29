# syntax=docker/dockerfile:1.7
ARG BASE_IMAGE=hyworld2:cuda12.8-gh200
FROM ${BASE_IMAGE}

ARG DEBIAN_FRONTEND=noninteractive
ARG TORCH_CUDA_ARCH_LIST=9.0
ARG MAX_JOBS=8
ARG COLMAP_REF=e99036415ec0cf0f75c1d0b8d60fdd91af0d6c68
ARG PYCOLMAP_REF=b6627db2266f098c21c3d7e4b7844b4b90d8e02d

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
        libboost-filesystem-dev \
        libboost-graph-dev \
        libboost-iostreams-dev \
        libboost-program-options-dev \
        libboost-regex-dev \
        libboost-serialization-dev \
        libboost-system-dev \
        libboost-test-dev \
        libceres-dev \
        libeigen3-dev \
        libflann-dev \
        libfreeimage-dev \
        libgflags-dev \
        libglm-dev \
        libgoogle-glog-dev \
        liblz4-dev \
        libmetis-dev \
        libsqlite3-dev \
        libspatialindex-dev \
        libsuitesparse-dev \
    && rm -rf /var/lib/apt/lists/*

RUN python -m pip install --upgrade pip setuptools wheel packaging \
    && python -m pip install \
        "fastapi" \
        "nanobind" \
        "pybind11==2.11.1" \
        "pytest" \
        "pytest-cov" \
        "python-multipart" \
        "rtree" \
        "ruff" \
        "scikit-build-core" \
        "tensorly==0.9.0" \
        "tensorboard" \
        "transformers[accelerate,tiktoken]==5.2.0" \
        "uvicorn[standard]" \
    && (python -m pip install "pymeshlab==2023.12.post2" || python -m pip install "pymeshlab==2025.7.post1") \
    && python -m pip install --no-build-isolation \
        "git+https://github.com/nerfstudio-project/nerfview@4538024fe0d15fd1a0e4d760f3695fc44ca72787" \
        "git+https://github.com/rahul-goel/fused-ssim@328dc9836f513d00c4b5bc38fe30478b4435cbb5" \
        "git+https://github.com/nianticlabs/spz.git@v3.0.0"

RUN git clone https://github.com/colmap/colmap.git /tmp/colmap-src \
    && cd /tmp/colmap-src \
    && git checkout "${COLMAP_REF}" \
    && cmake -S /tmp/colmap-src -B /tmp/colmap-src/build -GNinja \
        -DCMAKE_BUILD_TYPE=Release \
        -DCMAKE_INSTALL_PREFIX=/usr/local \
        -DCUDA_ENABLED=OFF \
        -DGUI_ENABLED=OFF \
        -DCGAL_ENABLED=OFF \
        -DTESTS_ENABLED=OFF \
    && cmake --build /tmp/colmap-src/build --target install --parallel "${MAX_JOBS}" \
    && git clone --recursive https://github.com/colmap/pycolmap.git /tmp/pycolmap-src \
    && cd /tmp/pycolmap-src \
    && git checkout "${PYCOLMAP_REF}" \
    && git submodule update --init --recursive \
    && python -m pip install --no-build-isolation /tmp/pycolmap-src \
    && rm -rf /tmp/colmap-src /tmp/pycolmap-src

RUN python - <<'PY'
import importlib

mods = [
    "diffusers", "fastapi", "flash_attn_interface", "fused_ssim",
    "moge", "nanobind", "nerfview", "onnxruntime", "pybind11", "python_multipart",
    "pycolmap", "pymeshlab", "pytest", "pytest_cov", "rtree", "ruff", "spz",
    "tensorboard", "tensorly", "transformers", "uvicorn",
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
