# syntax=docker/dockerfile:1.7
FROM sugar:cuda12.8-gh200

ARG DEBIAN_FRONTEND=noninteractive
ARG MAX_JOBS=8
ARG TORCH_CUDA_ARCH_LIST=9.0
ARG ONNXRUNTIME_REF=v1.23.2
ARG DECORD_REF=v0.6.0

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

WORKDIR /workspace/HY-World-2.0

RUN apt-get update && apt-get install -y --no-install-recommends \
        build-essential \
        ca-certificates \
        ffmpeg \
        git \
        git-lfs \
        libavcodec-dev \
        libavdevice-dev \
        libavfilter-dev \
        libavformat-dev \
        libavutil-dev \
        libegl1 \
        libgl1 \
        libglib2.0-0 \
        libglvnd0 \
        libgomp1 \
        libsm6 \
        libswresample-dev \
        libswscale-dev \
        libxext6 \
        libxrender1 \
        ninja-build \
        pkg-config \
        python3-dev \
    && rm -rf /var/lib/apt/lists/*

COPY . .

# The base image already carries the hard-to-build aarch64/GH200 pieces:
# PyTorch cu128, Open3D, PyTorch3D, and several CUDA rasterization packages.
# Keep that stack, and add the HY-World-specific Python runtime on top.
RUN python -m pip install --upgrade pip setuptools wheel packaging \
    && python -m pip install \
        "accelerate" \
        "diffusers==0.36.0" \
        "easydict" \
        "ftfy" \
        "huggingface_hub[cli]" \
        "imagesize" \
        "kornia" \
        "loguru==0.7.3" \
        "matplotlib==3.10.3" \
        "opencv-python==4.10.0.84" \
        "openai" \
        "peft==0.18.1" \
        "regex" \
        "safetensors==0.7.0" \
        "scikit-image==0.25.2" \
        "scikit-build-core" \
        "sentencepiece" \
        "splines" \
        "timm==1.0.11" \
        "tokenizers==0.22.0" \
        "transformers[accelerate,tiktoken]==4.57.1" \
        "tyro==1.0.8" \
        "viser" \
    && python -m pip install "cupy-cuda12x==13.6.0"

RUN git clone --recursive --branch ${DECORD_REF} https://github.com/dmlc/decord.git /tmp/decord \
    && cmake -S /tmp/decord -B /tmp/decord/build -GNinja \
        -DCMAKE_BUILD_TYPE=Release \
        -DUSE_CUDA=OFF \
    && cmake --build /tmp/decord/build --parallel ${MAX_JOBS} \
    && cd /tmp/decord/python \
    && python -m pip install . \
    && rm -rf /tmp/decord

RUN python -m pip install --no-build-isolation \
        "git+https://github.com/microsoft/MoGe.git@0286b495230a074aadf1c76cc5c679e943e5d1c6"

RUN git clone --recursive --branch ${ONNXRUNTIME_REF} https://github.com/microsoft/onnxruntime.git /tmp/onnxruntime \
    && cd /tmp/onnxruntime \
    && ./build.sh \
        --config Release \
        --update \
        --build \
        --build_wheel \
        --parallel ${MAX_JOBS} \
        --skip_tests \
        --allow_running_as_root \
        --use_cuda \
        --cuda_home /usr/local/cuda \
        --cudnn_home /usr/local/cuda \
        --cmake_extra_defines CMAKE_CUDA_ARCHITECTURES=90 \
    && python -m pip install /tmp/onnxruntime/build/Linux/Release/dist/onnxruntime_gpu*.whl \
    && rm -rf /tmp/onnxruntime

RUN python -m pip install "zim_anything"

RUN python -m pip install "pybind11"

RUN RECAST_PATH=/workspace/HY-World-2.0/hyworld2/worldgen/third_party/recastnavigation \
       python -m pip install hyworld2/worldgen/third_party/navmesh --no-build-isolation

RUN python - <<'PY'
import importlib
mods = [
    "diffusers", "transformers", "safetensors", "open3d", "pytorch3d",
    "cupy", "decord", "moge", "onnxruntime", "utils3d", "zim_anything", "recast",
]
for mod in mods:
    importlib.import_module(mod)
import onnxruntime as ort
print("onnxruntime providers", ort.get_available_providers())
print("HY-World runtime imports OK")
PY

CMD ["bash", "-lc", "if [ -n \"${HYWORLD_CMD:-}\" ]; then exec bash -lc \"${HYWORLD_CMD}\"; else exec sleep infinity; fi"]
