#!/usr/bin/env bash
set -euo pipefail

HYWORLD_ROOT="${HYWORLD_ROOT:-/workspace/HY-World-2.0}"
cd "${HYWORLD_ROOT}"

install_local_extensions() {
  python -m pip install --no-build-isolation --force-reinstall \
    --no-deps \
    "${HYWORLD_ROOT}/hyworld2/worldgen/third_party/gsplat_maskgaussian"

  env RECAST_PATH="${HYWORLD_ROOT}/hyworld2/worldgen/third_party/recastnavigation" \
    python -m pip install --no-build-isolation \
    --no-deps \
    "${HYWORLD_ROOT}/hyworld2/worldgen/third_party/navmesh"
}

check_fused_ssim_cuda() {
  python - <<'PY'
import torch
from fused_ssim import fused_ssim

x = torch.rand(1, 3, 32, 32, device="cuda")
y = torch.rand(1, 3, 32, 32, device="cuda")
out = fused_ssim(x, y, padding="valid")
torch.cuda.synchronize()
print(f"fused_ssim CUDA OK: {float(out.detach().cpu()):.6f}")
PY
}

ensure_fused_ssim_cuda() {
  if check_fused_ssim_cuda; then
    return
  fi

  echo "fused_ssim CUDA check failed; rebuilding for TORCH_CUDA_ARCH_LIST=${TORCH_CUDA_ARCH_LIST:-9.0}" >&2
  TORCH_CUDA_ARCH_LIST="${TORCH_CUDA_ARCH_LIST:-9.0}" \
    FORCE_CUDA="${FORCE_CUDA:-1}" \
    MAX_JOBS="${MAX_JOBS:-8}" \
    python -m pip install --no-cache-dir --no-build-isolation --force-reinstall \
      "git+https://github.com/rahul-goel/fused-ssim@328dc9836f513d00c4b5bc38fe30478b4435cbb5"

  check_fused_ssim_cuda
}

if [[ "${HYWORLD_INSTALL_LOCAL_EXTENSIONS:-1}" == "1" ]]; then
  install_local_extensions
fi

if [[ "${HYWORLD_CHECK_FUSED_SSIM:-1}" == "1" ]]; then
  ensure_fused_ssim_cuda
fi

if [[ $# -gt 0 ]]; then
  exec "$@"
fi

if [[ -n "${HYWORLD_CMD:-}" ]]; then
  exec bash -lc "${HYWORLD_CMD}"
fi

exec sleep infinity
