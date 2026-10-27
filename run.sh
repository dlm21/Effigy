#!/usr/bin/env sh
# Start Effigy: sync the backend env, build the UI once, serve on 127.0.0.1 (port 8000 unless chosen in Setup or
# set in EFFIGY_PORT; python -m server opens the browser there).
set -e
cd "$(dirname "$0")"
command -v uv >/dev/null || { echo "uv not found: install it from https://docs.astral.sh/uv/"; exit 1; }
uv sync --frozen
if uv run python -m server.gpu_amd; then
  # AMD RX 7900 / 9070: torch for the system's ROCm + wheels/rocm (docs/howto/amd-rocm-linux.md), installed again
  # only when requirements-ml-rocm.txt changed (.venv-ml keeps a copy)
  if ! cmp -s requirements-ml-rocm.txt .venv-ml/rocm-installed.txt; then
    [ -x .venv-ml/bin/python ] || uv venv --python 3.12 .venv-ml
    uv pip install --python .venv-ml -r requirements-ml-rocm.txt
    cp requirements-ml-rocm.txt .venv-ml/rocm-installed.txt
  fi
else
  UV_PROJECT_ENVIRONMENT=.venv-ml uv sync --frozen --group ml  # steps' venv; a no-op once synced
fi
uv run python -m server.blender_install  # pinned Blender, first run only
uv run python -m server.gltfpack_install  # pinned gltfpack (LODs, web profile), first run only
[ -f ui/dist/index.html ] || (cd ui && npm ci && npm run build)
exec uv run python -m server
