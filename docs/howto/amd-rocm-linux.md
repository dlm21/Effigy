# Run Effigy on an AMD GPU (Linux, ROCm)

An AMD Radeon RX 7900 (XT, XTX, GRE: gfx1100) on Linux with the system's
ROCm 7.2.4 in `/opt/rocm`. Tested on an RX 7900 XTX (24 GB, CachyOS, kernel 7.3, a Ryzen 9800X3D whose iGPU HIP
sees too): everything but parallax layers.

Measured 2026-10-26, one run each (the first run of a step also loads its models from disk):

| Feature | Run | Result |
|---|---|---|
| Fast (TRELLIS.2) | `tools.bench --preset game-prop --weights fast`, golden anvil (`docs/bench/amd_rocm_linux_anvil.csv`) | every step to the Godot export; 9,656 faces, 0 non-manifold, 0 checklist fails; trellis2 146.9 s, torch peak 4.74 GB |
| Faithful (Pixal3D) | the same with `--weights faithful` | every step; 9,858 faces, 0 non-manifold, 0 checklist fails; trellis2 137.4 s, torch peak 9.3 GB |
| Concepts (Z-Image-Turbo) | brief → `concepts` n 4, "wooden storage crate…" | 4 images, all gates passed, 72 s with the load |
| Refine (Z-Image + ControlNet Tile) | `refine` on the Faithful anvil's bake | 8 views, 81 % of the texels redrawn, 193 s, peak 16.0 GB |
| Characters | `tools.bench --preset game-character`, the archer | every step: re-pose (FLUX.2-klein) 240 s at 20.2 GB, 52/52 joints inside, SkinTokens 14 s on the GPU (QA chose bone heat), Idle/Walk/Jog |
| Parts (GeoSAM2) | `parts` views, then one click on the hat | the hat (with the braid) in 10 s, 2.4 GB; `full` from the side view found no part |
| Motion from text (Kimodo) | "A person waves with the right hand.", 4 s, encoder on the CPU | 120 frames in 30 s; the right arm moves 12× the left |
| Motion from video (GEM-X) | `best`, a 200-frame render of the archer walking | 98 s, 5.8 GB; both legs swing, the hips cross the frame, feet locked |
| **Parallax layers (MoGe-3)** | `depth` | **refused on AMD**: one of FlexGEMM's Triton kernels hung the card ("HW Exception … GPU Hang") and froze the desktop it drives; Setup refuses its install on AMD and `steps/depth.py` stops before loading it |

What differs from Windows: the big GEMMs are right here (torch 2.9.1+rocm7.2.4 on the 7900 XTX: every shape to
4.2M rows in fp32/fp16/bf16 against the CPU), so `steps/_rocm.py`'s row blocks stay a Windows workaround; the steps
are pinned to the big card (`CUDA_VISIBLE_DEVICES`, which HIP honours: Kimodo's text encoder split itself over the
iGPU too and hung); GEM-X's SOMA runs without NVIDIA Warp (engine/gemx/PATCHES.md).

## Install

1. ROCm 7.2.4 from your distribution (Arch/CachyOS: the `rocm-hip-sdk` packages) or AMD's installer; your
   user in the `video` and `render` groups. Check: `rocminfo | grep gfx` names your card.
2. Clone, then `./run.sh`. It sees the card (no `nvidia-smi`, an "RX 7900" in libdrm's
   `amdgpu.ids`: `uv run python -m server.gpu_amd`) and sets up `.venv-ml` from `requirements-ml-rocm.txt`
   instead of `uv.lock`: AMD's torch 2.9.1 built for ROCm 7.2.4 (`repo.radeon.com/rocm/manylinux`, a
   "lightweight" wheel that uses the libraries in `/opt/rocm`, ~4 GB) and the kernel wheels in
   `wheels/rocm/`. Again only when that file changed (`.venv-ml/rocm-installed.txt` is the copy).
3. Install **Fast** in Setup as usual (DINOv3 is gated: accept its licence on Hugging Face first).

The other feature environments (`.venv-skin`, `-parts`, `-moge`, `-kimodo`, `-gemx`) come from Setup's
**Install** as usual: on such a card it `uv pip sync`s uv.lock's packages for the group, minus CUDA
torch and its `nvidia-*` libraries, plus AMD's torch (2.9.1; GEM-X's 2.10), and keeps the list as
`<venv>/rocm-requirements.txt`; the environment is "outdated" when the lock asks for something else
(`server/install.py`). One built by `uv sync` (CUDA torch) shows as outdated: **Repair** replaces it.

The card's name, VRAM and VRAM in use come from the amdgpu driver's sysfs files, its HIP index from KFD's
topology (`server/gpu_amd.py`); the "driver" Setup shows is the kernel's version. The wheels hold gfx1201 kernels
too, but an RX 9070 is not in `ROCM_CARDS["posix"]` until it is measured on Linux: `steps/_rocm.py`'s fix for the
wrong big GEMMs (found on a 9070 XT under Windows) is off on Linux. To try one: run the GEMM check against the CPU
and `tools/smoke_amd.py`, then add "RX 9070" there.

- **The steps are pinned to the card** by its KFD index (`CUDA_VISIBLE_DEVICES`). If
  `.venv-ml/bin/python -c "import torch; print(torch.cuda.get_device_name(0))"` under that variable names another
  GPU, set `HIP_VISIBLE_DEVICES` yourself before `run.sh`: the app then leaves the choice to you.
- **A `.venv-ml` synced for NVIDIA before**: `uv pip install` leaves its `nvidia-*` packages behind; delete
  `.venv-ml` and start `run.sh` again.

Check the kernels: `EFFIGY_DIR=. .venv-ml/bin/python tools/smoke_amd.py`.

## Rebuild the HIP kernel wheels

`wheels/rocm/*-linux_x86_64.whl` are CuMesh, FlexGEMM and o_voxel built with the system's hipcc for gfx1100
and gfx1201 against AMD's torch 2.9.1+rocm7.2.4, at the pins in `.github/workflows/wheels.yml`.

1. A build folder `$SP` with `venv-build`: `uv venv --python 3.12 $SP/venv-build`, then
   `uv pip install --python $SP/venv-build` the torch and triton URLs from `requirements-ml-rocm.txt` plus
   `ninja wheel setuptools pip numpy`.
2. Clone the sources at the pins: JeffreyXiang/CuMesh `12289e1` (with submodules), JeffreyXiang/FlexGEMM
   `6dd94a8`, microsoft/TRELLIS.2 `75fbf01` (with `o-voxel/third_party/eigen`).
3. Apply the three `engine/trellis2/*-rocm-windows.patch` (each already holds its `*-windows.patch`: apply
   only the ROCm one). Copy Eigen into `CuMesh/third_party/cubvh/third_party/eigen`.
4. `SP=<build folder> tools/build_rocm_wheel.sh <source dir> <log name>` for each library (CuMesh takes
   the longest); stripped wheels land in `$SP/dist`. Another architecture: change `GPU_ARCHS` and
   `PYTORCH_ROCM_ARCH` in the script, and `ROCM_CARDS` in `server/gpu_amd.py`.

Linux needed two more lines in the CuMesh patch, both harmless on Windows: `ATen/hip/HIPContext.h` with a
forward slash, and a global `using std::fill_n;` in `cubvh_hip/src/bindings_bind.hip` (Eigen's sparse
headers say `using ::fill_n` in HIP's device pass, and Linux's libstdc++ declares no global one).
