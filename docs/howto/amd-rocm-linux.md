# Run Effigy on an AMD GPU (Linux, ROCm)

An AMD Radeon RX 7900 (XT, XTX, GRE: gfx1100) or RX 9070 / 9070 XT (gfx1201) on Linux with the system's
ROCm 7.2.4 in `/opt/rocm`. Tested on an RX 7900 XTX (24 GB, CachyOS, kernel 7.3) with the Fast chain only,
as on Windows ([amd-rocm-windows.md](amd-rocm-windows.md)).

Measured 2026-10-26, one run (`tools.bench --preset game-prop --weights fast`, `tests/fixtures/golden/anvil.png`,
raw data `docs/bench/amd_rocm_linux_anvil.csv`): every step through the Godot export, 9,656 faces, 0
non-manifold edges, 1 component, 0 checklist fails. trellis2 146.9 s (torch peak 4.74 GB; the whole card 20.0 GB,
the resident worker and the desktop included), clean 10.0 s, remesh 3.0 s, uv 2.0 s, bake 11.5 s. The first
run of a step is slower: models load from disk.

## Install

1. ROCm 7.2.4 from your distribution (Arch/CachyOS: the `rocm-hip-sdk` packages) or AMD's installer; your
   user in the `video` and `render` groups. Check: `rocminfo | grep gfx` names your card.
2. Clone, then `./run.sh`. It sees the card (no `nvidia-smi`, an "RX 7900" or "RX 9070" in libdrm's
   `amdgpu.ids`: `uv run python -m server.gpu_amd`) and sets up `.venv-ml` from `requirements-ml-rocm.txt`
   instead of `uv.lock`: AMD's torch 2.9.1 built for ROCm 7.2.4 (`repo.radeon.com/rocm/manylinux`, a
   "lightweight" wheel that uses the libraries in `/opt/rocm`, ~4 GB) and the kernel wheels in
   `wheels/rocm/`. Again only when that file changed (`.venv-ml/rocm-installed.txt` is the copy).
3. Install **Fast** in Setup as usual (DINOv3 is gated: accept its licence on Hugging Face first).

The rest is as on Windows: `.venv-skin` is not set up, the character chain and Faithful are not on ROCm yet.
The card's name, VRAM and VRAM in use come from the amdgpu driver's sysfs files (`server/gpu_amd.py`); the
"driver" Setup shows is the kernel's version. The RX 9070 is in the Linux list because the wheels are built for
gfx1201 too; nobody has run it on Linux yet.

- **An AMD iGPU beside the card** (a Ryzen 7000/9000 desktop): HIP lists both. The steps use device 0, the
  7900 XTX on the test machine (`.venv-ml/bin/python -c "import torch; print(torch.cuda.get_device_name(0))"`).
  If it names the iGPU, start the app with `HIP_VISIBLE_DEVICES=<the card's index>`.
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
