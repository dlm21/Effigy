#!/bin/sh
# Linux twin of build_rocm_wheel.bat (docs/howto/amd-rocm-linux.md).
# usage: SP=<build dir> tools/build_rocm_wheel.sh <source dir> <log name>
# SP holds venv-build (Python 3.12, AMD's torch for the system ROCm, ninja, wheel) and gets dist/ and logs/;
# the compiler is the system's (/opt/rocm). The wheel is stripped of its debug info (65 MB -> 0.8 MB for o_voxel).
set -e
[ -n "$SP" ] || { echo "set SP to the build folder first"; exit 1; }
export BUILD_TARGET=rocm GPU_ARCHS="gfx1100;gfx1201" PYTORCH_ROCM_ARCH="gfx1100;gfx1201" ROCM_HOME=/opt/rocm MAX_JOBS=12
mkdir -p "$SP/dist" "$SP/logs" "$SP/raw"
"$SP/venv-build/bin/python" -I -m pip wheel --no-build-isolation --no-deps -v -w "$SP/raw" "$1" >"$SP/logs/$2.log" 2>&1
t=$(mktemp -d)
for w in "$SP"/raw/*.whl; do
  "$SP/venv-build/bin/python" -m wheel unpack -d "$t" "$w" >/dev/null
  find "$t" -name "*.so" -exec strip --strip-debug {} +
  "$SP/venv-build/bin/python" -m wheel pack -d "$SP/dist" "$t"/* >/dev/null  # rewrites RECORD's hashes
  rm -rf "$t"/* "$w"
done
rmdir "$t"
echo "$2: ok"
