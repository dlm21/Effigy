"""AMD: what nvidia-smi answers, for an AMD card (no nvidia-smi, no working amd-smi on Windows).

Windows: name, VRAM and driver from DXGI; VRAM in use from the "GPU Adapter Memory" performance counter, which
counts every process (HIP's own mem_get_info only sees the asking process). Linux: the amdgpu driver's sysfs
files (every process too), the name from libdrm's amdgpu.ids. No torch, so the backend's venv can ask.
ponytail: no GPU utilisation (the per-engine counters need two samples).
"""

import ctypes
import functools
import os
import platform
import shutil
from ctypes import wintypes
from pathlib import Path

AMD = 0x1002
DRM = Path("/sys/class/drm")
IDS = Path("/usr/share/libdrm/amdgpu.ids")  # "744C,\tC8,\tAMD Radeon RX 7900 XTX"
KFD = Path("/sys/class/kfd/kfd/topology/nodes")  # ROCm numbers the GPUs in this order
# the cards that run on ROCm: gfx1201 on Windows; gfx1100 on Linux. The Linux wheels have gfx1201 kernels too,
# but the RX 9070 joins only once measured there: steps/_rocm.py's GEMM fix stays off on Linux, measured on gfx1100
ROCM_CARDS = {"nt": ("RX 9070",), "posix": ("RX 7900",)}


class _Desc1(ctypes.Structure):  # DXGI_ADAPTER_DESC1
    _fields_ = [
        ("Description", ctypes.c_wchar * 128),
        ("VendorId", wintypes.UINT),
        ("DeviceId", wintypes.UINT),
        ("SubSysId", wintypes.UINT),
        ("Revision", wintypes.UINT),
        ("DedicatedVideoMemory", ctypes.c_size_t),
        ("DedicatedSystemMemory", ctypes.c_size_t),
        ("SharedSystemMemory", ctypes.c_size_t),
        ("LuidLow", wintypes.DWORD),
        ("LuidHigh", wintypes.LONG),
        ("Flags", wintypes.UINT),
    ]


class _Value(ctypes.Structure):  # PDH_FMT_COUNTERVALUE with the LONGLONG member
    _fields_ = [("CStatus", wintypes.DWORD), ("largeValue", ctypes.c_longlong)]


def _guid(s: str) -> bytes:
    import uuid

    return uuid.UUID(s).bytes_le


def _call(obj, index, *args, argtypes=()):
    """COM method `index` of `obj`'s vtable."""
    vtbl = ctypes.cast(obj, ctypes.POINTER(ctypes.POINTER(ctypes.c_void_p)))[0]
    fn = ctypes.WINFUNCTYPE(ctypes.c_long, ctypes.c_void_p, *argtypes)(vtbl[index])
    return fn(obj, *args)


@functools.cache  # ponytail: once per app run, as machine.sm()
def adapter() -> dict | None:
    """{name, total_mib, driver, luid (Windows) | vram (Linux)} of the AMD adapter with the most dedicated
    memory, or None. None beside
    an NVIDIA card too: a failing nvidia-smi must not report an AMD iGPU as the GPU the steps run on."""
    if shutil.which("nvidia-smi"):
        return None
    if os.name != "nt":
        return _sysfs()
    try:
        factory = ctypes.c_void_p()
        iid = _guid("770aae78-f26f-4dba-a829-253c83d1b387")  # IDXGIFactory1
        if ctypes.windll.dxgi.CreateDXGIFactory1(iid, ctypes.byref(factory)) < 0:
            return None
        found, i = [], 0
        while True:
            a = ctypes.c_void_p()
            if (
                _call(factory, 12, i, ctypes.byref(a), argtypes=(wintypes.UINT, ctypes.c_void_p))
                < 0
            ):
                break  # DXGI_ERROR_NOT_FOUND: no more adapters
            d, umd = _Desc1(), ctypes.c_longlong()
            _call(a, 10, ctypes.byref(d), argtypes=(ctypes.c_void_p,))  # GetDesc1
            dev = _guid("54ec77fa-1377-44e6-8c32-88fd5f44c84c")  # IDXGIDevice: the UMD version
            _call(a, 9, dev, ctypes.byref(umd), argtypes=(ctypes.c_char_p, ctypes.c_void_p))
            _call(a, 2)  # Release
            if d.VendorId == AMD and not d.Flags & 2:  # not the software adapter
                v = umd.value
                found.append(
                    {
                        "name": d.Description,
                        "total_mib": d.DedicatedVideoMemory / 2**20,
                        "driver": ".".join(str(v >> s & 0xFFFF) for s in (48, 32, 16, 0)),
                        "luid": f"luid_0x{d.LuidHigh & 0xFFFFFFFF:08x}_0x{d.LuidLow:08x}_phys_0",
                    }
                )
            i += 1
        _call(factory, 2)
        return max(found, key=lambda g: g["total_mib"], default=None)
    except OSError:
        return None


def _hip_order() -> list[int]:
    """The render minors (renderD128 → 128) of the GPUs in the order HIP numbers them: KFD's nodes with compute
    units (node 0 is the CPU)."""
    out = []
    for node in sorted(KFD.glob("[0-9]*"), key=lambda p: int(p.name)):
        try:
            props = dict(line.split() for line in (node / "properties").read_text().splitlines())
            if int(props.get("simd_count", 0)):
                out.append(int(props["drm_render_minor"]))
        except (OSError, ValueError, KeyError):
            continue
    return out


def _sysfs() -> dict | None:
    """Linux: adapter()'s answer from the amdgpu driver, `vram` the file used_mib() reads, `hip` its HIP device
    index (None if KFD doesn't list it): beside an AMD iGPU, HIP sees two GPUs."""
    found, order = [], _hip_order()
    for dev in DRM.glob("card[0-9]*/device"):
        try:
            if int((dev / "vendor").read_text(), 16) != AMD:
                continue
            total = int((dev / "mem_info_vram_total").read_text())
            device = (dev / "device").read_text().strip()[2:].upper()
            rev = (dev / "revision").read_text().strip()[2:].upper()
        except (OSError, ValueError):
            continue  # not amdgpu, or a card without VRAM files
        try:
            rows = [p for line in IDS.read_text().splitlines() if len(p := line.split(",", 2)) == 3]
        except OSError:
            rows = []
        # the exact revision, else any of the device's (a revision newer than the file names the same card)
        names = [n.strip() for d, r, n in rows if d.strip().upper() == device]
        exact = [
            n.strip() for d, r, n in rows if (d.strip().upper(), r.strip().upper()) == (device, rev)
        ]
        minors = [int(r.name[7:]) for r in (dev / "drm").glob("renderD[0-9]*")]
        found.append(
            {
                "name": (exact or names or [f"AMD GPU {device},{rev}"])[0],
                "total_mib": total / 2**20,
                "driver": platform.release(),  # amdgpu is in the kernel
                "vram": str(dev / "mem_info_vram_used"),
                "hip": next((order.index(m) for m in minors if m in order), None),
            }
        )
    return max(found, key=lambda g: g["total_mib"], default=None)


def active() -> bool:
    """The steps run on an AMD card: one is there and nvidia-smi is not (an AMD iGPU beside an NVIDIA card
    changes nothing)."""
    return shutil.which("nvidia-smi") is None and adapter() is not None


def used_mib() -> float | None:
    """VRAM in use on `adapter()` by every process, or None."""
    if not (g := adapter()):
        return None
    if "vram" in g:
        try:
            return int(Path(g["vram"]).read_text()) / 2**20
        except (OSError, ValueError):
            return None
    pdh, q, c, v = ctypes.windll.pdh, ctypes.c_void_p(), ctypes.c_void_p(), _Value()
    path = f"\\GPU Adapter Memory({g['luid']})\\Dedicated Usage"
    try:
        if pdh.PdhOpenQueryW(None, None, ctypes.byref(q)):
            return None
        try:
            if pdh.PdhAddEnglishCounterW(q, path, None, ctypes.byref(c)) or pdh.PdhCollectQueryData(
                q
            ):
                return None
            if pdh.PdhGetFormattedCounterValue(c, 0x400, None, ctypes.byref(v)):  # PDH_FMT_LARGE
                return None
            return v.largeValue / 2**20
        finally:
            pdh.PdhCloseQuery(q)
    except OSError:
        return None


def smi(*fields: str) -> list[float | None] | None:
    """jobs._smi's answer from the AMD card. "index": its HIP device index on Linux, so jobs._pin_gpu keeps the
    steps off an iGPU (HIP honours CUDA_VISIBLE_DEVICES; Kimodo's text encoder splits itself over every GPU it
    sees and hung); None on Windows (DXGI's order isn't HIP's), and the choice stays HIP's."""
    if not (g := adapter()) or (used := used_mib()) is None:
        return None
    if "index" in fields and g.get("hip") is None:
        return None
    total = g["total_mib"]
    have = {
        "memory.total": total,
        "memory.used": used,
        "memory.free": total - used,
        "index": g.get("hip"),
    }
    return [have.get(f) for f in fields]  # utilization.gpu: None


def rocm_fits() -> bool:
    """run.bat, run.sh: install .venv-ml for ROCm. ponytail: by name, the cards in ROCM_CARDS; other cards when
    wheels for them exist."""
    return active() and any(c in adapter()["name"] for c in ROCM_CARDS[os.name])


if __name__ == "__main__":  # run.bat, run.sh: exit 0 = the ROCm environment, 1 = the NVIDIA one
    g = adapter()
    print(
        f"AMD GPU: {g['name'] if g else 'none'}, nvidia-smi: {shutil.which('nvidia-smi') or 'none'}"
    )
    raise SystemExit(0 if rocm_fits() else 1)
