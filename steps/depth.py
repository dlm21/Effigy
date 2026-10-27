"""Depth of a picture (M55, research 15 §5.5): Microsoft MoGe-3 ViT-g (MIT) in `.venv-moge`, for parallax layers.

Input: a version with a picture (`image`, or one of a set with `key: "image_2"`). Params: key; resolution_level:
0-9 (default 9, MoGe's finest). Outputs depth.npz {disparity: float16 0-1 (1 the nearest, 0 the sky and anything
MoGe finds no depth for), valid: bool}, preview.png (the disparity in grey), image.png (the picture, which `layers`
takes with it). About 11 s and 8.9 GB on the reference laptop (docs/bench/m55.md). A sky or any pixel MoGe marks
invalid counts as the farthest. AMD: without Triton (`_moge_without_triton`).
"""

import json
import re
import time
from pathlib import Path
from types import SimpleNamespace

import numpy as np
from PIL import Image

from steps._common import StepError, main, progress

MODEL = "Ruicheng/moge-3-vitg"
REVISION = "6ef26c5a4b4148dab5ccaacdb08b72dc66380475"


def disparity(depth: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """(disparity scaled to 0-1, valid): 1 / depth where MoGe gave one, 0 elsewhere (the sky: infinitely far)."""
    valid = np.isfinite(depth) & (depth > 0)
    out = np.zeros(depth.shape, np.float32)
    out[valid] = 1 / depth[valid]
    return out / max(float(out.max()), 1e-6), valid


def _moge_without_triton():
    """AMD: MoGe-3's refiner (a sparse U-Net) without FlexGEMM's Triton kernels (b2fadb2), one of which hung an
    RX 7900 XTX ("HW Exception ... GPU Hang", ROCm 7.2.4) and froze the desktop it drives: FlexGEMM's own PyTorch
    neighbour maps and explicit_gemm convs, and the two layers that have no such path in plain torch (a pool with
    kernel = stride, the mean of the voxels present; nearest upsampling, each voxel its parent's value: FlexGEMM's
    rules). A Triton launch left fails the step instead of reaching the card. 1344 × 768, level 9: 1.9 s, 10.8 GB."""
    import flex_gemm
    import torch
    import triton.runtime.jit
    from moge.model.modules import flex_sparse_blocks as fsb

    def no_triton(kernel, grid):
        raise StepError(
            "failed", f"depth on AMD reached a Triton kernel ({kernel}), which can hang the card"
        )

    def pool(self, feats, coords, shape):
        coarse = coords.clone()
        coarse[:, 1:] //= self.factor
        coarse, inverse = torch.unique(coarse, dim=0, return_inverse=True)
        n = torch.zeros(len(coarse), 1, device=feats.device, dtype=feats.dtype)
        n.index_add_(0, inverse, torch.ones_like(feats[:, :1]))
        mean = torch.zeros(len(coarse), feats.shape[1], device=feats.device, dtype=feats.dtype)
        feats = self.linear(mean.index_add_(0, inverse, feats) / n)
        out = fsb._ceil_downsample_shape(shape, coords.shape[1], self.factor)
        return (
            feats,
            coarse,
            fsb._with_channels(out, feats.shape[-1]),
            SimpleNamespace(T=inverse),  # the U-Net hands the upsample this cache's .T
        )

    def up(self, feats, coords, shape, target_coords, target_shape, up_cache=None):
        feats = self.linear(feats)[up_cache]  # each fine voxel takes its coarse voxel's row
        return feats, target_coords, fsb._with_channels(target_shape, feats.shape[-1])

    triton.runtime.jit.KernelInterface.__getitem__ = no_triton
    flex_gemm.config._USE_PYTORCH_FOR_TEST = True
    fsb._pick_spconv_algorithm = lambda in_channels, out_channels: "explicit_gemm"
    fsb.PoolDown.forward, fsb.NearestUp.forward = pool, up


def run(op, in_dir, out_dir, params):
    if op != "run":
        raise StepError("bad_params", f"op {op!r}: only run")
    if not in_dir:
        raise StepError("missing_input", "depth needs a version with a picture")
    meta = json.loads((Path(in_dir) / "meta.json").read_text(encoding="utf-8"))
    outs, key = meta["outputs"], params.get("key", "image")
    level = params.get("resolution_level", 9)
    if (
        not re.fullmatch(r"image(_\d+)?", str(key))
        or not isinstance(level, int)
        or not 0 <= level <= 9
    ):
        raise StepError(
            "bad_params", "depth: key a picture (image, image_<k>); resolution_level 0-9"
        )
    if key not in outs:
        raise StepError("missing_input", f"{meta['version']} has no picture {key!r}")
    import torch
    from huggingface_hub import hf_hub_download
    from huggingface_hub.errors import LocalEntryNotFoundError

    try:
        path = hf_hub_download(MODEL, "model.pt", revision=REVISION, local_files_only=True)
    except LocalEntryNotFoundError:
        raise StepError(
            "missing_input", "the depth model is not installed: Setup, Parallax layers"
        ) from None
    import flex_gemm
    from moge.model.v3 import MoGeModel

    # a one-shot process would tune again on every run (MoGe's own app turns it off too)
    flex_gemm.config.AUTOTUNE_MODE = "never"
    if torch.version.hip:
        _moge_without_triton()
    t0, out = time.time(), Path(out_dir)
    picture = Image.open(Path(in_dir) / outs[key]).convert("RGB")
    progress(0.1, "loading MoGe-3")
    model = MoGeModel.from_pretrained(path).cuda().eval()
    torch.cuda.reset_peak_memory_stats()
    progress(0.4, "depth")
    x = torch.from_numpy(np.asarray(picture).copy()).permute(2, 0, 1).float() / 255
    with torch.inference_mode():
        depth = model.infer(x.cuda(), resolution_level=level, use_fp16=True)["depth"]
    peak = torch.cuda.max_memory_allocated() / 2**30
    disp, valid = disparity(depth.float().cpu().numpy())
    del model
    torch.cuda.empty_cache()
    np.savez_compressed(out / "depth.npz", disparity=disp.astype(np.float16), valid=valid)
    Image.fromarray((disp * 255).round().astype(np.uint8)).save(out / "preview.png")
    picture.save(out / "image.png")
    return {
        "outputs": {"depth": "depth.npz", "preview": "preview.png", "image": "image.png"},
        "metrics": {
            "seconds": round(time.time() - t0, 1),
            "peak_vram_gb": round(peak, 2),
            "no_depth": round(float(1 - valid.mean()), 3),  # the sky's share, mostly
            "width": picture.width,
            "height": picture.height,
        },
        "models": [{"id": MODEL, "revision": REVISION}],
        "licenses": [MODEL, "moge"],
    }


if __name__ == "__main__":
    main(run)
