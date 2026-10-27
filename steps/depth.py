"""Depth of a picture (M55, research 15 §5.5): Microsoft MoGe-3 ViT-g (MIT) in `.venv-moge`, for parallax layers.

Input: a version with a picture (`image`, or one of a set with `key: "image_2"`). Params: key; resolution_level:
0-9 (default 9, MoGe's finest). Outputs depth.npz {disparity: float16 0-1 (1 the nearest, 0 the sky and anything
MoGe finds no depth for), valid: bool}, preview.png (the disparity in grey), image.png (the picture, which `layers`
takes with it). About 11 s and 8.9 GB on the reference laptop (docs/bench/m55.md). A sky or any pixel MoGe marks
invalid counts as the farthest.
"""

import json
import re
import time
from pathlib import Path

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

    if torch.version.hip:
        # MoGe-3's sparse U-Net runs on FlexGEMM's Triton kernels (b2fadb2, no other path): on an RX 7900 XTX
        # (ROCm 7.2.4) one hung the GPU ("HW Exception ... GPU Hang") and froze the desktop it also drives
        raise StepError(
            "failed",
            "depth (MoGe-3) is not supported on AMD GPUs yet: its Triton kernels hang the card",
        )
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
