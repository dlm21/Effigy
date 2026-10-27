"""Animate from a text prompt (IMPLEMENTATION.md M26, resident since M30): NVIDIA Kimodo, called as its
own CLI calls it.

Runs in .venv-kimodo, one-shot or resident (`--serve`): the loaded model (the Kimodo-SOMA-RP-v1.1
denoiser on the GPU, the LLM2Vec text encoder on the CPU as upstream advises under 16 GB VRAM, on the GPU too
when the budget holds it, M46a) stays in MODEL between requests, so the first prompt pays the load (about 35 s) and the next ones take seconds;
the server evicts it after its idle time like the other resident families. The motion is written as a
BVH on the SOMA 77-joint skeleton and Blender (blender_ops/bvh_clip) turns it into clip.glb, a clip
like the pack's with one animation named from the prompt, so the `anim` step can retarget it
(params.clips may name this version).
params: {"prompt": "A person waves with the right hand.", "seconds": 5, "seed": 0, "steps": 100,
"offload": true|false (the text encoder on the CPU; default: by the budget)}
Outputs: clip.glb, motion.bvh, prompt.json.
"""

import contextlib
import gc
import json
import os
import re
import shutil
import sys
import time
from pathlib import Path

from steps._blender import BlenderError, run_blender
from steps._common import StepError, main, offload, progress

MODEL_ID = "nvidia/Kimodo-SOMA-RP-v1.1"
ENCODER = "meta-llama/Meta-Llama-3-8B-Instruct"
ADAPTERS = (
    "McGill-NLP/LLM2Vec-Meta-Llama-3-8B-Instruct-mntp",
    "McGill-NLP/LLM2Vec-Meta-Llama-3-8B-Instruct-mntp-supervised",
)
MAX_SECONDS = 10.0  # Kimodo's limit per prompt
MODEL: dict = {}  # "model" → the loaded Kimodo model, "device" → its text encoder's; kept between --serve requests
# M46a: the budget that holds the text encoder on the GPU too: Kimodo's README, "requires ~17GB of VRAM to generate
# locally entirely on GPU" (claimed; until a user's gpu_report measures it). Below it the encoder runs on the CPU:
# a slower first prompt (30.7 s, m26) and 16 GB of host RAM.
ENCODER_GPU_GB = 18.0


def clip_name(prompt: str, limit: int = 48) -> str:
    """The animation's name from the prompt, as the viewer's menu and the exports show it: "A person
    waves with the right hand." → "Waves with the right hand"."""
    words = re.sub(r"[^\w\s'-]", " ", prompt).split()
    while words and words[0].lower() in (
        "a",
        "an",
        "the",
        "person",
        "someone",
        "man",
        "woman",
        "character",
        "he",
        "she",
        "they",
    ):
        words.pop(0)
    name = " ".join(words) or prompt.strip() or "motion"
    name = name[0].upper() + name[1:]
    return name if len(name) <= limit else name[:limit].rsplit(" ", 1)[0]


def _gated(e: Exception) -> bool:
    t = str(e).lower()
    return "gated" in t or "401" in t or "403" in t


def _model(device: str):
    """The Kimodo model with its text encoder on `device`, loaded once per process, again when a request wants the
    encoder elsewhere (prints of the library go to stderr: stdout is the step protocol)."""
    import torch

    if MODEL.get("device") != device:
        MODEL.clear()
        gc.collect()  # the old model's 16 GB gone before the new one loads (an in-place swap peaked at 21 GB, M45)
        torch.cuda.empty_cache()
    if "model" not in MODEL:
        from kimodo.model.load_model import load_model

        from tools.download_models import pin_refs

        pin_refs()  # an install from before pin_refs: Kimodo looks its repos up by id, offline (server/jobs.py)
        os.environ["TEXT_ENCODER_DEVICE"] = device  # read by Kimodo when the encoder loads
        progress(0.05, f"loading the motion model and its text encoder (on the {device.upper()})")
        try:
            with contextlib.redirect_stdout(sys.stderr):
                MODEL["model"] = load_model(
                    MODEL_ID.split("/")[1], device="cuda" if torch.cuda.is_available() else "cpu"
                )
        except Exception as e:
            if _gated(e):
                raise StepError(
                    "missing_input",
                    f"the text encoder {ENCODER} is gated: accept its licence on huggingface.co, then "
                    f"`hf auth login`\n{type(e).__name__}: {str(e)[:400]}",
                ) from e
            raise
        MODEL["device"] = device
    return MODEL["model"]


def encoder_device(params: dict, cuda: bool) -> str:
    """Where the text encoder runs: "offload" decides, else the budget (ENCODER_GPU_GB). A worker that loaded it on
    the CPU keeps it there unless asked: a budget hovering at the threshold must not reload it (30-60 s) each time,
    while one that loaded it on the GPU moves it off when the budget no longer holds it."""
    if not cuda:
        return "cpu"
    if params.get("offload") is None and MODEL.get("device") == "cpu":
        return "cpu"
    return "cpu" if offload(params, ENCODER_GPU_GB) else "cuda"


def generate(
    prompt: str, seconds: float, seed: int, steps: int, out: Path, params: dict | None = None
) -> Path:
    """The motion as out/motion.bvh (SOMA 77 joints, the standard T-pose as rest), the way
    kimodo_gen --bvh --bvh_standard_tpose writes it."""
    import torch
    from kimodo.exports.bvh import save_motion_bvh
    from kimodo.skeleton import SOMASkeleton30, global_rots_to_local_rots
    from kimodo.tools import seed_everything

    model = _model(encoder_device(params or {}, torch.cuda.is_available()))
    seed_everything(seed)
    frames = round(seconds * model.fps)
    with contextlib.redirect_stdout(sys.stderr):
        output = model(
            [prompt],
            [frames],
            num_denoising_steps=steps,
            num_samples=1,
            multi_prompt=True,
            post_processing=True,
            return_numpy=True,
        )
    device = "cuda" if torch.cuda.is_available() else "cpu"
    skeleton = model.skeleton
    if isinstance(skeleton, SOMASkeleton30):  # the model's output is already on somaskel77
        skeleton = skeleton.somaskel77.to(device)
    joints_pos = torch.from_numpy(output["posed_joints"][0]).to(device)
    joints_rot = torch.from_numpy(output["global_rot_mats"][0]).to(device)
    local = global_rots_to_local_rots(joints_rot, skeleton)
    bvh = out / "motion.bvh"
    save_motion_bvh(
        bvh,
        local,
        joints_pos[:, skeleton.root_idx, :],
        skeleton=skeleton,
        fps=model.fps,
        standard_tpose=True,
    )
    return bvh


def run(op, in_dir, out_dir, params):
    out = Path(out_dir)
    prompt = str(params.get("prompt", "")).strip()
    try:
        seconds = float(params.get("seconds", 5.0))
        seed = int(params.get("seed", 0))
        steps = int(params.get("steps", 100))
    except (TypeError, ValueError):
        raise StepError("bad_params", f"bad motion params: {params}") from None
    if not prompt or not 0.5 <= seconds <= MAX_SECONDS or not 1 <= steps <= 1000:
        raise StepError(
            "bad_params",
            f"bad motion params: {params} (prompt required, 0.5–{MAX_SECONDS:g} s, 1–1000 steps)",
        )
    progress(0.1, "encoding the prompt and sampling the motion")
    t = time.time()
    bvh = generate(prompt, seconds, seed, steps, out, params)
    gen_s = time.time() - t
    progress(0.8, "the clip as a GLB")
    name = clip_name(prompt)
    taken = set()
    for d in out.parent.glob("*_motion"):  # the asset's other motions: one name each
        if d != out and (d / "prompt.json").is_file():
            taken.add(json.loads((d / "prompt.json").read_text(encoding="utf-8")).get("name"))
    if name in taken:
        name = f"{name} {out.name[:3]}"
    try:
        res = run_blender("bvh_clip", [bvh], out / "_clip", {"name": name, "fps": 30}, timeout=300)
    except BlenderError as e:
        raise StepError("failed", str(e)) from e
    shutil.move(str(out / "_clip" / "clip.glb"), str(out / "clip.glb"))
    shutil.rmtree(out / "_clip", ignore_errors=True)
    (out / "prompt.json").write_text(
        json.dumps(
            {"prompt": prompt, "name": name, "seconds": seconds, "seed": seed, "steps": steps},
            indent=1,
        ),
        encoding="utf-8",
    )
    metrics = {
        "prompt": prompt,
        "name": name,
        "seconds": res["seconds"],
        "frames": res["frames"],
        "fps": res["fps"],
        "bones": res["bones"],
        "generate_seconds": round(gen_s, 1),
        "model_was_loaded": "model" in MODEL and gen_s < 20,  # a resident worker's second prompt
        "encoder_device": MODEL.get("device", "cpu"),
        "seed": seed,
        "steps": steps,
    }
    return {
        "outputs": {"clip": "clip.glb", "bvh": "motion.bvh", "prompt": "prompt.json"},
        "metrics": metrics,
        "models": [{"id": MODEL_ID}, {"id": ENCODER}, *({"id": a} for a in ADAPTERS)],
        "licenses": [MODEL_ID, ENCODER, *ADAPTERS, "kimodo"],
    }


if __name__ == "__main__":
    main(run)
