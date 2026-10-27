"""M55 parallax layers (research 15 §5.5): the depth's disparity, the bands cut in the depth's gaps, the scroll speeds,
the Godot Parallax2D and Paper2D exports, validate's items, and the layers step on a stand-in Z-Image. CPU; the real
models' run is in docs/bench/m55.md."""

import json
import sys
from types import SimpleNamespace

import numpy as np
import pytest
from PIL import Image

from steps import depth, export, layers, validate
from steps._common import StepError


def street(h=120, w=320):
    """A synthetic picture's disparity: sky on top (no depth), far buildings, a middle row, the near street, and one
    speck of the near band inside the far buildings."""
    disp = np.zeros((h, w), np.float32)
    valid = np.ones((h, w), bool)
    valid[:20] = False
    disp[20:60] = 0.15
    disp[60:90] = 0.45
    disp[90:] = 0.95
    disp[30:32, 100:102] = 0.95  # 4 px: a speck
    return disp + np.random.default_rng(0).normal(0, 0.01, disp.shape).astype(
        np.float32
    ) * valid, valid


def test_disparity_puts_the_sky_farthest():
    d = np.array([[np.inf, 2.0], [1.0, -1.0]])
    disp, valid = depth.disparity(d)
    assert valid.tolist() == [[False, True], [True, False]]
    assert disp[1, 0] == 1 and disp[0, 1] == 0.5 and disp[0, 0] == disp[1, 1] == 0


def test_bands_follow_the_depths_gaps_and_absorb_specks():
    disp, valid = street()
    labels = layers.bands(disp, valid, 4)
    assert labels[:20].max() == 0  # the sky
    assert (labels[25:58] == 1).all()  # the far buildings, the speck taken in
    assert (labels[62:88] == 2).all() and (labels[92:] == 3).all()
    scales = layers.scroll_scales(disp, labels)
    assert scales[0] == 0.05 and scales[-1] == 1 and scales == sorted(scales)
    two = layers.bands(disp, valid, 2)  # the sky and everything else
    assert two.max() == 1 and two[:20].max() == 0 and two[20:].min() == 1


def test_an_object_leaning_across_a_cut_stays_whole(monkeypatch):
    """A car parked along the street: its roof farther than its bumper, across the cut between the far buildings and
    the near road it stands on. Cut by depth alone it falls in two (the owner's street, 2026-10-13)."""
    h, w = 120, 200
    disp = np.full((h, w), 0.1, np.float32)  # the far buildings, a sharp edge round the car
    disp[60:] = np.linspace(0.45, 0.9, 60, dtype=np.float32)[:, None]  # the near road, receding
    disp[30:70, 80:120] = np.linspace(0.33, 0.47, 40, dtype=np.float32)[
        :, None
    ]  # the car, on the road
    labels = layers.bands(disp, np.ones((h, w), bool), 2)
    car = labels[
        32:68, 82:118
    ]  # its inside: the median filter gives the far band its outermost pixel or two
    assert (
        car.min() == car.max() == labels[115, 5] == 1 and labels[5, 5] == 0
    )  # whole, with the road it stands on
    monkeypatch.setattr(layers, "MARGIN", 0.0)  # depth alone
    assert len(np.unique(layers.bands(disp, np.ones((h, w), bool), 2)[32:68, 82:118])) == 2


def test_bands_of_no_depth_and_of_one_depth():
    h, w = 40, 60
    assert (
        layers.bands(np.zeros((h, w), np.float32), np.zeros((h, w), bool), 4).max() == 0
    )  # all sky
    flat = layers.bands(np.full((h, w), 0.5, np.float32), np.ones((h, w), bool), 4)
    assert flat.max() == 0  # one depth: one band, the empty ones dropped


def test_bad_layer_params_are_refused():
    for bad in ({"count": 1}, {"count": 6}, {"subject": "yes"}, {"strength": 0.1}, {"steps": 0}):
        with pytest.raises(StepError) as e:
            layers._params(bad)
        assert e.value.code == "bad_params", bad


def layers_version(root, name="030_layers", far_gap=False):
    d = root / name
    d.mkdir(parents=True)
    data = []
    for k, s in enumerate((0.05, 0.4, 1.0)):
        a = np.zeros((40, 64, 4), np.uint8)
        a[..., :3] = 60 * k
        a[k * 10 :, :, 3] = 255  # each nearer layer holds less
        if k == 0 and far_gap:
            a[0, 0, 3] = 0
        Image.fromarray(a).save(d / f"layer_{k}.png")
        data.append({"file": f"layer_{k}.png", "scroll_scale": s, "share": 0.3})
    (d / "layers.json").write_text(json.dumps(data))
    outs = {f"layer_{k}": f"layer_{k}.png" for k in range(3)} | {"layers": "layers.json"}
    meta = {"version": name, "asset_id": "street", "licenses": [], "outputs": outs}
    (d / "meta.json").write_text(json.dumps(meta))
    return d


def test_layers_export_as_a_godot_parallax_scene_and_paper2d_pictures(tmp_path):
    src = layers_version(tmp_path)
    out = tmp_path / "out"
    out.mkdir()
    r = export.run("run", str(src), str(out), {"profile": "godot-2d"})
    files = ["layer_0.png", "layer_1.png", "layer_2.png", "parallax.tscn", "README.txt"]
    assert sorted(r["outputs"]) == sorted([*files, "CREDITS.txt", "LICENSES.json"])
    scene = (out / "godot-2d" / "parallax.tscn").read_text()
    assert scene.count('type="Parallax2D"') == 3 and scene.count('type="Sprite2D"') == 3
    assert "scroll_scale = Vector2(0.05, 1)" in scene and "scroll_scale = Vector2(1, 1)" in scene
    assert (
        'path="layer_2.png" id="3_layer"' in scene and 'texture = ExtResource("3_layer")' in scene
    )
    assert "0.05, 0.4, 1" in (out / "godot-2d" / "README.txt").read_text()
    out2 = tmp_path / "out2"
    out2.mkdir()
    r = export.run("run", str(src), str(out2), {"profile": "unreal-2d"})
    assert "parallax.tscn" not in r["outputs"] and "layer_0.png" in r["outputs"]


def test_validate_checks_the_layers(tmp_path):
    for far_gap, status in ((False, "pass"), (True, "warn")):
        src = layers_version(tmp_path / str(far_gap), far_gap=far_gap)
        out = tmp_path / f"v{far_gap}"
        out.mkdir()
        validate.run("run", str(src), str(out), {})
        items = json.loads((out / "checklist.json").read_text())["items"]
        assert items["layers"] == {"status": "pass", "value": 3}
        assert items["far_layer_whole"]["status"] == status


def test_the_layers_step_draws_what_the_nearer_bands_hide(tmp_path, monkeypatch):
    pytest.importorskip("torch")
    disp, valid = street()
    src = tmp_path / "020_depth"
    src.mkdir()
    np.savez(src / "depth.npz", disparity=disp.astype(np.float16), valid=valid)
    rgb = np.zeros((*disp.shape, 3), np.uint8)
    rgb[..., 0] = (disp * 255).astype(np.uint8)
    Image.fromarray(rgb).save(src / "image.png")
    (tmp_path / "asset.json").write_text(
        json.dumps({"brief": {"prompt": "a street", "prompt_full": "a street, game"}})
    )

    class Part:
        def to(self, *_):
            return self

    calls = []

    class Pipe:
        transformer, text_encoder, vae = Part(), Part(), Part()

        def to(self, *_):
            return self

        def encode_prompt(self, texts, device, do_classifier_free_guidance):
            calls.append(("prompt", texts[0]))
            return [object()], None

        def __call__(self, image, mask_image, height, width, strength, **kw):
            calls.append(("draw", (width, height), strength))
            return SimpleNamespace(images=[Image.new("RGB", (width, height), (0, 200, 0))])

    monkeypatch.setattr(layers, "_pipeline", lambda size: Pipe())
    monkeypatch.setattr(layers, "progress", lambda *a, **k: None)
    out = tmp_path / "030_layers"
    out.mkdir()
    res = layers.run("run", str(src), str(out), {"count": 4, "seed": 1})
    assert calls[0] == (
        "prompt",
        "a street, game" + layers.EMPTY,
    )  # the kind's look, the ground empty
    assert [c[1:] for c in calls[1:]] == [
        ((320, 256), 0.8)
    ] * 3  # three layers drawn in, on Z-Image's 64 px grid, at least 256 a side
    data = json.loads((out / "layers.json").read_text())
    assert [d["file"] for d in data] == [f"layer_{k}.png" for k in range(4)]
    far = np.asarray(Image.open(out / "layer_0.png"))
    near = np.asarray(Image.open(out / "layer_3.png"))
    assert far[..., 3].min() == 255  # the farthest covers everything
    assert far[100, 10, 1] == 200  # behind the street: drawn in
    assert near[10, 10, 3] == 0 and near[110, 10, 3] == 255  # the nearest: only its band
    assert res["metrics"]["count"] == 4 and res["metrics"]["scroll_scales"][-1] == 1


def test_depth_on_amd_pools_and_upsamples_without_triton(monkeypatch):
    """AMD: MoGe-3's refiner without FlexGEMM's Triton kernels (one hung an RX 7900 XTX): the pool averages the
    voxels in each coarse one, the upsample gives each fine voxel its parent's value, a Triton launch fails."""
    torch = pytest.importorskip("torch")

    class Layer:
        def __init__(self):
            self.factor, self.linear = 2, lambda x: x

    fsb = SimpleNamespace(
        PoolDown=type("PoolDown", (Layer,), {}),
        NearestUp=type("NearestUp", (Layer,), {}),
        _ceil_downsample_shape=lambda shape, dim, f: torch.Size(
            [1, *((s + f - 1) // f for s in shape[1:4]), shape[4]]
        ),
        _with_channels=lambda shape, c: torch.Size([*shape[:-1], c]),
    )
    jit = SimpleNamespace(KernelInterface=type("KernelInterface", (), {}))
    for name, mod in {
        "flex_gemm": SimpleNamespace(config=SimpleNamespace()),
        "triton": SimpleNamespace(runtime=SimpleNamespace(jit=jit)),
        "triton.runtime": SimpleNamespace(jit=jit),
        "triton.runtime.jit": jit,
        "moge.model.modules": SimpleNamespace(flex_sparse_blocks=fsb),
        "moge.model.modules.flex_sparse_blocks": fsb,
    }.items():
        monkeypatch.setitem(sys.modules, name, mod)
    depth._moge_without_triton()
    coords = torch.tensor(
        [[0, 0, 0, 0], [0, 1, 1, 1], [0, 2, 0, 0], [0, 3, 1, 0]], dtype=torch.int32
    )
    feats = torch.tensor([[1.0], [3.0], [5.0], [7.0]])
    pooled, coarse, shape, cache = fsb.PoolDown().forward(
        feats, coords, torch.Size([1, 4, 2, 2, 1])
    )
    assert coarse.tolist() == [[0, 0, 0, 0], [0, 1, 0, 0]] and pooled.flatten().tolist() == [
        2.0,
        6.0,
    ]
    assert shape == torch.Size([1, 2, 1, 1, 1]) and coarse.dtype == torch.int32
    up, at, _ = fsb.NearestUp().forward(
        pooled, coarse, shape, coords, torch.Size([1, 4, 2, 2, 1]), cache.T
    )
    assert up.flatten().tolist() == [2.0, 2.0, 6.0, 6.0] and at is coords
    with pytest.raises(StepError) as e:
        jit.KernelInterface()[1]
    assert e.value.code == "failed" and "Triton" in str(e.value)
