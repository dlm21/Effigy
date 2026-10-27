"""AMD: steps/_rocm.py computes big products a block of rows at a time, with the same result."""

import sys

import pytest

torch = pytest.importorskip("torch")

from steps import _rocm


@pytest.fixture
def small_blocks(monkeypatch):
    """Blocks of 4 rows, on any device (the real check only fires on a ROCm GPU)."""
    monkeypatch.setattr(_rocm, "ROWS", 4)
    monkeypatch.setattr(
        _rocm,
        "_big",
        lambda t: isinstance(t, torch.Tensor) and t.dim() >= 2 and t.shape[:-1].numel() > 4,
    )


def test_linear_matmul_mm_addmm_match_unblocked(small_blocks):
    torch.manual_seed(0)
    x, x3 = torch.randn(11, 5), torch.randn(3, 7, 5)
    w, b, m = torch.randn(4, 5), torch.randn(4), torch.randn(5, 4)
    F = torch.nn.functional
    cases = [
        (F.linear, "linear", (x, w, b), {}),
        (F.linear, "linear", (x, w), {"bias": b}),
        (F.linear, "linear", (x3, w, b), {}),
        (torch.matmul, "matmul", (x, m), {}),
        (torch.matmul, "matmul", (x3, m), {}),
        (torch.mm, "mm", (x, m), {}),
        (torch.addmm, "addmm", (b, x, m), {}),
        (torch.addmm, "addmm", (torch.randn(11, 4), x, m), {}),
    ]
    for fn, kind, args, kwargs in cases:
        assert torch.allclose(
            _rocm._rows(fn, kind)(*args, **kwargs), fn(*args, **kwargs), atol=1e-6
        ), kind


def test_other_shapes_pass_through(small_blocks):
    calls = []

    def fake(*a, **k):
        calls.append(a)
        return "upstream"

    batched = _rocm._rows(fake, "matmul")
    # bmm-like: not one 2-D matrix
    assert batched(torch.randn(9, 3, 3), torch.randn(9, 3, 3)) == "upstream"
    assert batched(torch.randn(3, 3), torch.randn(3, 3)) == "upstream"  # small
    assert len(calls) == 2
    # keyword calls go through unsplit, they used to fail unpacking the positional ones
    x, m, b = torch.randn(11, 5), torch.randn(5, 4), torch.randn(4)
    F = torch.nn.functional
    for fn, kind, args, kwargs in [
        (torch.matmul, "matmul", (), {"input": x, "other": m}),
        (torch.Tensor.mm, "mm", (x,), {"mat2": m}),
        (torch.addmm, "addmm", (b,), {"mat1": x, "mat2": m}),
        (F.linear, "linear", (), {"input": x, "weight": m.t()}),
    ]:
        assert torch.allclose(_rocm._rows(fn, kind)(*args, **kwargs), fn(*args, **kwargs)), kind


def test_install_is_lazy_and_noop_without_hip():
    _rocm.install()  # torch is imported here already: patches now, or nothing on CUDA/CPU builds and on Linux
    want = bool(torch.version.hip) and sys.platform == "win32"
    assert getattr(torch.matmul, "effigy_rows", False) == want


def test_rows_only_on_windows(monkeypatch):
    """The wrong big GEMMs were measured on Windows ROCm; on Linux they were right and the wrapped torch.matmul broke
    torch.jit.script (Kimodo), so Linux keeps torch's own."""
    F = torch.nn.functional
    targets = [(F, "linear"), (torch, "matmul"), (torch, "mm"), (torch, "addmm")]
    targets += [(torch.Tensor, n) for n in ("__matmul__", "matmul", "mm")]
    for obj, name in targets:  # put back whatever _patch replaces
        monkeypatch.setattr(obj, name, getattr(obj, name))
    monkeypatch.setattr(torch.version, "hip", "7.2.53211")
    monkeypatch.setattr(_rocm.sys, "platform", "linux")
    _rocm._patch()
    assert not getattr(torch.matmul, "effigy_rows", False)
    monkeypatch.setattr(_rocm.sys, "platform", "win32")
    _rocm._patch()
    assert torch.matmul.effigy_rows and F.linear.effigy_rows
