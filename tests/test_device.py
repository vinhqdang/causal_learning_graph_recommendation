"""Device switch of the GPU-capable experiment scripts (experiments/run_din.py and experiments/run_dax.py,
--device cpu | cuda).

(a) --device cpu, and the default, reproduce the stored smoke outputs of tests/data/device_smoke byte for byte. Those
    were written by the code before the switch existed, so the switch changes no CPU computation. (The floats depend
    on the CPU's kernels; a machine with other kernels may differ in the last bits.) Skipped without the KuaiRand-27K
    inputs.
(b) --device cuda stops with a clear message when no CUDA device is available, before any data is read.
(c) a non-CPU device is refused for the losses that have no device path (drup.learned.train).
(d) with a CUDA device, the same smoke settings run on cuda and give the same keys and finite values.
"""

import json
import os
import subprocess
import sys

import numpy as np
import pytest
import torch

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)
from drup.learned import LightGCN, train  # noqa: E402

GOLD = os.path.join(ROOT, "tests", "data", "device_smoke")
RAW = os.path.join(ROOT, "data", "raw")
INPUTS = [os.path.join(RAW, "seq_27k.npz"), os.path.join(RAW, "lifelong_27k.npz"),
          os.path.join(RAW, "KuaiRand-Pure", "data", "video_features_basic_pure.csv")]
SMALL = ["--n_users", "60", "--epochs", "2", "--lrs", "1e-2", "--wds", "1e-4", "--dax_grid", "small",
         "--gammas", "0.5", "1.0", "--seeds", "2", "--threads", "2"]
SMOKE = {
    "din": ("run_din.py", ["--arch", "din", "--dataset", "kuairand27k", "--seq", "data/raw/seq_27k.npz"] + SMALL),
    "leaf": ("run_dax.py", ["--backbone", "leaf_lgn", "--mode", "posthoc", "--w", "0.5", "--dim", "8"] + SMALL),
    "qp": ("run_dax.py", ["--backbone", "qp_posthoc", "--qp_source", "grid"] + SMALL),
}


def _run(name, out, extra):
    script, args = SMOKE[name]
    cmd = [sys.executable, os.path.join(ROOT, "experiments", script)] + args + extra + ["--out", str(out)]
    return subprocess.run(cmd, cwd=ROOT, capture_output=True, text=True, timeout=900)


def _need_inputs():
    missing = [p for p in INPUTS if not os.path.exists(p)]
    if missing:
        pytest.skip(f"smoke inputs missing: {missing[0]}")


@pytest.mark.parametrize("name", sorted(SMOKE))
def test_cpu_device_reproduces_stored_smoke_output(tmp_path, name):
    _need_inputs()
    out = tmp_path / f"{name}.json"
    r = _run(name, out, ["--device", "cpu"])
    assert r.returncode == 0, r.stderr[-2000:]
    with open(os.path.join(GOLD, f"{name}_smoke.json"), "rb") as f:
        assert out.read_bytes() == f.read()


def test_default_device_is_the_cpu(tmp_path):
    _need_inputs()
    out = tmp_path / "din.json"
    r = _run("din", out, [])
    assert r.returncode == 0, r.stderr[-2000:]
    with open(os.path.join(GOLD, "din_smoke.json"), "rb") as f:
        assert out.read_bytes() == f.read()


@pytest.mark.skipif(torch.cuda.is_available(), reason="a CUDA device is present")
@pytest.mark.parametrize("name", sorted(SMOKE))
def test_cuda_without_a_device_fails_before_any_data_is_read(tmp_path, name):
    r = _run(name, tmp_path / "x.json", ["--device", "cuda"])
    assert r.returncode != 0
    assert "no CUDA device" in r.stderr
    assert not (tmp_path / "x.json").exists()


def test_non_cpu_device_is_refused_for_losses_without_a_device_path():
    torch.manual_seed(0)
    m, n = 6, 5
    O = torch.ones(m, n)
    Y = (torch.rand(m, n) < 0.3).float()
    P = torch.full((m, n), 0.5)
    model = LightGCN(m, n, 4, *torch.nonzero(Y > 0, as_tuple=True), torch.ones(int(Y.sum())), layers=1)
    with pytest.raises(NotImplementedError):
        train(model, O, Y, P, P, "naive", 1e-2, 1e-4, 1, 4, lambda mod, ep: True, device="cuda")


@pytest.mark.skipif(not torch.cuda.is_available(), reason="no CUDA device")
@pytest.mark.parametrize("name", sorted(SMOKE))
def test_cuda_smoke_has_the_same_keys_and_finite_values(tmp_path, name):
    _need_inputs()
    out = tmp_path / f"{name}.json"
    r = _run(name, out, ["--device", "cuda"])
    assert r.returncode == 0, r.stderr[-2000:]
    got = json.load(open(out))
    ref = json.load(open(os.path.join(GOLD, f"{name}_smoke.json")))
    assert got["args"] == ref["args"]
    assert set(got["results"]) == set(ref["results"])
    for nm, res in got["results"].items():
        assert set(res) == set(ref["results"][nm])
        assert [set(p) for p in res["per_user"]] == [set(p) for p in ref["results"][nm]["per_user"]]
        assert np.isfinite(res["test"]["ndcg@10"][0])
