"""ONNX export with a parity check and a CPU latency measurement (the phone-tier acceptance test)."""
from __future__ import annotations

import time
import warnings
from pathlib import Path

import numpy as np
import torch

from forecast.models.patchnet import PatchNet


def export_onnx(model: PatchNet, path: Path, n_tab: int) -> dict:
    import onnxruntime as ort

    model = model.cpu().float().eval()
    c, lb = model.cfg["n_channels"], model.lookback
    x, tab = torch.randn(2, c, lb), torch.randn(2, n_tab)
    path.parent.mkdir(parents=True, exist_ok=True)
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        torch.onnx.export(model, (x, tab), str(path), input_names=["x", "tab"], output_names=["quantiles"],
                          dynamic_axes={"x": {0: "batch"}, "tab": {0: "batch"}, "quantiles": {0: "batch"}},
                          opset_version=17, dynamo=False)
    so = ort.SessionOptions()
    so.intra_op_num_threads = 1
    sess = ort.InferenceSession(str(path), so, providers=["CPUExecutionProvider"])
    feed = {"x": x.numpy(), "tab": tab.numpy()}
    got = sess.run(None, feed)[0]
    with torch.no_grad():
        want = model(x, tab).numpy()
    one = {"x": x[:1].numpy(), "tab": tab[:1].numpy()}
    t0 = time.perf_counter()
    for _ in range(50):
        sess.run(None, one)
    return {"onnx_bytes": path.stat().st_size, "onnx_max_abs_diff": float(np.abs(got - want).max()),
            "cpu_ms_per_forecast": round((time.perf_counter() - t0) / 50 * 1000, 3)}
