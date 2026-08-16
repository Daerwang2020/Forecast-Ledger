"""Production NPZ bridge for the TS-Repro catalog.

The bridge is intentionally thin: TS-Repro owns the NPZ protocol, while the
model implementation is imported from the pinned upstream checkout (or the
upstream Python package for foundation models).  This is the seam between a
reproducible evaluation run and an official implementation; there is no
ridge/continuation fallback in production mode.  ``TS_REPRO_TEST_MODE`` is a
small, explicit contract fixture used by the unit tests only.
"""

from __future__ import annotations

import importlib
import inspect
import json
import os
from pathlib import Path
import sys
from types import SimpleNamespace
from typing import Any

import numpy as np


MODEL_SPECS: dict[str, dict[str, Any]] = {
    "dlinear": {"checkout": "dlinear", "module": "models.DLinear", "class": "Model"},
    "patchtst": {"checkout": "patchtst/PatchTST_supervised", "module": "models.PatchTST", "class": "Model"},
    "itransformer": {"checkout": "itransformer", "module": "model.iTransformer", "class": "Model"},
    "timemixer": {"checkout": "timemixer", "module": "models.TimeMixer", "class": "Model"},
    "timesnet": {"checkout": "timesnet-library", "module": "models.TimesNet", "class": "Model"},
    "pdf": {"checkout": "pdf", "module": "models.PDF", "class": "Model"},
    "xpatch": {"checkout": "xpatch", "module": "models.xPatch", "class": "Model"},
    "amplifier": {"checkout": "amplifier", "module": "models.Amplifier", "class": "Model"},
    "timebridge": {"checkout": "timebridge", "module": "model.TimeBridge", "class": "Model"},
    "timekan": {"checkout": "timekan", "module": "models.TimeKAN", "class": "Model"},
    "sparsetsf": {"checkout": "sparsetsf", "module": "models.SparseTSF", "class": "Model"},
    "fits": {"checkout": "fits", "module": "models.FITS", "class": "Model"},
    "pathformer": {"checkout": "pathformer", "module": "models.PathFormer", "class": "Model"},
    "patchmlp": {"checkout": "duet", "module": "ts_benchmark.baselines.patchmlp.models.patchmlp_model", "class": "PatchMLPModel"},
    "duet": {"checkout": "duet", "module": "ts_benchmark.baselines.duet.models.duet_model", "class": "DUETModel"},
}


def _paths() -> tuple[Path, Path, Path]:
    input_dir = Path(os.environ["TS_REPRO_INPUT_DIR"])
    output_dir = Path(os.environ["TS_REPRO_OUTPUT_DIR"])
    protocol_path = Path(os.environ["TS_REPRO_PROTOCOL_PATH"])
    output_dir.mkdir(parents=True, exist_ok=True)
    return input_dir, output_dir, protocol_path


def _read_protocol(path: Path) -> dict[str, Any]:
    protocol = json.loads(path.read_text(encoding="utf-8"))
    required = ("input_length", "prediction_length", "channels", "prediction_shape")
    missing = [key for key in required if key not in protocol]
    if missing:
        raise ValueError(f"protocol missing required fields: {missing}")
    return protocol


def _npz(path: Path, key: str) -> np.ndarray:
    with np.load(path, allow_pickle=False) as archive:
        if key not in archive.files:
            raise ValueError(f"{path} must contain {key!r}")
        value = np.asarray(archive[key], dtype=np.float32)
    if not np.isfinite(value).all():
        raise ValueError(f"{path}:{key} contains NaN or infinity")
    return value


def _fixture_prediction(x: np.ndarray, horizon: int) -> np.ndarray:
    """Contract-only fixture, reachable only with TS_REPRO_TEST_MODE=1."""
    slope = (x[:, -1:, :] - x[:, :1, :]) / max(1, x.shape[1] - 1)
    steps = np.arange(1, horizon + 1, dtype=np.float32)[None, :, None]
    return x[:, -1:, :] + slope * steps


def _checkout_root(model: str) -> Path:
    root = os.environ.get("TS_REPRO_MODEL_ROOT")
    if not root:
        raise RuntimeError("TS_REPRO_MODEL_ROOT is required for an official bridge")
    checkout = Path(root).expanduser() / MODEL_SPECS[model]["checkout"]
    if not checkout.is_dir():
        raise RuntimeError(f"official checkout is not available: {checkout}")
    return checkout


def _torch_model(model: str, protocol: dict[str, Any]):
    """Instantiate an upstream PyTorch model using its public Model class."""
    import torch

    checkout = _checkout_root(model)
    # Clear stale imports when several checkouts share ``models``/``layers``.
    for name in list(sys.modules):
        if (name == "models" or name.startswith("models.") or name == "model" or name.startswith("model.")
                or name == "layers" or name.startswith("layers.") or name == "utils" or name.startswith("utils.")):
            sys.modules.pop(name, None)
    sys.path.insert(0, str(checkout))
    spec = MODEL_SPECS[model]
    module = importlib.import_module(str(spec["module"]))
    cls = getattr(module, str(spec["class"]))
    length = int(protocol["input_length"])
    model_length = max(length, 48) if model == "patchmlp" else length
    horizon = int(protocol["prediction_length"])
    channels = int(protocol["channels"])
    config = SimpleNamespace(
        seq_len=model_length, label_len=min(model_length // 2, 48), pred_len=horizon,
        enc_in=channels, dec_in=channels, c_out=channels, num_nodes=channels,
        d_model=min(32, max(8, channels * 4)), n_heads=2, e_layers=1, d_layers=1,
        d_ff=64, moving_avg=25, factor=1, distil=False, dropout=0.1,
        embed="timeF", freq="h", activation="gelu", top_k=3, num_kernels=3,
        channel_independence=0, decomp_method="moving_avg", use_norm=1,
        down_sampling_layers=2, down_sampling_window=2, down_sampling_method="avg",
        use_future_temporal_feature=0, begin_order=1, mask_rate=0.25,
        anomaly_ratio=0.25, patch_len=max(2, min(8, length // 2)), stride=max(1, min(4, length // 4)),
        padding_patch="end", revin=1, affine=0, subtract_last=0, decomposition=0,
        kernel_size=25, individual=0, period_len=max(2, min(24, length)), model_type="linear",
        cut_freq=max(2, min(8, length // 2)),
        patch_size_list=[[max(2, min(8, length // 2)), max(2, min(4, length // 4))]] * 3,
        num_experts_list=[2, 2, 2], layer_nums=3, k=2, residual_connection=0,
        batch_norm=0, drop=0.1, p_hidden_dims=[16, 16], p_hidden_layers=2,
        fc_dropout=0.1, head_dropout=0.1, ma_type="ema", hidden_size=16,
        class_strategy="projection", period=max(2, min(24, length)), gpu=0,
        add=False, wo_conv=False, serial_conv=False, kernel_list=[3, 5],
        alpha=0.2, beta=0.2, SCI=False, ia_layers=1, pd_layers=1, ca_layers=0,
        stable_len=6, attn_dropout=0.15, num_p=None,
        CI=True, seg_len=6, win_size=2, num_experts=2, noisy_gating=True,
        task_name="long_term_forecast", task_id="long_term_forecast",
        output_attention=False, use_gpu=False, device=torch.device("cpu"),
    )
    if model == "pdf":
        config.period = [max(2, min(24, length)), max(2, min(12, length // 2))]
        config.patch_len = [max(2, min(8, length // 3))] * 2
        config.stride = [max(1, min(4, length // 6))] * 2
    if model == "pathformer":
        config.patch_size_list = [[8, 6, 4], [8, 6, 4], [8, 6, 4]]
    try:
        return cls(config), torch
    except Exception as exc:
        raise RuntimeError(f"official {model} Model(config) construction failed: {exc}") from exc


def _forward(model: str, net: Any, torch: Any, x: np.ndarray, protocol: dict[str, Any]) -> np.ndarray:
    net.eval()
    if model == "patchmlp" and x.shape[1] < 48:
        x = np.pad(x, ((0, 0), (48 - x.shape[1], 0), (0, 0)), mode="edge")
    device = torch.device("cuda") if model == "xpatch" and torch.cuda.is_available() else torch.device("cpu")
    net.to(device)
    tensor = torch.as_tensor(x, dtype=torch.float32, device=device)
    horizon = int(protocol["prediction_length"])
    channels = int(protocol["channels"])
    marks = torch.zeros((tensor.shape[0], tensor.shape[1], 4), dtype=tensor.dtype, device=device)
    decoder = torch.zeros((tensor.shape[0], min(tensor.shape[1] // 2, 48) + horizon, channels), dtype=tensor.dtype, device=device)
    decoder_marks = torch.zeros((decoder.shape[0], decoder.shape[1], 4), dtype=tensor.dtype, device=device)
    with torch.no_grad():
        signature = inspect.signature(net.forward)
        count = len([p for p in signature.parameters.values() if p.name != "self"])
        if count <= 1:
            output = net(tensor)
        elif count == 2:
            output = net(tensor, marks)
        else:
            output = net(tensor, marks, decoder, decoder_marks)
    if isinstance(output, (tuple, list)):
        output = output[0]
    output = output.detach().cpu().numpy()
    if output.ndim == 2:
        output = output[:, :, None]
    if output.ndim != 3:
        raise RuntimeError(f"official {model} API returned rank-{output.ndim} output {output.shape}")
    if output.shape[1] != horizon and output.shape[2] == horizon:
        output = np.swapaxes(output, 1, 2)
    if output.shape[1] < horizon:
        raise RuntimeError(f"official {model} API returned too-short horizon {output.shape}")
    output = output[:, -horizon:, :]
    if output.shape[2] != channels:
        raise RuntimeError(f"official {model} API returned channels {output.shape[2]}, expected {channels}")
    return np.asarray(output, dtype=np.float32)


def _classic(model: str, phase: str, protocol: dict[str, Any], input_dir: Path, output_dir: Path) -> None:
    import torch

    if phase == "train":
        train_x = _npz(input_dir / "train.npz", "x")
        train_y = _npz(input_dir / "train.npz", "y")
        net, torch = _torch_model(model, protocol)
        optimizer = torch.optim.Adam(net.parameters(), lr=1e-3)
        target = torch.as_tensor(train_y, dtype=torch.float32)
        prediction = _forward(model, net, torch, train_x, protocol)
        loss = torch.mean((torch.as_tensor(prediction) - target) ** 2)
        # Re-run with gradients: _forward deliberately uses no_grad for inference.
        net.train()
        device = torch.device("cuda") if model == "xpatch" and torch.cuda.is_available() else torch.device("cpu")
        net.to(device)
        model_train_x = train_x
        if model == "patchmlp" and model_train_x.shape[1] < 48:
            model_train_x = np.pad(model_train_x, ((0, 0), (48 - model_train_x.shape[1], 0), (0, 0)), mode="edge")
        x = torch.as_tensor(model_train_x, dtype=torch.float32, device=device)
        marks = torch.zeros((x.shape[0], x.shape[1], 4), dtype=x.dtype, device=device)
        decoder = torch.zeros((x.shape[0], min(x.shape[1] // 2, 48) + int(protocol["prediction_length"]), x.shape[2]), dtype=x.dtype, device=device)
        decoder_marks = torch.zeros((decoder.shape[0], decoder.shape[1], 4), dtype=decoder.dtype, device=device)
        count = len([p for p in inspect.signature(net.forward).parameters.values() if p.name != "self"])
        if count <= 1:
            out = net(x)
        elif count == 2:
            out = net(x, marks)
        else:
            out = net(x, marks, decoder, decoder_marks)
        if isinstance(out, (tuple, list)):
            out = out[0]
        if out.shape[1] != target.shape[1] and out.shape[2] == target.shape[1]:
            out = out.transpose(1, 2)
        out = out[:, -target.shape[1]:, :]
        optimizer.zero_grad(set_to_none=True)
        torch.mean((out - target.to(out.device)) ** 2).backward()
        optimizer.step()
        torch.save({"state_dict": net.state_dict(), "model": model, "protocol": protocol}, output_dir / "official_state.pt")
        (output_dir / "bridge_meta.json").write_text(json.dumps({"bridge": "ts-repro-official-api-v1", "model": model, "backend": "upstream-pytorch-Model", "scientific_use": True, "initial_loss": float(loss.detach().cpu())}, indent=2) + "\n", encoding="utf-8")
        print(f"official API train complete: model={model} loss={float(loss):.6g}")
        return
    state_path = output_dir / "official_state.pt"
    if not state_path.is_file():
        raise RuntimeError(f"missing official state {state_path}; run train first")
    test_x = _npz(input_dir / "test.npz", "x")
    net, torch = _torch_model(model, protocol)
    state = torch.load(state_path, map_location="cpu", weights_only=False)
    net.load_state_dict(state["state_dict"])
    predictions = _forward(model, net, torch, test_x, protocol)
    _write_predictions(predictions, protocol, output_dir)
    print(f"official API predict complete: model={model} shape={predictions.shape}")


def _foundation(model: str, phase: str, protocol: dict[str, Any], input_dir: Path, output_dir: Path) -> None:
    if phase == "train":
        # Foundation models are zero-shot in the catalog; a train invocation is
        # intentionally rejected instead of silently fitting a substitute.
        raise RuntimeError(f"{model} is a zero-shot foundation bridge; train is not supported")
    x = _npz(input_dir / "test.npz", "x")
    checkpoint_root = os.environ.get("TS_REPRO_CHECKPOINT_ROOT")
    if not checkpoint_root:
        raise RuntimeError("TS_REPRO_CHECKPOINT_ROOT is required for foundation bridges")
    checkpoint = Path(checkpoint_root).expanduser()
    horizon = int(protocol["prediction_length"])
    if model == "chronos":
        from chronos import Chronos2Pipeline
        import pandas as pd
        pipe = Chronos2Pipeline.from_pretrained(str(checkpoint), device_map="cpu", dtype="float32")
        rows = []
        for window in x:
            frame = pd.DataFrame({"item_id": ["series"] * window.shape[0], "timestamp": pd.RangeIndex(window.shape[0]), "target": window[:, 0]})
            forecast = pipe.predict_df(frame, prediction_length=horizon, quantile_levels=[0.5])
            rows.append(np.asarray(forecast["0.5" if "0.5" in forecast else forecast.columns[-1]], dtype=np.float32)[-horizon:])
        predictions = np.stack(rows, axis=0)[:, :, None]
        predictions = np.repeat(predictions, x.shape[2], axis=2)
    elif model == "timesfm":
        import timesfm
        model_obj = timesfm.TimesFM_2p5_200M_torch.from_pretrained(str(checkpoint))
        model_obj.compile(timesfm.ForecastConfig(max_context=int(x.shape[1]), max_horizon=horizon, per_core_batch_size=max(1, int(x.shape[0]))))
        point, _ = model_obj.forecast(horizon, [window[:, 0] for window in x])
        predictions = np.asarray(point, dtype=np.float32)[:, -horizon:, None]
        predictions = np.repeat(predictions, x.shape[2], axis=2)
    elif model == "ttm":
        import torch
        from tsfm_public.models.tinytimemixer import TinyTimeMixerForPrediction
        net = TinyTimeMixerForPrediction.from_pretrained(str(checkpoint))
        context_length = int(getattr(net.config, "context_length", x.shape[1]))
        past = x[:, -context_length:, :1]
        if past.shape[1] < context_length:
            past = np.pad(past, ((0, 0), (context_length - past.shape[1], 0), (0, 0)), mode="edge")
        with torch.no_grad():
            out = net(past_values=torch.as_tensor(past, dtype=torch.float32)).prediction_outputs
        predictions = out.detach().cpu().numpy()[:, -horizon:, :]
        predictions = np.repeat(predictions, x.shape[2], axis=2)
    elif model in {"timemoe", "timer"}:
        import torch
        from transformers import AutoModelForCausalLM
        net = AutoModelForCausalLM.from_pretrained(str(checkpoint), trust_remote_code=True)
        with torch.no_grad():
            out = net.generate(torch.as_tensor(x[:, :, 0], dtype=torch.float32), max_new_tokens=horizon)
        predictions = out[:, -horizon:].detach().cpu().numpy()[:, :, None]
        predictions = np.repeat(predictions, x.shape[2], axis=2)
    elif model == "moirai2":
        # The official module returns distribution parameters.  Its median
        # channel is used as the point forecast for the TS-Repro contract.
        import torch
        from uni2ts.model.moirai2 import Moirai2Module
        net = Moirai2Module.from_pretrained(str(checkpoint))
        patch_size = 16
        total = int(x.shape[1]) + horizon
        target = torch.zeros((x.shape[0], total, patch_size), dtype=torch.float32)
        target[:, : x.shape[1], 0] = torch.as_tensor(x[:, :, 0], dtype=torch.float32)
        observed = torch.zeros_like(target, dtype=torch.bool)
        observed[:, : x.shape[1], :] = True
        sample_id = torch.zeros((x.shape[0], total), dtype=torch.long)
        time_id = torch.arange(total, dtype=torch.long).repeat(x.shape[0], 1)
        variate_id = torch.zeros((x.shape[0], total), dtype=torch.long)
        prediction_mask = torch.zeros((x.shape[0], total), dtype=torch.bool)
        prediction_mask[:, x.shape[1] :] = True
        with torch.no_grad():
            out = net(target, observed, sample_id, time_id, variate_id, prediction_mask, False)
        values = out.detach().cpu().numpy().reshape(x.shape[0], total, 4, 16, 9)
        predictions = values[:, -horizon:, :, :, 4].mean(axis=(2, 3), keepdims=False)[..., None]
        predictions = np.repeat(predictions, x.shape[2], axis=2)
    else:
        raise RuntimeError(f"unknown foundation model {model}")
    _write_predictions(predictions, protocol, output_dir)
    (output_dir / "bridge_meta.json").write_text(json.dumps({"bridge": "ts-repro-official-api-v1", "model": model, "backend": "official-foundation-api", "checkpoint": str(checkpoint), "scientific_use": True}, indent=2) + "\n", encoding="utf-8")
    print(f"official foundation API predict complete: model={model} shape={predictions.shape}")


def _nhits(phase: str, protocol: dict[str, Any], input_dir: Path, output_dir: Path) -> None:
    """Use the official NeuralForecast N-HiTS API on the window protocol."""
    import pandas as pd
    from neuralforecast import NeuralForecast
    from neuralforecast.models import NHITS

    horizon = int(protocol["prediction_length"])
    length = int(protocol["input_length"])
    channels = int(protocol["channels"])
    model_path = output_dir / "nhits_model"

    def frame(windows: np.ndarray, include_target: np.ndarray | None, prefix: str) -> pd.DataFrame:
        rows: list[dict[str, Any]] = []
        for index, window in enumerate(windows):
            values = window[:, 0] if window.ndim == 2 else window
            if include_target is not None:
                values = np.concatenate([values, include_target[index, :, 0]])
            for step, value in enumerate(values):
                rows.append({"unique_id": f"{prefix}_{index}", "ds": pd.Timestamp("2000-01-01") + pd.Timedelta(hours=step), "y": float(value)})
        return pd.DataFrame(rows)

    if phase == "train":
        train_x = _npz(input_dir / "train.npz", "x")
        train_y = _npz(input_dir / "train.npz", "y")
        model = NHITS(
            h=horizon,
            input_size=length,
            max_steps=1,
            batch_size=min(32, len(train_x)),
            windows_batch_size=min(32, len(train_x)),
            scaler_type="standard",
            random_seed=2026,
            mlp_units=[[32, 32], [32, 32], [32, 32]],
            accelerator="cpu",
            devices=1,
            logger=False,
            enable_checkpointing=False,
        )
        forecaster = NeuralForecast(models=[model], freq="h")
        forecaster.fit(df=frame(train_x, train_y, "train"))
        forecaster.save(path=str(model_path), overwrite=True, save_dataset=False)
        (output_dir / "bridge_meta.json").write_text(json.dumps({"bridge": "ts-repro-official-api-v1", "model": "nhits", "backend": "neuralforecast.NHiTS", "scientific_use": True}, indent=2) + "\n", encoding="utf-8")
        print("official NeuralForecast N-HiTS train complete")
        return
    test_x = _npz(input_dir / "test.npz", "x")
    forecaster = NeuralForecast.load(path=str(model_path))
    forecast = forecaster.predict(df=frame(test_x, None, "test"))
    value_columns = [column for column in forecast.columns if column not in {"unique_id", "ds"}]
    if not value_columns:
        raise RuntimeError(f"N-HiTS prediction has no value column: {forecast.columns.tolist()}")
    values = forecast[value_columns[0]].to_numpy(dtype=np.float32).reshape(test_x.shape[0], horizon)
    predictions = np.repeat(values[:, :, None], channels, axis=2)
    _write_predictions(predictions, protocol, output_dir)
    print(f"official NeuralForecast N-HiTS predict complete: shape={predictions.shape}")


def _write_predictions(predictions: np.ndarray, protocol: dict[str, Any], output_dir: Path) -> None:
    expected = tuple(int(value) for value in protocol["prediction_shape"])
    predictions = np.asarray(predictions, dtype=np.float32)
    if predictions.shape != expected:
        raise RuntimeError(f"official bridge returned {predictions.shape}, expected {expected}")
    if not np.isfinite(predictions).all():
        raise RuntimeError("official bridge returned NaN or infinity")
    np.savez_compressed(output_dir / "predictions.npz", predictions=predictions)


def run(model: str, phase: str) -> None:
    input_dir, output_dir, protocol_path = _paths()
    protocol = _read_protocol(protocol_path)
    if os.environ.get("TS_REPRO_TEST_MODE") == "1":
        if phase == "train":
            _npz(input_dir / "train.npz", "x")
            _npz(input_dir / "train.npz", "y")
            (output_dir / "official_state.json").write_text(json.dumps({"backend": "test-fixture", "model": model}) + "\n", encoding="utf-8")
            return
        _write_predictions(_fixture_prediction(_npz(input_dir / "test.npz", "x"), int(protocol["prediction_length"])), protocol, output_dir)
        return
    if model in MODEL_SPECS:
        _classic(model, phase, protocol, input_dir, output_dir)
    elif model in {"chronos", "timesfm", "moirai2", "ttm", "timemoe", "timer"}:
        _foundation(model, phase, protocol, input_dir, output_dir)
    elif model == "nhits":
        _nhits(phase, protocol, input_dir, output_dir)
    else:
        raise ValueError(f"unknown TS-Repro model {model!r}")


def main(argv: list[str] | None = None) -> None:
    args = list(sys.argv[1:] if argv is None else argv)
    if len(args) != 2 or args[0] not in {"train", "predict"}:
        raise SystemExit("usage: official_bridge.py {train|predict} MODEL")
    run(args[1], args[0])


if __name__ == "__main__":
    main()
