"""Compute cost of AEDL, the LSTM baseline, and TFT on the paper input shapes (CPU).

All models receive random inputs with the shapes of the paper windows (48 h
history, 24 h horizon), so no data are loaded. Reported per model:

* parameters,
* forward FLOPs per forecast origin (``torch.utils.flop_counter``),
* training memory: parameters, gradients, and AdamW states (16 bytes per
  parameter) plus the activations saved for backward for one batch,
* median training-step time (forward, backward, AdamW step) and epoch time,
* median inference latency for one origin.
"""

from __future__ import annotations

import argparse
import logging
import statistics
import sys
import time
import warnings
from pathlib import Path

import pandas as pd
import torch
from torch.utils.flop_counter import FlopCounterMode

ROOT = Path(__file__).resolve().parents[2]
for path in (ROOT / "src", ROOT / "src" / "scripts"):
    if str(path) not in sys.path:
        sys.path.insert(0, str(path))

from load_forecasting.models import build_model as build_aedl
from load_forecasting.pipeline import ExperimentConfig
from run_tft import BaselineConfig, build_model as build_tft, forward as tft_forward

PAST_LEN, PRED_LEN = 48, 24
N_PAST_FEATURES = 9  # load, temperature, GHI, six calendar features
TFT_FUTR = [f"f{i}" for i in range(8)]  # temperature, GHI, six calendar features
TRAIN_BATCH = 256
TRAIN_ORIGINS = 34033


def aedl_factory(future_dim: int, output_skip: str = "none", **overrides):
    cfg = ExperimentConfig(**overrides)

    def make():
        return build_aedl(
            "gaussian",
            n_past_features=N_PAST_FEATURES,
            future_dim=future_dim,
            hidden_size=cfg.hidden_size,
            num_layers=cfg.num_layers,
            attention_heads=cfg.attn_heads,
            dropout=cfg.dropout,
            pred_len=PRED_LEN,
            quantiles=cfg.quantiles,
            output_skip=output_skip,
            architecture=cfg.architecture,
        )

    def inputs(batch: int):
        return torch.randn(batch, PAST_LEN, N_PAST_FEATURES), torch.randn(batch, PRED_LEN, future_dim)

    def run(model, batch_inputs):
        return model(*batch_inputs)[0]

    return make, inputs, run


def tft_factory():
    def make():
        return build_tft(BaselineConfig(), past_len=PAST_LEN, pred_len=PRED_LEN, futr_names=TFT_FUTR)

    def inputs(batch: int):
        return torch.randn(batch, PAST_LEN, 1), torch.randn(batch, PAST_LEN + PRED_LEN, len(TFT_FUTR))

    def run(model, batch_inputs):
        return tft_forward(model, *batch_inputs)

    return make, inputs, run


# Decoder inputs: 2 weather, +6 calendar, +3 values 24 h earlier.
MODELS = {
    "AEDL, original (2 x 128)": aedl_factory(2, hidden_size=128, num_layers=2),
    "A (96 x 1, window scaling)": aedl_factory(11),
    "B (96 x 1, linear shortcut)": aedl_factory(11, "linear"),
    "Plain AEDL (96 x 1)": aedl_factory(8),
    "Plain LSTM (96 x 1)": aedl_factory(8, architecture="lstm"),
    "TFT (hidden 64)": tft_factory(),
}


def median_seconds(fn, warmup: int, repeats: int) -> float:
    for _ in range(warmup):
        fn()
    timings = []
    for _ in range(repeats):
        started = time.perf_counter()
        fn()
        timings.append(time.perf_counter() - started)
    return statistics.median(timings)


def benchmark(name: str, factory) -> dict[str, object]:
    make, inputs, run = factory
    # Train mode: the fused eval-mode path of nn.MultiheadAttention is not seen
    # by FlopCounterMode. Dropout does not change the FLOP count.
    model = make().train()
    parameters = sum(p.numel() for p in model.parameters())
    with torch.no_grad(), FlopCounterMode(display=False) as counter:
        run(model, inputs(TRAIN_BATCH))

    # Unique non-parameter tensors saved for backward (one batch).
    parameter_storages = {p.untyped_storage().data_ptr() for p in model.parameters()}
    storages: dict[int, int] = {}

    def pack(tensor):
        storage = tensor.untyped_storage()
        if storage.data_ptr() not in parameter_storages:
            storages[storage.data_ptr()] = storage.nbytes()
        return tensor

    with torch.autograd.graph.saved_tensors_hooks(pack, lambda tensor: tensor):
        output = run(model, inputs(TRAIN_BATCH))
    output.float().mean().backward()

    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-4)
    batch = inputs(TRAIN_BATCH)

    def train_step():
        optimizer.zero_grad(set_to_none=True)
        run(model, batch).float().mean().backward()
        optimizer.step()

    step = median_seconds(train_step, warmup=5, repeats=30)
    model.eval()
    single = inputs(1)
    with torch.no_grad():
        latency = median_seconds(lambda: run(model, single), warmup=10, repeats=100)
    return {
        "model": name,
        "parameters": parameters,
        "MFLOPs_per_origin": counter.get_total_flops() / TRAIN_BATCH / 1e6,
        "train_memory_MB": (16 * parameters + sum(storages.values())) / 2**20,
        "epoch_s": step * -(-TRAIN_ORIGINS // TRAIN_BATCH),
        "latency_ms": 1000 * latency,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--csv", type=Path, default=ROOT / "results" / "compute_benchmark.csv")
    args = parser.parse_args()
    warnings.filterwarnings("ignore")
    logging.getLogger("lightning_fabric").setLevel(logging.ERROR)
    logging.getLogger("pytorch_lightning").setLevel(logging.ERROR)
    torch.manual_seed(0)
    table = pd.DataFrame([benchmark(name, factory) for name, factory in MODELS.items()]).set_index("model")
    args.csv.parent.mkdir(parents=True, exist_ok=True)
    table.to_csv(args.csv)
    print(table.round(2).to_string())


if __name__ == "__main__":
    main()
