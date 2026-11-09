#!/usr/bin/env python3
"""Evaluate the preserved Transformer on the reconstructed Rimmer test set."""

import argparse
import hashlib
import json
import platform
import zipfile
from io import BytesIO
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import DataLoader, TensorDataset

from transforms import apply_defense, normalize_length
from transformer_fingerprinting import TransformerFingerprintingModel
from trainer import evaluate


DATASET_SHA256 = "6b284576f2b482bc3676bd7d71b863bdd671c67bef611fac683f41a6d15bc750"
CHECKPOINT_SHA256 = "5ca6278abe08c6df725a3388dfa59c5215757c6920eb5689a897d51c15edb881"
TEST_ROWS_LABELS_SHA256 = "3d1c602715489a340342d1206702d8becbb8be160cd809cc9440d2bc27ee5227"
SEED = 42
WINDOW = 5000
STRIDE = 10
PREFIX = 200
REWRITE_WINDOW = 12
BATCH_SIZE = 32


def verified_bytes(path, expected_hash):
    payload = path.read_bytes()
    if hashlib.sha256(payload).hexdigest() != expected_hash:
        raise ValueError(f"SHA256 differs from the preserved artifact: {path}")
    return payload


def reconstruct_test_rows(archive):
    # The archive digest has already been verified. Its object labels use pickle.
    with archive.open("labels.npy") as stream:
        labels = np.lib.format.read_array(stream, allow_pickle=True)
    classes, class_ids = np.unique(labels, return_inverse=True)
    if len(classes) != 100 or np.min(np.bincount(class_ids)) < 200:
        raise ValueError("Unexpected archive class inventory")

    rng = np.random.RandomState(SEED)
    selected = []
    for class_id in range(len(classes)):
        rows = np.flatnonzero(class_ids == class_id)
        rng.shuffle(rows)
        selected.extend(rows[:200])
    pool = np.asarray(selected)
    rng.shuffle(pool)
    pool_labels = class_ids[pool]

    test = []
    for class_id in range(len(classes)):
        positions = np.flatnonzero(pool_labels == class_id)
        rng.shuffle(positions)
        test.extend(pool[positions[:40]])
    rows = np.asarray(test)
    targets = class_ids[rows]
    # Hash ordered (zero-based source row, sorted-label class ID) pairs.
    packed = np.column_stack((rows, targets)).astype("<i8").tobytes(order="C")
    digest = hashlib.sha256(packed).hexdigest()
    if digest != TEST_ROWS_LABELS_SHA256:
        raise ValueError("Reconstructed test rows and labels differ from the reference")
    return rows, targets, digest


def read_directions(archive, rows):
    order = np.argsort(rows)
    row_ids = rows[order]
    with archive.open("data.npy") as stream:
        if np.lib.format.read_magic(stream) != (1, 0):
            raise ValueError("Unexpected data array version")
        shape, fortran, dtype = np.lib.format.read_array_header_1_0(stream)
        if fortran or shape != (249983, WINDOW) or dtype != np.int16:
            raise ValueError("Unexpected archive array layout")
        directions = np.empty((len(rows), WINDOW), dtype=np.int8)
        for start in range(0, shape[0], 2048):
            count = min(2048, shape[0] - start)
            block = np.frombuffer(stream.read(count * WINDOW * dtype.itemsize), dtype=dtype)
            lo, hi = np.searchsorted(row_ids, [start, start + count])
            directions[order[lo:hi]] = np.sign(block.reshape(count, WINDOW)[row_ids[lo:hi] - start])
    return directions


def rewrite_directions(directions):
    # Preserve the historical exclusive right bound and read the original input.
    window = REWRITE_WINDOW
    changes = directions[:, 1:] != directions[:, :-1]
    counts = np.pad(np.cumsum(changes, axis=1, dtype=np.int16), ((0, 0), (1, 0)))
    active = counts[:, window - 1:-1] - counts[:, :-window] > window // 2
    rewritten = directions.copy()
    middle = rewritten[:, window // 2:window // 2 + active.shape[1]]
    np.copyto(middle, directions[:, :active.shape[1]], where=active)
    return rewritten


def run(dataset, checkpoint):
    dataset_bytes = verified_bytes(dataset, DATASET_SHA256)
    checkpoint_bytes = verified_bytes(checkpoint, CHECKPOINT_SHA256)
    with zipfile.ZipFile(BytesIO(dataset_bytes)) as archive:
        rows, targets, split_digest = reconstruct_test_rows(archive)
        directions = read_directions(archive, rows)

    ckpt = torch.load(BytesIO(checkpoint_bytes), map_location="cpu", weights_only=True)
    config = ckpt["config"]["model"]
    model = TransformerFingerprintingModel(
        num_classes=config["num_classes"],
        seq_len=config["seq_len"],
        d_model=config["d_model"],
        nhead=config["nhead"],
        num_layers=config["num_layers"],
        dropout=config["transformer_dropout"],
    )
    model.load_state_dict(ckpt["model_state_dict"])

    def prefix(value):
        return np.stack([
            normalize_length(apply_defense(row, "front_padding", pad_length=PREFIX, pad_value=value), WINDOW)
            for row in directions
        ])

    rewritten = rewrite_directions(directions)
    fixtures = np.random.default_rng(SEED).integers(-1, 2, (200, 36), dtype=np.int8)
    for source, transformed in [(fixtures, rewrite_directions(fixtures)), (directions[:64], rewritten[:64])]:
        for original, actual in zip(source, transformed):
            np.testing.assert_array_equal(apply_defense(original, "burst_smoothing", window=REWRITE_WINDOW), actual)
    identity = np.stack([apply_defense(row, "timing_jitter", jitter_std=0.03) for row in directions])
    np.testing.assert_array_equal(identity, directions)

    criterion = torch.nn.CrossEntropyLoss()
    conditions = {}
    for name, inputs in [
        ("baseline", directions),
        ("zero_prefix_200", prefix(0)),
        ("outgoing_prefix_200", prefix(1)),
        ("direction_rewrite_12", rewritten),
    ]:
        tensors = TensorDataset(torch.from_numpy(inputs).float().unsqueeze(1), torch.from_numpy(targets).long())
        loader = DataLoader(tensors, batch_size=BATCH_SIZE)
        loss, accuracy = evaluate(model, loader, criterion, torch.device("cpu"), subsample_step=STRIDE)
        conditions[name] = {
            "correct": round(accuracy * len(rows)),
            "total": len(rows),
            "accuracy": accuracy,
            "loss": round(loss, 8),
        }
    conditions["timing_identity"] = {
        **conditions["baseline"],
        "verified_elementwise_identity": True,
        "metrics_reused_from_baseline": True,
    }
    return {
        "dataset_sha256": DATASET_SHA256,
        "checkpoint_sha256": CHECKPOINT_SHA256,
        "test_rows_labels_sha256": split_digest,
        "test_digest_encoding": "C-order little-endian int64 pairs of zero-based source row and numeric class ID",
        "script_sha256": hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
        "environment": {
            "python": platform.python_version(),
            "torch": torch.__version__,
            "numpy": np.__version__,
            "platform": platform.system(),
            "architecture": platform.machine(),
            "device": "cpu",
            "torch_threads": torch.get_num_threads(),
        },
        "checkpoint_epoch": ckpt["epoch"],
        "checkpoint_validation_accuracy": ckpt["val_acc"],
        "batch_size": BATCH_SIZE,
        "pool_and_test_seed": SEED,
        "source_window": WINDOW,
        "sample_stride": STRIDE,
        "historical_run_binding": "unresolved",
        "conditions": conditions,
    }


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset", required=True, type=Path)
    parser.add_argument("--checkpoint", required=True, type=Path)
    args = parser.parse_args()
    torch.set_num_threads(4)
    print(json.dumps(run(args.dataset, args.checkpoint), indent=2))
