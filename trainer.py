"""
Training and evaluation loops for OnionFinger.

These functions are deliberately kept simple and reusable so the same
train/eval machinery can be used both for the undefended baseline and
after applying any defense from `onionfinger.defenses`.
"""

import torch
from torch.utils.data import DataLoader
from typing import Optional, Tuple


def train_epoch(
    model: torch.nn.Module,
    loader: DataLoader,
    optimizer: torch.optim.Optimizer,
    criterion: torch.nn.Module,
    device: torch.device,
    max_grad_norm: float = 5.0,
    subsample_step: Optional[int] = None,
) -> Tuple[float, float]:
    """
    Run one training epoch.

    Returns:
        (average_loss, accuracy)
    """
    model.train()
    total_loss = 0.0
    correct = 0
    total = 0

    for X, y in loader:
        X = X.to(device, non_blocking=True)
        y = y.to(device, non_blocking=True)

        # Ensure input is (B, 1, L)
        if X.dim() == 2:
            X = X.unsqueeze(1)

        # Optional subsampling for Transformer (e.g. every 5th packet)
        if subsample_step is not None and subsample_step > 1:
            X = X[:, :, ::subsample_step]

        optimizer.zero_grad(set_to_none=True)
        logits = model(X)
        loss = criterion(logits, y)
        loss.backward()

        # Gradient clipping (helps stability on fingerprinting traces)
        if max_grad_norm is not None:
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_grad_norm)

        optimizer.step()

        # Metrics
        total_loss += loss.item() * X.size(0)
        pred = logits.argmax(dim=1)
        correct += (pred == y).sum().item()
        total += X.size(0)

    avg_loss = total_loss / total if total > 0 else 0.0
    acc = correct / total if total > 0 else 0.0
    return avg_loss, acc


@torch.no_grad()
def evaluate(
    model: torch.nn.Module,
    loader: DataLoader,
    criterion: torch.nn.Module,
    device: torch.device,
    subsample_step: Optional[int] = None,
) -> Tuple[float, float]:
    """
    Evaluate on a validation or test loader.

    Returns:
        (average_loss, accuracy)
    """
    model.eval()
    total_loss = 0.0
    correct = 0
    total = 0

    for X, y in loader:
        X = X.to(device, non_blocking=True)
        y = y.to(device, non_blocking=True)

        if X.dim() == 2:
            X = X.unsqueeze(1)

        if subsample_step is not None and subsample_step > 1:
            X = X[:, :, ::subsample_step]

        logits = model(X)
        loss = criterion(logits, y)

        total_loss += loss.item() * X.size(0)
        pred = logits.argmax(dim=1)
        correct += (pred == y).sum().item()
        total += X.size(0)

    avg_loss = total_loss / total if total > 0 else 0.0
    acc = correct / total if total > 0 else 0.0
    return avg_loss, acc
