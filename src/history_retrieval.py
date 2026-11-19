"""Inductive, time-causal history retrieval for users without trained ID embeddings.

Research feature; implicit BPR negatives are unobserved, not established dislikes.
Train examples use strictly earlier timestamps within the 65% training split.
"""
from __future__ import annotations

from dataclasses import dataclass
import numpy as np
import torch
from torch import nn
import torch.nn.functional as F


@dataclass(frozen=True)
class HistoryExample:
    target: int
    context: tuple[int, ...]
    excluded: frozenset[int]
    timestamp: int
    user: int


def history_examples(records: np.ndarray, catalog: np.ndarray, max_history: int = 32) -> list[HistoryExample]:
    """Build examples WITHOUT showing an event any item at its own timestamp.

    Catalog membership is frozen to the retriever's training-period items.
    Concurrent positive items cannot be sampled as negatives.
    """
    if max_history < 1:
        raise ValueError('max_history must be positive')
    allowed = {int(i) for i in catalog}
    if not allowed:
        raise ValueError('empty training catalog')
    if records.ndim != 2 or records.shape[1] != 4:
        raise ValueError('expected user,item,rating,timestamp records')
    ordered = records[np.argsort(records[:, 3], kind='stable')]
    prior_positive: dict[int, list[int]] = {}
    prior_seen: dict[int, set[int]] = {}
    examples: list[HistoryExample] = []
    # Cross-user timestamp ties and same-user timestamp ties are handled alike:
    # snapshot precedes the entire timestamp group, not each line within it.
    for group in np.split(ordered, np.flatnonzero(np.diff(ordered[:, 3])) + 1):
        same_time_positive: dict[int, set[int]] = {}
        for user, item, rating, _ in group:
            if rating >= 4 and int(item) in allowed:
                same_time_positive.setdefault(int(user), set()).add(int(item))
        for user, item, rating, timestamp in group:
            user, item = int(user), int(item)
            history = prior_positive.get(user, [])
            if rating >= 4 and item in allowed and history:
                exclusions = frozenset(prior_seen.get(user, set()) | same_time_positive.get(user, set()))
                examples.append(HistoryExample(item, tuple(history[-max_history:]), exclusions,
                                               int(timestamp), user))
        for user, item, rating, _ in group:
            user, item = int(user), int(item)
            prior_seen.setdefault(user, set()).add(item)
            if rating >= 4 and item in allowed:
                prior_positive.setdefault(user, []).append(item)
    return examples


class HistoryEncoder(nn.Module):
    """Shared item factors + learned history-to-query projection; no user ID table."""

    def __init__(self, items: int, dim: int = 32, max_history: int = 32):
        super().__init__()
        if max_history < 1:
            raise ValueError("max_history must be positive")
        self.max_history = max_history
        self.items = nn.Embedding(items, dim)
        self.projection = nn.Linear(dim, dim, bias=False)
        nn.init.normal_(self.items.weight, std=.05)
        nn.init.eye_(self.projection.weight)

    def query(self, history: torch.Tensor, valid: torch.Tensor) -> torch.Tensor:
        if history.ndim != 2 or history.shape != valid.shape or not valid.any(dim=1).all():
            raise ValueError('each query requires at least one usable positive item')
        embeddings = self.items(history) * valid.unsqueeze(-1)
        pooled = embeddings.sum(1) / valid.sum(1).unsqueeze(-1)
        return self.projection(pooled)

    @torch.no_grad()
    def encode(self, history: list[int] | tuple[int, ...],
               catalog: np.ndarray) -> np.ndarray:
        if not isinstance(history, (list, tuple)):
            raise ValueError("history must be timestamp ordered")
        allowed = set(int(i) for i in catalog)
        useful = [int(i) for i in history if int(i) in allowed][-self.max_history:]
        if not useful:
            raise ValueError('no known positive history for inductive retrieval')
        values = torch.tensor(useful, dtype=torch.long)[None, :]
        result = self.query(values, torch.ones_like(values, dtype=torch.bool))
        return result[0].cpu().numpy().astype(np.float32)


def fit_history_encoder(records: np.ndarray, catalog: np.ndarray, items: int,
                        seed: int, epochs: int = 8, dim: int = 32,
                        max_history: int = 32) -> tuple[HistoryEncoder, list[float], int]:
    """Train using 65%-period event prefixes only; return model/loss/sample count."""
    if epochs < 1 or dim < 1:
        raise ValueError('epochs/dim must be positive')
    examples = history_examples(records, catalog, max_history)
    if not examples:
        raise ValueError('no strictly historical positive-prefix examples')
    catalogue = np.asarray(catalog, dtype=np.int64)
    candidates = [(e, np.asarray(sorted(set(catalogue.tolist()) - e.excluded), dtype=np.int64))
                  for e in examples]
    candidates = [(e, neg) for e, neg in candidates if len(neg)]
    if not candidates:
        raise ValueError('no available proxy negatives')
    torch.manual_seed(seed)
    rng = np.random.default_rng(seed)
    model = HistoryEncoder(items, dim, max_history=max_history)
    opt = torch.optim.AdamW(model.parameters(), lr=.01, weight_decay=.01)
    loss_trace: list[float] = []
    model.train()
    for _ in range(epochs):
        epoch_total = 0.
        for part in np.array_split(rng.permutation(len(candidates)),
                                   max(1, (len(candidates) + 255) // 256)):
            selected = [candidates[int(i)] for i in part]
            batch = len(selected)
            width = max(len(e.context) for e, _ in selected)
            context = np.zeros((batch, width), dtype=np.int64)
            mask = np.zeros((batch, width), dtype=bool)
            pos = np.empty(batch, dtype=np.int64)
            neg = np.empty(batch, dtype=np.int64)
            for i, (example, negatives) in enumerate(selected):
                n = len(example.context)
                context[i, :n] = example.context
                mask[i, :n] = True
                pos[i] = example.target
                neg[i] = rng.choice(negatives)
            q = model.query(torch.from_numpy(context), torch.from_numpy(mask))
            positive = model.items(torch.from_numpy(pos))
            negative = model.items(torch.from_numpy(neg))
            loss = F.softplus(-((q * (positive - negative)).sum(-1))).mean()
            if not torch.isfinite(loss):
                raise FloatingPointError('nonfinite inductive loss')
            opt.zero_grad(set_to_none=True)
            loss.backward()
            opt.step()
            epoch_total += float(loss.detach()) * batch
        loss_trace.append(epoch_total / len(candidates))
    return model.eval(), loss_trace, len(candidates)
