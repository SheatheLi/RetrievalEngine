"""Trainable retrieval -> real FAISS HNSW -> LambdaMART -> diagnostic ranking.

Global chronological splits and full-catalog evaluation are in movielens_data.
No GPU, online recommendation, or novel ANN/learning algorithm claim is made.
"""
from __future__ import annotations

from dataclasses import dataclass
from hashlib import sha256
from pathlib import Path
import json
import time
import numpy as np
import torch
from torch import nn
import torch.nn.functional as F
import faiss
import lightgbm as lgb
from .movielens_data import TemporalData


class Retriever(nn.Module):
    def __init__(self, users: int, items: int, dim: int = 48):
        super().__init__()
        self.users = nn.Embedding(users, dim)
        self.items = nn.Embedding(items, dim)
        nn.init.normal_(self.users.weight, std=.05)
        nn.init.normal_(self.items.weight, std=.05)

    def score(self, users: torch.Tensor, items: torch.Tensor) -> torch.Tensor:
        return (self.users(users) * self.items(items)).sum(-1)


def interaction_statistics(records: np.ndarray, users: int, items: int):
    positive = records[records[:, 2] >= 4]
    interaction = np.zeros((users, items), dtype=np.float32)
    interaction[positive[:, 0], positive[:, 1]] = 1
    counts = interaction.sum(0)
    norms = np.sqrt(np.maximum(counts, 1))
    cooccurrence = interaction.T @ interaction
    similarity = cooccurrence / (norms[:, None] * norms[None, :])
    np.fill_diagonal(similarity, 0.)
    similarity *= np.minimum(cooccurrence / 20., 1.)
    neighbors = min(100, items)
    cutoff = np.partition(similarity, -neighbors, axis=1)[:, -neighbors]
    similarity[similarity < cutoff[:, None]] = 0.
    return counts, interaction.sum(1), interaction, similarity


def fit_retriever(data: TemporalData, seed: int, epochs: int = 30,
                  dim: int = 48) -> tuple[Retriever, list[float]]:
    torch.manual_seed(seed)
    rng = np.random.default_rng(seed)
    model = Retriever(data.users, data.items, dim)
    opt = torch.optim.AdamW(model.parameters(), lr=.015, weight_decay=.02)
    pairs = data.train[data.train[:, 2] >= 4, :2]
    if len(pairs) == 0:
        raise ValueError("no training positives")
    # Unknown items are proxy negatives, never asserted to be true dislikes.
    seen = np.zeros((data.users, data.items), dtype=bool)
    seen[data.train[:, 0], data.train[:, 1]] = True
    losses = []
    model.train()
    for _ in range(epochs):
        shuffled = pairs[rng.permutation(len(pairs))]
        total = 0.
        for start in range(0, len(shuffled), 1024):
            batch = shuffled[start:start + 1024]
            negative = rng.choice(data.catalog, len(batch))
            invalid = seen[batch[:, 0], negative]
            attempts = 0
            while invalid.any():
                negative[invalid] = rng.choice(data.catalog, int(invalid.sum()))
                invalid = seen[batch[:, 0], negative]
                attempts += 1
                if attempts > 100:
                    raise ValueError("no usable negative support for a training user")
            user, positive = torch.from_numpy(batch[:, 0]), torch.from_numpy(batch[:, 1])
            delta = model.score(user, positive) - model.score(user, torch.from_numpy(negative))
            loss = F.softplus(-delta).mean()
            opt.zero_grad(set_to_none=True)
            loss.backward()
            if not all(p.grad is None or torch.isfinite(p.grad).all() for p in model.parameters()):
                raise FloatingPointError("nonfinite training gradient")
            opt.step()
            total += float(loss.detach()) * len(batch)
        losses.append(total / len(pairs))
    return model.eval(), losses


@dataclass
class Engine:
    user_vectors: np.ndarray
    item_vectors: np.ndarray
    catalog: np.ndarray
    popularity: np.ndarray
    user_counts: np.ndarray
    interaction: np.ndarray
    similarity: np.ndarray
    ranker: lgb.Booster | None = None
    ef_search: int = 32
    baseline_user_vectors: np.ndarray | None = None
    baseline_item_vectors: np.ndarray | None = None
    baseline_similarity: np.ndarray | None = None
    history_item_vectors: np.ndarray | None = None
    history_projection: np.ndarray | None = None
    history_max_items: int = 32

    def __post_init__(self):
        if self.history_max_items < 1:
            raise ValueError("history_max_items must be positive")
        if (self.user_vectors.ndim != 2 or self.item_vectors.ndim != 2 or
            self.user_vectors.shape[1] != self.item_vectors.shape[1] or
            len(self.catalog) == 0 or len(np.unique(self.catalog)) != len(self.catalog) or
            (self.catalog < 1).any() or (self.catalog >= len(self.item_vectors)).any()):
            raise ValueError("incompatible representation/catalog shapes")
        if not np.isfinite(self.user_vectors).all() or not np.isfinite(self.item_vectors).all():
            raise ValueError("nonfinite model representation")
        faiss.omp_set_num_threads(1)
        self.exact = faiss.IndexFlatIP(self.item_vectors.shape[1])
        self.hnsw = faiss.IndexHNSWFlat(self.item_vectors.shape[1], 16, faiss.METRIC_INNER_PRODUCT)
        self.hnsw.hnsw.efConstruction = 80
        vectors = np.ascontiguousarray(self.item_vectors[self.catalog], dtype=np.float32)
        self.exact.add(vectors)
        self.hnsw.add(vectors)
        self.hnsw.hnsw.efSearch = self.ef_search
        self._make_history_indexes()

    def _make_history_indexes(self):
        self.history_exact = self.history_hnsw = None
        if self.history_item_vectors is None and self.history_projection is None:
            return
        if self.history_item_vectors is None or self.history_projection is None:
            raise ValueError("history encoder is missing a component")
        dim = self.history_item_vectors.shape[1] if self.history_item_vectors.ndim == 2 else 0
        if (not dim or self.history_item_vectors.shape[0] != len(self.item_vectors) or
            self.history_projection.shape != (dim, dim) or
            not np.isfinite(self.history_item_vectors).all() or
            not np.isfinite(self.history_projection).all()):
            raise ValueError("invalid history encoder representation")
        vectors = np.ascontiguousarray(self.history_item_vectors[self.catalog], dtype=np.float32)
        self.history_exact = faiss.IndexFlatIP(dim)
        self.history_exact.add(vectors)
        self.history_hnsw = faiss.IndexHNSWFlat(dim, 16, faiss.METRIC_INNER_PRODUCT)
        self.history_hnsw.hnsw.efConstruction = 80
        self.history_hnsw.hnsw.efSearch = self.ef_search
        self.history_hnsw.add(vectors)

    def attach_history_encoder(self, model):
        self.history_max_items = model.max_history
        self.history_item_vectors = model.items.weight.detach().cpu().numpy().copy()
        self.history_projection = model.projection.weight.detach().cpu().numpy().copy()
        self._make_history_indexes()

    def history_query(self, known_positives: list[int] | tuple[int, ...]) -> np.ndarray:
        if self.history_item_vectors is None:
            raise ValueError("inductive history model not trained")
        if not isinstance(known_positives, (list, tuple)):
            raise ValueError("positive history requires a timestamp-ordered sequence")
        allowed = set(self.catalog.tolist())
        known = [int(item) for item in known_positives if int(item) in allowed][-self.history_max_items:]
        if not known:
            raise ValueError("no known positive history for inductive retrieval")
        pooled = self.history_item_vectors[known].mean(axis=0)
        return np.asarray(self.history_projection @ pooled, dtype=np.float32)

    def history_retrieve(self, known_positives: list[int] | tuple[int, ...], seen: set[int],
                         budget: int, approximate: bool) -> np.ndarray:
        if not 1 <= budget <= len(self.catalog):
            raise ValueError("invalid retrieval budget")
        q = self.history_query(known_positives)[None, :]
        index = self.history_hnsw if approximate else self.history_exact
        count = min(len(self.catalog), budget + len(seen))
        _, positions = index.search(np.ascontiguousarray(q), count)
        ids = [int(self.catalog[p]) for p in positions[0] if p >= 0
               and int(self.catalog[p]) not in seen]
        return np.asarray(ids[:budget], dtype=np.int64)

    @classmethod
    def from_training(cls, data: TemporalData, model: Retriever) -> "Engine":
        # Every competitor can use the same 80% development evidence. The neural
        # retriever intentionally freezes at 65%; the next 15% trains the ranker.
        development = np.concatenate((data.train, data.rank_train))
        counts, user_counts, interaction, similarity = interaction_statistics(development, data.users, data.items)
        return cls(model.users.weight.detach().numpy().copy(),
                   model.items.weight.detach().numpy().copy(), data.catalog.copy(), counts,
                   user_counts, interaction, similarity)

    def features(self, user: int, items: np.ndarray) -> np.ndarray:
        dot = self.item_vectors[items] @ self.user_vectors[user]
        knn = self.interaction[user] @ self.similarity[:, items]
        return np.column_stack((dot, np.log1p(self.popularity[items]), knn,
                                np.full(len(items), np.log1p(self.user_counts[user])),
                                np.linalg.norm(self.item_vectors[items], axis=1))).astype(np.float32)

    def retrieve(self, user: int, seen: set[int], budget: int, approximate: bool) -> np.ndarray:
        if not 0 < user < len(self.user_vectors) or not 1 <= budget <= len(self.catalog):
            raise ValueError("unknown user or invalid retrieval budget")
        index = self.hnsw if approximate else self.exact
        count = min(len(self.catalog), budget + len(seen))
        _, positions = index.search(self.user_vectors[user:user + 1].astype(np.float32), count)
        candidates = [int(self.catalog[p]) for p in positions[0] if p >= 0 and int(self.catalog[p]) not in seen]
        return np.asarray(candidates[:budget], dtype=np.int64)

    def recommend(self, user: int, seen: set[int], budget: int = 100,
                  method: str = "hnsw_ranker", top: int = 10,
                  history_positive: list[int] | tuple[int, ...] | None = None) -> tuple[np.ndarray, np.ndarray]:
        if not 0 < user < len(self.user_vectors) or type(top) is not int or top <= 0:
            raise ValueError("unknown user or invalid result count")
        if method in ("history_exact", "history_hnsw"):
            if history_positive is None:
                raise ValueError("positive history must be supplied separately from all rated items")
            vector = self.history_query(history_positive)
            ids = self.history_retrieve(history_positive, seen, budget, method == "history_hnsw")
            scores = self.history_item_vectors[ids] @ vector
        elif method in ("popularity", "item_knn", "mf80_exact"):
            ids = np.asarray([int(i) for i in self.catalog if int(i) not in seen], dtype=np.int64)
            if method == "popularity":
                scores = self.popularity[ids]
            elif method == "item_knn":
                similarity = self.baseline_similarity if self.baseline_similarity is not None else self.similarity
                scores = self.interaction[user] @ similarity[:, ids]
            else:
                if self.baseline_user_vectors is None or self.baseline_item_vectors is None:
                    raise ValueError("80% development-data MF baseline not trained")
                scores = self.baseline_item_vectors[ids] @ self.baseline_user_vectors[user]
        elif method in ("mf_exact", "mf_hnsw", "hnsw_ranker"):
            ids = self.retrieve(user, seen, budget, method != "mf_exact")
            if len(ids) == 0:
                return ids, np.empty(0, dtype=np.float64)
            if method == "hnsw_ranker":
                if self.ranker is None:
                    raise ValueError("ranker not trained")
                scores = self.ranker.predict(self.features(user, ids), num_threads=1)
            else:
                scores = self.item_vectors[ids] @ self.user_vectors[user]
        else:
            raise ValueError("unknown recommendation method")
        order = np.lexsort((ids, -scores))[:top]
        return ids[order], np.asarray(scores)[order]

    def save(self, folder: Path, metadata: dict) -> dict:
        if self.ranker is None:
            raise ValueError("cannot save an incomplete model")
        folder.mkdir(parents=True, exist_ok=True)
        state = {name: getattr(self, name) for name in (
            "user_vectors", "item_vectors", "catalog", "popularity", "user_counts", "interaction",
            "similarity", "baseline_user_vectors", "baseline_item_vectors", "baseline_similarity",
            "history_item_vectors", "history_projection")}
        np.savez_compressed(folder / "retrieval.npz", **{k: v for k, v in state.items() if v is not None})
        self.ranker.save_model(str(folder / "ranker.txt"))
        faiss.write_index(self.hnsw, str(folder / "hnsw.index"))
        names = ["retrieval.npz", "ranker.txt", "hnsw.index"]
        if self.history_hnsw is not None:
            faiss.write_index(self.history_hnsw, str(folder / "history_hnsw.index"))
            names.append("history_hnsw.index")
        hashes = {n: sha256((folder / n).read_bytes()).hexdigest() for n in names}
        manifest = {"history_max_items": self.history_max_items,
                    "schema": "two-stage-recommender.v3" if self.history_hnsw is not None
                    else "two-stage-recommender.v2", "ef_search": self.ef_search,
                    "sha256": hashes, **metadata}
        (folder / "model.json").write_text(json.dumps(manifest, indent=2) + "\n")
        return manifest

    @classmethod
    def load(cls, folder: Path) -> "Engine":
        manifest = json.loads((folder / "model.json").read_text())
        if manifest.get("schema") not in ("two-stage-recommender.v2", "two-stage-recommender.v3"):
            raise ValueError("unsupported model schema")
        names = ["retrieval.npz", "ranker.txt", "hnsw.index"]
        if manifest["schema"] == "two-stage-recommender.v3":
            names.append("history_hnsw.index")
        for name in names:
            if sha256((folder / name).read_bytes()).hexdigest() != manifest["sha256"][name]:
                raise ValueError("model/index version mismatch: " + name)
        with np.load(folder / "retrieval.npz", allow_pickle=False) as z:
            state = {name: z[name].copy() for name in z.files}
        engine = cls(**state, ranker=lgb.Booster(model_file=str(folder / "ranker.txt")),
                     ef_search=int(manifest["ef_search"]),
                     history_max_items=int(manifest.get("history_max_items", 32)))
        index = faiss.read_index(str(folder / "hnsw.index"))
        if index.d != engine.item_vectors.shape[1] or index.ntotal != len(engine.catalog):
            raise ValueError("index/representation shape mismatch")
        engine.hnsw = index
        if manifest["schema"] == "two-stage-recommender.v3":
            if engine.history_hnsw is None:
                raise ValueError("missing inductive history parameters")
            history_index = faiss.read_index(str(folder / "history_hnsw.index"))
            if history_index.d != engine.history_item_vectors.shape[1] or history_index.ntotal != len(engine.catalog):
                raise ValueError("history index/representation shape mismatch")
            engine.history_hnsw = history_index
        return engine


def fit_ranker(data: TemporalData, engine: Engine, seed: int) -> lgb.Booster:
    rng = np.random.default_rng(seed)
    features, labels, groups = [], [], []
    prefix_seen = {int(u): set(data.train[data.train[:, 0] == u, 1].tolist())
                   for u in np.unique(data.train[:, 0])}
    # Prevent the ranker feature generator from observing its own future labels.
    old_statistics = (engine.popularity, engine.user_counts, engine.interaction, engine.similarity)
    engine.popularity, engine.user_counts, engine.interaction, engine.similarity = interaction_statistics(
        data.train, data.users, data.items)
    try:
        for u in sorted(prefix_seen):
            positive = set(data.rank_train[(data.rank_train[:, 0] == u) & (data.rank_train[:, 2] >= 4), 1].tolist())
            positive = (positive & set(data.catalog.tolist())) - prefix_seen[u]
            if not positive:
                continue
            available = np.asarray(sorted(set(data.catalog.tolist()) - prefix_seen[u] - positive))
            negative = rng.choice(available, min(len(available), max(40, 3 * len(positive))), replace=False)
            ids = np.r_[sorted(positive), negative].astype(np.int64)
            features.append(engine.features(u, ids))
            labels.extend([1] * len(positive) + [0] * len(negative))
            groups.append(len(ids))
    finally:
        engine.popularity, engine.user_counts, engine.interaction, engine.similarity = old_statistics
    if not groups:
        raise ValueError("no ranker training groups")
    ranker = lgb.LGBMRanker(objective="lambdarank", n_estimators=80, num_leaves=15,
                           learning_rate=.05, min_child_samples=40, reg_lambda=1.,
                           n_jobs=1, random_state=seed, verbosity=-1, deterministic=True,
                           force_col_wise=True)
    ranker.fit(np.concatenate(features), np.asarray(labels), group=groups)
    return ranker.booster_


def metrics(ranked: np.ndarray, relevant: set[int]) -> tuple[float, float]:
    if not relevant:
        raise ValueError("a positive-only ranking query needs at least one relevant item")
    hit = np.asarray([int(i) in relevant for i in ranked], dtype=np.float64)
    dcg = (hit / np.log2(np.arange(len(hit)) + 2)).sum()
    ideal = (1. / np.log2(np.arange(min(len(relevant), len(hit))) + 2)).sum()
    return float(hit.sum() / len(relevant)), float(dcg / ideal if ideal else 0.)


def evaluate(data: TemporalData, engine: Engine, stage: str, budget: int = 100) -> dict:
    queries = data.queries(stage)
    if not queries:
        raise ValueError("no evaluable future-positive queries")
    methods = ["popularity", "item_knn", "mf80_exact", "mf_exact", "mf_hnsw", "hnsw_ranker"]
    if engine.baseline_user_vectors is None:
        methods.remove("mf80_exact")
    rows = []
    rng = np.random.default_rng(418)
    for u, seen, relevant in queries:
        exact = set(engine.retrieve(u, seen, budget, False).tolist())
        approximate = set(engine.retrieve(u, seen, budget, True).tolist())
        possible = min(10, len(relevant & approximate))
        ideal = (1. / np.log2(np.arange(min(10, len(relevant))) + 2)).sum()
        ceiling = float((1. / np.log2(np.arange(possible) + 2)).sum() / ideal)
        for method in rng.permutation(methods).tolist():
            start = time.perf_counter_ns()
            ranked, _ = engine.recommend(u, seen, budget, method)
            elapsed = (time.perf_counter_ns() - start) / 1e6
            recall, ndcg = metrics(ranked, relevant)
            rows.append({"user": u, "method": method, "relevant": len(relevant),
                         "recall10": recall, "ndcg10": ndcg, "elapsed_ms": elapsed,
                         "candidate_recall": len(relevant & approximate) / len(relevant),
                         "candidate_ndcg10_ceiling": ceiling,
                         "ann_vs_exact_recall": len(exact & approximate) / max(1, len(exact))})
    summaries = {}
    for method in methods:
        subset = [r for r in rows if r["method"] == method]
        times = [r["elapsed_ms"] for r in subset]
        summaries[method] = {"queries": len(subset),
            **{key: float(np.mean([r[key] for r in subset])) for key in ("recall10", "ndcg10")},
            "p50_ms": float(np.quantile(times, .5)), "p95_ms": float(np.quantile(times, .95)),
            "p99_ms": float(np.quantile(times, .99))}
    future = data.validation if stage == "validation" else data.test
    unique = [r for r in rows if r["method"] == "mf_exact"]
    return {"stage": stage, "budget": budget, "queries": len(queries),
            "all_future_users": int(len(np.unique(future[:, 0]))),
            "future_positive_events": int((future[:, 2] >= 4).sum()),
            "candidate_recall": float(np.mean([r["candidate_recall"] for r in unique])),
            "candidate_ndcg10_ceiling": float(np.mean([r["candidate_ndcg10_ceiling"] for r in unique])),
            "ann_vs_exact_recall": float(np.mean([r["ann_vs_exact_recall"] for r in unique])),
            "methods": summaries, "rows": rows,
            "boundary": "Offline positive-only full-catalog ranking; excludes unknown users/items; function time, not service SLA"}
