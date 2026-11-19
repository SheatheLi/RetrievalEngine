"""Verified, *exact-only* read-only diagnostic backend for saved recommender artifacts.

It deliberately does not import FAISS or claim HNSW performance or equivalence.
It accepts existing v2/v3 artifacts after verifying *every* published component hash.
Queries over recorded ratings use a frozen global pre-test prefix, never test labels.
"""
from __future__ import annotations

from dataclasses import dataclass
from hashlib import sha256
import json
from pathlib import Path

import lightgbm as lgb
import numpy as np


@dataclass
class ExactDiagnostic:
    manifest: dict
    state: dict[str, np.ndarray]
    ranker: lgb.Booster

    @classmethod
    def load(cls, path: Path) -> "ExactDiagnostic":
        path = Path(path)
        meta = json.loads((path / "model.json").read_text(encoding="utf-8"))
        schema = meta.get("schema")
        if schema not in ("two-stage-recommender.v2", "two-stage-recommender.v3",
                          "two-stage-recommender.exact.v1"):
            raise ValueError("unsupported artifact schema")
        names = ["retrieval.npz", "ranker.txt"]
        if schema in ("two-stage-recommender.v2", "two-stage-recommender.v3"):
            names.append("hnsw.index")
        if schema.endswith("v3"):
            names.append("history_hnsw.index")
        hashes = meta.get("sha256")
        if not isinstance(hashes, dict) or set(hashes) != set(names):
            raise ValueError("incomplete artifact component hash inventory")
        for filename in names:
            if not isinstance(hashes[filename], str) or len(hashes[filename]) != 64:
                raise ValueError("invalid artifact digest: " + filename)
            if sha256((path / filename).read_bytes()).hexdigest() != hashes[filename]:
                raise ValueError("model/index version mismatch: " + filename)
        with np.load(path / "retrieval.npz", allow_pickle=False) as archive:
            arrays = {k: archive[k].copy() for k in archive.files}
        required = {"user_vectors", "item_vectors", "catalog", "popularity", "user_counts",
                    "interaction", "similarity"}
        if not required.issubset(arrays):
            raise ValueError("model array inventory incomplete")
        users, items, catalog = arrays["user_vectors"], arrays["item_vectors"], arrays["catalog"]
        if (users.ndim != 2 or items.ndim != 2 or users.shape[1] != items.shape[1]
            or users.shape[1] < 1 or catalog.ndim != 1 or not len(catalog)
            or catalog.dtype.kind not in "iu" or np.any(catalog < 1)
            or np.any(catalog >= len(items)) or len(np.unique(catalog)) != len(catalog)
            or arrays["popularity"].shape != (len(items),)
            or arrays["user_counts"].shape != (len(users),)
            or arrays["interaction"].shape != (len(users), len(items))
            or arrays["similarity"].shape != (len(items), len(items))):
            raise ValueError("invalid model representation dimensions/catalog")
        for name, shape in (("baseline_user_vectors", users.shape),
                            ("baseline_item_vectors", items.shape),
                            ("baseline_similarity", (len(items), len(items)))):
            if name in arrays and arrays[name].shape != shape:
                raise ValueError("invalid auxiliary array dimensions: " + name)
        for key, array in arrays.items():
            if array.dtype.kind not in "ifu" or not np.isfinite(array).all():
                raise ValueError("non-finite/non-numeric model state: " + key)
        if schema.endswith("v3") or (schema == "two-stage-recommender.exact.v1"
                                      and ("history_item_vectors" in arrays or "history_projection" in arrays)):
            hi, hp = arrays.get("history_item_vectors"), arrays.get("history_projection")
            if hi is None or hp is None or hi.ndim != 2 or hi.shape[0] != len(items) \
                    or hp.shape != (hi.shape[1], hi.shape[1]):
                raise ValueError("invalid history encoder dimensions")
            if type(meta.get("history_max_items")) is not int or meta["history_max_items"] < 1:
                raise ValueError("invalid history encoder max items")
        return cls(meta, arrays, lgb.Booster(model_file=str(path / "ranker.txt")))

    @property
    def catalog(self) -> np.ndarray:
        return self.state["catalog"]

    def _candidates(self, vector: np.ndarray, matrix: np.ndarray, seen: set[int], budget: int) -> np.ndarray:
        if type(budget) is not int or budget < 1 or budget > len(self.catalog):
            raise ValueError("invalid retrieval budget")
        ids = self.catalog[~np.isin(self.catalog, list(seen))]
        # A stable tie-breaking rule is essential when exact and ANN are compared.
        values = np.asarray(matrix[ids] @ vector, dtype=np.float64)
        if not np.isfinite(values).all():
            raise ValueError("invalid ranking score")
        return ids[np.lexsort((ids, -values))[:budget]]

    def recommend(self, user: int, seen: set[int], budget: int, *,
                  history: list[int] | None = None, top: int = 10) -> dict:
        if type(user) is not int or user < 1 or user >= len(self.state["user_vectors"]):
            raise ValueError("unknown user")
        if type(top) is not int or top < 1:
            raise ValueError("invalid result count")
        if history is not None:
            if "history_item_vectors" not in self.state or "history_projection" not in self.state:
                raise ValueError("history encoder unavailable in v2 artifact")
            if type(history) is not list:
                raise ValueError("history must be ordered")
            positive = [i for i in history if type(i) is int and i in set(self.catalog.tolist())]
            if not positive:
                raise ValueError("no usable positive history")
            vec = self.state["history_projection"] @ self.state["history_item_vectors"][positive[-self.manifest["history_max_items"]:]].mean(axis=0)
            ids = self._candidates(vec, self.state["history_item_vectors"], seen, budget)
            scores = self.state["history_item_vectors"][ids] @ vec
            source = "history_exact"
        else:
            ids = self._candidates(self.state["user_vectors"][user], self.state["item_vectors"], seen, budget)
            matrix = self.state
            features = np.column_stack((matrix["item_vectors"][ids] @ matrix["user_vectors"][user],
                np.log1p(matrix["popularity"][ids]),
                matrix["interaction"][user] @ matrix["similarity"][:, ids],
                np.full(len(ids), np.log1p(matrix["user_counts"][user])),
                np.linalg.norm(matrix["item_vectors"][ids], axis=1))).astype(np.float32)
            scores = self.ranker.predict(features, num_threads=1) if len(ids) else np.empty(0)
            source = "exact_ranker"
        if not np.isfinite(scores).all():
            raise ValueError("nonfinite recommendation scores")
        order = np.lexsort((ids, -scores))[:top]
        return {"user": user, "budget": budget, "method": source,
                "backend": "numpy_exact_no_faiss", "approximate": False,
                "items": [{"item": int(ids[i]), "score": float(scores[i])} for i in order]}
