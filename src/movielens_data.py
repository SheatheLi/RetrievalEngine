"""Research-only MovieLens 100K input and global chronological evaluation.

No demographic files are read. Raw GroupLens data must not be redistributed.
The download hash pins the bytes fetched from the official source on 2026-10-15.
"""
from __future__ import annotations

from dataclasses import dataclass
from hashlib import sha256
from pathlib import Path
import urllib.request
from zipfile import ZipFile
import numpy as np

URL = "https://files.grouplens.org/datasets/movielens/ml-100k.zip"
ZIP_SHA256 = "50d2a982c66986937beb9ffb3aa76efe955bf3d5c6b761f4e3a7cd717c6a3229"


def download(destination: Path) -> Path:
    destination.mkdir(parents=True, exist_ok=True)
    archive = destination / "ml-100k.zip"
    if not archive.exists():
        with urllib.request.urlopen(URL, timeout=60) as response:
            data = response.read(10_000_001)
        if len(data) > 10_000_000:
            raise ValueError("unexpected archive size")
        if sha256(data).hexdigest() != ZIP_SHA256:
            raise ValueError("official input changed; review before updating pin")
        archive.write_bytes(data)
    with ZipFile(archive) as z:
        if sha256(archive.read_bytes()).hexdigest() != ZIP_SHA256 or z.testzip():
            raise ValueError("MovieLens archive integrity failure")
        # Only the ratings needed by this experiment; no arbitrary extraction.
        ratings = destination / "u.data"
        ratings.write_bytes(z.read("ml-100k/u.data"))
    return ratings


@dataclass(frozen=True)
class TemporalData:
    train: np.ndarray
    rank_train: np.ndarray
    validation: np.ndarray
    test: np.ndarray
    users: int
    items: int
    catalog: np.ndarray
    cutoffs: tuple[int, int, int]
    input_sha256: str

    def prefix(self, stage: str) -> np.ndarray:
        if stage == "validation":
            return np.concatenate((self.train, self.rank_train))
        if stage == "test":
            return np.concatenate((self.train, self.rank_train, self.validation))
        raise ValueError("stage must be validation or test")

    def queries(self, stage: str) -> list[tuple[int, set[int], set[int]]]:
        """Freeze user history at the global cutoff, predict future liked items.

        This is an offline, positive-only ranking proxy, not exposure-aware CTR
        or an online business effect. Future events do not update the model.
        """
        future = self.validation if stage == "validation" else self.test
        history = self.prefix(stage)
        catalog = set(self.catalog.tolist())
        trained_users = set(self.train[self.train[:, 2] >= 4, 0].tolist())
        result = []
        for u in sorted(set(future[:, 0].tolist()) & trained_users):
            seen = set(history[history[:, 0] == u, 1].tolist())
            relevant = set(future[(future[:, 0] == u) & (future[:, 2] >= 4), 1].tolist())
            relevant = (relevant & catalog) - seen
            if relevant:
                result.append((int(u), seen, relevant))
        return result


    def query_cohort_audit(self, stage: str) -> dict:
        """Separate training-ID coverage, usable history, and catalogue support.

        Does not inspect labels beyond the chosen evaluation stage.
        This is a diagnostic, not an improvement in NDCG or an unbiased CTR estimate.
        """
        future = self.validation if stage == "validation" else self.test if stage == "test" else None
        if future is None:
            raise ValueError("stage must be validation or test")
        history = self.prefix(stage)
        catalog = set(map(int, self.catalog))
        trained = set(map(int, self.train[self.train[:, 2] >= 4, 0]))
        cohorts = {name: [] for name in (
            "trained_id", "history_only", "no_positive_history",
            "history_items_not_in_training_catalog", "no_evaluable_future_positive")}
        evaluable_queries = []
        for user in sorted(set(map(int, future[:, 0]))):
            past = history[history[:, 0] == user]
            seen = set(map(int, past[:, 1]))
            known_positive = set(map(int, past[past[:, 2] >= 4, 1]))
            usable_history = known_positive & catalog
            future_positive = set(map(int, future[(future[:, 0] == user) & (future[:, 2] >= 4), 1]))
            evaluable = (future_positive & catalog) - seen
            if not evaluable:
                cohorts["no_evaluable_future_positive"].append(user)
                continue
            if user in trained:
                cohort = "trained_id"
            elif usable_history:
                cohort = "history_only"
            elif known_positive:
                cohort = "history_items_not_in_training_catalog"
            else:
                cohort = "no_positive_history"
            cohorts[cohort].append(user)
            ordered_known = [int(i) for i in past[past[:, 2] >= 4, 1] if int(i) in catalog]
            evaluable_queries.append({"user": user, "cohort": cohort, "seen": seen,
                                      "known_positive": ordered_known, "relevant": evaluable})
        return {"stage": stage, "future_users": int(len(set(future[:, 0]))),
                "counts": {name: len(users) for name, users in cohorts.items()},
                "user_ids": cohorts, "queries": evaluable_queries}


def load(path: Path) -> TemporalData:
    raw = path.read_bytes()
    ratings = np.loadtxt(path, dtype=np.int64, delimiter="\t", ndmin=2)
    if ratings.shape[1] != 4 or len(ratings) < 100:
        raise ValueError("expected user, item, rating, timestamp records")
    if (ratings[:, :2] < 1).any() or not np.isin(ratings[:, 2], [1, 2, 3, 4, 5]).all():
        raise ValueError("invalid identifiers or ratings")
    # Stable ordering plus strict timestamp masks keep tied timestamps together.
    ratings = ratings[np.argsort(ratings[:, 3], kind="stable")]
    cuts = tuple(int(np.quantile(ratings[:, 3], q, method="nearest")) for q in (.65, .8, .9))
    t = ratings[:, 3]
    stages = [ratings[t < cuts[0]], ratings[(t >= cuts[0]) & (t < cuts[1])],
              ratings[(t >= cuts[1]) & (t < cuts[2])], ratings[t >= cuts[2]]]
    if any(len(x) == 0 for x in stages):
        raise ValueError("timestamp distribution cannot support four stages")
    return TemporalData(*stages, int(ratings[:, 0].max()) + 1,
                        int(ratings[:, 1].max()) + 1, np.unique(stages[0][:, 1]), cuts,
                        sha256(raw).hexdigest())
