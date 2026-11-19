"""Offline retrieval evidence contracts for one frozen vector snapshot.

This module deliberately does NOT claim that offline exact-neighbor fidelity
implies recommendation relevance or online engagement. It is a small,
reproducible audit oracle independent of FAISS internals.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Sequence
import numpy as np


@dataclass(frozen=True)
class FrozenCatalog:
    ids: np.ndarray
    vectors: np.ndarray
    version: str

    def __post_init__(self) -> None:
        ids = np.asarray(self.ids)
        vec = np.asarray(self.vectors)
        if (ids.ndim != 1 or ids.dtype.kind not in 'iu' or
            len(ids) == 0 or len(np.unique(ids)) != len(ids)):
            raise ValueError('catalog ids must be a nonempty unique integer vector')
        if (vec.ndim != 2 or vec.shape[0] != len(ids) or vec.shape[1] == 0 or
            vec.dtype.kind not in 'fiu' or not np.isfinite(vec).all()):
            raise ValueError('finite 2d vectors must align with catalog IDs')
        if type(self.version) is not str or not self.version.strip():
            raise ValueError('frozen catalog version required')
        # Own immutable copies to prevent callers from mutating our evidence.
        safe_ids = ids.astype(np.int64, copy=True)
        safe_vec = vec.astype(np.float64, copy=True)
        safe_ids.setflags(write=False)
        safe_vec.setflags(write=False)
        object.__setattr__(self, 'ids', safe_ids)
        object.__setattr__(self, 'vectors', safe_vec)


@dataclass(frozen=True)
class QueryCase:
    key: str
    vector: np.ndarray
    seen: frozenset[int]
    relevant: frozenset[int]
    cohort: str = 'known_user'

    def __post_init__(self) -> None:
        vec = np.asarray(self.vector)
        if vec.ndim != 1 or vec.size == 0 or vec.dtype.kind not in 'fiu' or not np.isfinite(vec).all():
            raise ValueError('finite 1d query vector required')
        if not self.key or not self.cohort:
            raise ValueError('query key and cohort required')
        if any(type(item) is not int for item in self.seen | self.relevant):
            raise ValueError('seen and relevant must contain Python integer IDs')
        safe = vec.astype(np.float64, copy=True)
        safe.setflags(write=False)
        object.__setattr__(self, 'vector', safe)


@dataclass(frozen=True)
class ResultRow:
    query: str
    cohort: str
    snapshot: str
    method: str
    k: int
    ranking: tuple[int, ...]
    exact_ranking: tuple[int, ...]
    neighbor_recall: float | None
    relevant_recall: float | None
    elapsed_ns: int
    # Failure to fill k may be caused by seen filtering, not index approximate quality.
    eligible_count: int


def exact_topk(catalog: FrozenCatalog, case: QueryCase, k: int) -> tuple[int, ...]:
    """Deterministic inner-product truth after excluding seen IDs.

    Ties break by ascending business ID, not a backend's opaque index position.
    This is float64 oracle arithmetic; extreme magnitude/rounding is rejected.
    """
    if type(k) is not int or not 1 <= k <= len(catalog.ids):
        raise ValueError('k outside catalogue capacity')
    if case.vector.shape != (catalog.vectors.shape[1],):
        raise ValueError('query dimension mismatches snapshot')
    allowed = ~np.isin(catalog.ids, list(case.seen))
    ids = catalog.ids[allowed]
    if len(ids) == 0:
        return ()
    with np.errstate(over='raise', invalid='raise'):
        scores = catalog.vectors[allowed] @ case.vector
    if not np.isfinite(scores).all():
        raise ValueError('nonfinite scores make exact oracle invalid')
    order = np.lexsort((ids, -scores))
    return tuple(int(x) for x in ids[order[:k]])


def audit_result(catalog: FrozenCatalog, case: QueryCase, k: int,
                 ranking: Sequence[int], method: str, elapsed_ns: int) -> ResultRow:
    """Reject stale/unqualified outputs before comparing ANN to exact truth.

    For offline positive-only recommendation, relevant_recall is conditional
    on future positives and *not* CTR, causal uplift, or user satisfaction.
    """
    if not method or type(elapsed_ns) is not int or elapsed_ns < 0:
        raise ValueError('method and nonnegative measured latency required')
    if type(k) is not int or not 1 <= k <= len(catalog.ids):
        raise ValueError('invalid k')
    values = list(ranking)
    if (any(type(x) is not int for x in values) or len(values) != len(set(values))
        or len(values) > k):
        raise ValueError('duplicate, noninteger, or excess ranked IDs')
    ids = set(int(x) for x in catalog.ids)
    if (set(values) - ids) or (set(values) & case.seen):
        raise ValueError('result contains unknown or excluded ID')
    truth = exact_topk(catalog, case, k)
    eligible = len(ids - case.seen)
    if len(values) != min(k, eligible):
        raise ValueError('truncated result: cannot silently count as valid query')
    denominator = len(truth)
    neighbor_recall = len(set(values) & set(truth)) / denominator if denominator else None
    possible_relevant = (case.relevant & ids) - case.seen
    relevant_recall = (len(possible_relevant & set(values)) / len(possible_relevant)
                       if possible_relevant else None)
    return ResultRow(case.key, case.cohort, catalog.version, method, k,
                     tuple(values), truth, neighbor_recall,
                     relevant_recall, elapsed_ns, eligible)


def summarize(rows: Sequence[ResultRow]) -> dict:
    """Micro-average neighbor recall, query-macro relevant recall, latency percentiles.

    We do not attach confidence intervals unless query resampling is performed.
    The caller must group homogeneous k/snapshot/methods before summarizing.
    """
    if not rows:
        raise ValueError('nonempty rows required')
    if len({(x.method, x.snapshot, x.k) for x in rows}) != 1:
        raise ValueError('cannot aggregate across method, snapshot, or k')
    if len(set(x.query for x in rows)) != len(rows):
        raise ValueError('duplicate query: latency distribution would be biased')
    measured = np.asarray([r.elapsed_ns / 1e6 for r in rows], dtype=float)
    eligible_relevance = [r.relevant_recall for r in rows if r.relevant_recall is not None]
    eligible_neighbors = [r.neighbor_recall for r in rows if r.neighbor_recall is not None]
    return {
        'method': rows[0].method, 'snapshot': rows[0].snapshot, 'k': rows[0].k,
        'queries': len(rows), 'relevance_queries': len(eligible_relevance),
        'neighbor_evaluable_queries': len(eligible_neighbors),
        'mean_neighbor_recall': float(np.mean(eligible_neighbors)) if eligible_neighbors else None,
        'mean_future_positive_recall': (float(np.mean(eligible_relevance))
                                        if eligible_relevance else None),
        'p50_ms': float(np.percentile(measured, 50)),
        'p95_ms': float(np.percentile(measured, 95)),
        'p99_ms': float(np.percentile(measured, 99)),
        'not_proven': ['online CTR', 'user satisfaction', 'p95 service latency',
                       'equal-recall Pareto optimality', 'deployment scalability'],
    }


def paired_neighbor_delta(a: Sequence[ResultRow], b: Sequence[ResultRow], *,
                          bootstrap_seed: int = 17, bootstrap_samples: int = 1000) -> dict:
    """Query-level paired bootstrap for *descriptive* recall difference.

    Resampling queries cannot address biased cohorts, nonindependent users,
    snapshot drift or unrepresentative data. No generalization guarantee.
    """
    if type(bootstrap_samples) is not int or bootstrap_samples < 100:
        raise ValueError('insufficient bootstrap samples')
    left, right = {r.query: r for r in a}, {r.query: r for r in b}
    if len(left) != len(a) or len(right) != len(b) or not left or left.keys() != right.keys():
        raise ValueError('requires identical unique query keys')
    ordered = sorted(left)
    for key in ordered:
        l, r = left[key], right[key]
        if (l.snapshot != r.snapshot or l.k != r.k or l.exact_ranking != r.exact_ranking
            or l.cohort != r.cohort):
            raise ValueError('paired trial metadata or ground truth mismatch')
    if any(left[key].neighbor_recall is None or right[key].neighbor_recall is None for key in ordered):
        raise ValueError('a paired neighbor comparison needs eligible catalog objects')
    diffs = np.array([left[key].neighbor_recall - right[key].neighbor_recall for key in ordered])
    rng = np.random.default_rng(bootstrap_seed)
    indices = rng.integers(0, len(diffs), size=(bootstrap_samples, len(diffs)))
    sampled = diffs[indices].mean(axis=1)
    return {'queries': len(diffs), 'a_minus_b_mean': float(diffs.mean()),
            'paired_bootstrap_95_percentile_interval': [float(x) for x in np.percentile(sampled,[2.5,97.5])],
            'limitations': 'query bootstrap is descriptive; correlated users and cohort shift are not controlled'}
