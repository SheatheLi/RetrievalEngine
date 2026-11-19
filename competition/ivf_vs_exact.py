"""Fair *retrieval fidelity* audit against original saved MovieLens model vectors.

This is a locally constructed IVF KMeans diagnostic. It is NOT FAISS HNSW, and
has no observed relevance labels; metrics are exact-candidate preservation,
final LightGBM ranking agreement, work and local latency, not NDCG/CTR.
"""
from __future__ import annotations
import argparse
from dataclasses import dataclass
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path
import statistics
import sys
import time
from typing import Iterable

import numpy as np
from sklearn.cluster import KMeans

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))
from src.exact_diagnostic import ExactDiagnostic

N_CLUSTERS = 32
RETRIEVAL_BUDGET = 50
FINAL_TOP = 10
DEFAULT_PROBES = (1, 2, 4, 8, 16, 32)
DEFAULT_SEEN = {1, 2, 3, 4, 5, 9}  # synthetic; NOT actual MovieLens user history


@dataclass
class IVFDiagnostic:
    model: ExactDiagnostic
    n_clusters: int
    random_state: int

    def __post_init__(self):
        catalog = self.model.catalog
        vectors = np.asarray(self.model.state['item_vectors'][catalog], dtype=np.float64)
        if not 1 <= self.n_clusters <= len(vectors):
            raise ValueError('invalid cluster count')
        start = time.perf_counter_ns()
        self.kmeans = KMeans(n_clusters=self.n_clusters, random_state=self.random_state,
                             n_init=2, max_iter=100, algorithm='lloyd').fit(vectors)
        self.build_ms = (time.perf_counter_ns() - start) / 1e6
        self.centroids = np.asarray(self.kmeans.cluster_centers_)
        self.members = [np.flatnonzero(self.kmeans.labels_ == i) for i in range(self.n_clusters)]
        self.vectors = vectors
        self.catalog = np.asarray(catalog)

    def search(self, user: int, seen: set[int], budget: int, *, nprobe: int):
        if not isinstance(nprobe, int) or not 1 <= nprobe <= self.n_clusters:
            raise ValueError('invalid nprobe')
        if type(budget) is not int or not 1 <= budget <= len(self.catalog):
            raise ValueError('invalid budget')
        if not isinstance(seen, set) or not all(type(x) is int for x in seen):
            raise ValueError('invalid seen set')
        if not 0 < user < len(self.model.state['user_vectors']):
            raise ValueError('invalid user')
        query = np.asarray(self.model.state['user_vectors'][user], dtype=np.float64)
        # Score ALL centroids, then visit enough clusters to return the FULL budget.
        center_scores = self.centroids @ query
        order = np.lexsort((np.arange(len(center_scores)), -center_scores))
        selected: list[np.ndarray] = []
        eligible = 0
        visited = 0
        probes = 0
        for cluster_id in order:
            positions = self.members[int(cluster_id)]
            selected.append(positions)
            visited += len(positions)
            eligible += int(np.count_nonzero(~np.isin(self.catalog[positions], list(seen))))
            probes += 1
            if probes >= nprobe and eligible >= budget:
                break
        positions = np.concatenate(selected)
        ids = self.catalog[positions]
        keep = ~np.isin(ids, list(seen))
        ids, positions = ids[keep], positions[keep]
        scores = np.asarray(self.vectors[positions] @ query)
        ix = np.lexsort((ids, -scores))[:budget]
        result = np.asarray(ids[ix], dtype=np.int64)
        if len(result) < min(budget, int(np.count_nonzero(~np.isin(self.catalog, list(seen))))):
            raise RuntimeError('failed to fill candidate budget despite scanning all clusters')
        return result, {'probes_actual': probes, 'item_dot_products': visited,
                        'centroid_dot_products': self.n_clusters,
                        'total_dot_products': visited + self.n_clusters}


def rerank(model: ExactDiagnostic, user: int, ids: np.ndarray, top: int = FINAL_TOP) -> list[int]:
    state = model.state
    if not len(ids):
        return []
    # Identical feature formula to src.exact_diagnostic.ExactDiagnostic.recommend.
    features = np.column_stack((
        state['item_vectors'][ids] @ state['user_vectors'][user],
        np.log1p(state['popularity'][ids]),
        state['interaction'][user] @ state['similarity'][:, ids],
        np.full(len(ids), np.log1p(state['user_counts'][user])),
        np.linalg.norm(state['item_vectors'][ids], axis=1),
    )).astype(np.float32)
    scores = model.ranker.predict(features, num_threads=1)
    if not np.isfinite(scores).all():
        raise ValueError('nonfinite ranker prediction')
    return [int(ids[i]) for i in np.lexsort((ids, -scores))[:top]]


def percentile(numbers: Iterable[float], fraction: float) -> float:
    return float(np.percentile(list(numbers), fraction))


def audit(seeds=(7, 11, 19), queries=80, probes=DEFAULT_PROBES):
    outcome = []
    for seed in seeds:
        model_dir = ROOT / 'models' / f'seed{seed}'
        model = ExactDiagnostic.load(model_dir)
        idx = IVFDiagnostic(model, N_CLUSTERS, random_state=20261016)
        nusers = len(model.state['user_vectors']) - 1
        # Query identities chosen before comparing methods: no quality-dependent selection.
        userids = list(range(1, min(queries, nusers) + 1))
        exact_candidates = {}
        exact_ranked = {}
        exact_times = []
        for user in userids:
            start = time.perf_counter_ns()
            ids = model._candidates(model.state['user_vectors'][user], model.state['item_vectors'],
                                    DEFAULT_SEEN, RETRIEVAL_BUDGET)
            exact_times.append((time.perf_counter_ns()-start)/1e6)
            exact_candidates[user] = [int(t) for t in ids]
            exact_ranked[user] = rerank(model, user, ids)
            # Check independent oracle equals actual existing production diagnostic call.
            canonical = model.recommend(user, DEFAULT_SEEN, RETRIEVAL_BUDGET, top=FINAL_TOP)
            assert exact_ranked[user] == [x['item'] for x in canonical['items']], user

        cases = []
        for nprobe in probes:
            overlaps, ranked_agreement, top_rank_exact, works, actual_probes, times = [], [], [], [], [], []
            for user in userids:
                start = time.perf_counter_ns()
                approximate, stats = idx.search(user, DEFAULT_SEEN, RETRIEVAL_BUDGET, nprobe=nprobe)
                times.append((time.perf_counter_ns()-start)/1e6)
                approx_list = [int(t) for t in approximate]
                baseline = exact_candidates[user]
                if len(approx_list) != RETRIEVAL_BUDGET:
                    raise AssertionError('candidate budget mismatch')
                if len(set(approx_list)) != len(approx_list) or set(approx_list)&DEFAULT_SEEN:
                    raise AssertionError('invalid candidates')
                overlaps.append(len(set(approx_list)&set(baseline))/RETRIEVAL_BUDGET)
                ranked = rerank(model, user, approximate)
                ranked_agreement.append(len(set(ranked)&set(exact_ranked[user]))/FINAL_TOP)
                top_rank_exact.append(ranked == exact_ranked[user])
                works.append(stats['total_dot_products'])
                actual_probes.append(stats['probes_actual'])
            cases.append({
                'minimum_nprobe': int(nprobe), 'queries': len(userids),
                'mean_candidate_recall_at_50_vs_exact': statistics.mean(overlaps),
                'mean_reranked_overlap_at_10_vs_exact': statistics.mean(ranked_agreement),
                'identical_rerank_top10_query_count': sum(top_rank_exact),
                'mean_dot_products_including_32_centers': statistics.mean(works),
                'mean_actual_probed_clusters': statistics.mean(actual_probes),
                'query_local_ms_p50': percentile(times,50),
                'query_local_ms_p95': percentile(times,95),
            })
        outcome.append({
            'model_seed': seed, 'original_model_input_sha256': model.manifest['input_sha256'],
            'original_model_manifest_sha256': hashlib.sha256((model_dir/'model.json').read_bytes()).hexdigest(),
            'model_schema': model.manifest['schema'],
            'catalog_items': len(model.catalog), 'query_count':len(userids),
            'training_cluster_build_ms': idx.build_ms,
            'exact_query_local_ms_p50': percentile(exact_times,50),
            'exact_query_local_ms_p95': percentile(exact_times,95),
            'exact_item_dot_products_per_query': len(model.catalog),
            'cases': cases,
        })
    return {'status':'PASS_RETRIEVAL_FIDELITY_ON_SAVED_EMBEDDINGS',
            'timestamp_utc':datetime.now(timezone.utc).isoformat(),
            'provenance':'three saved real MovieLens-trained v2 embeddings; no original ratings supplied',
            'competitors':['full-catalog exact dot-product','local sklearn KMeans IVF clustered dot-product'],
            'fixed_parameters':{'cluster_count':N_CLUSTERS,'kmeans_seed':20261016,'kmeans_n_init':2,
               'retrieval_budget':RETRIEVAL_BUDGET,'rerank_top':FINAL_TOP,
               'synthetic_seen_item_ids':sorted(DEFAULT_SEEN),'query_selection':'user ids 1..N fixed, no post-hoc selection'},
            'scope_limits':['not FAISS, not HNSW','candidate recall measured against exact retrieval, NOT relevance',
               'rerank agreement != NDCG','missing MovieLens original ratings, no unbiased business relevance metric',
               'both methods share pretrained embeddings and original LightGBM ranker',
               'build time reported separately; query time Python local only',
               'KMeans index built on full training-era item catalog; no held-out label access',
               'approximate candidate search uses adaptive probe count to preserve fixed budget',
               'no claim of superiority over FAISS ANN or production latency'],
            'models':outcome}


if __name__=='__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--queries',type=int,default=80)
    parser.add_argument('--output',type=Path,default=ROOT/'evidence'/'r21_saved_embedding_ivf_comparison.json')
    args=parser.parse_args()
    result=audit(queries=args.queries)
    args.output.parent.mkdir(parents=True,exist_ok=True)
    args.output.write_text(json.dumps(result,indent=2,ensure_ascii=False)+'\n',encoding='utf-8')
    for model in result['models']:
        print(f"seed={model['model_seed']}, users={model['query_count']}, build_ms={model['training_cluster_build_ms']:.1f}")
        for row in model['cases']:
            print(f"  nprobe>={row['minimum_nprobe']}: candidate={row['mean_candidate_recall_at_50_vs_exact']:.3f} rerank@10={row['mean_reranked_overlap_at_10_vs_exact']:.3f} p95_ms={row['query_local_ms_p95']:.2f}")
