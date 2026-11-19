"""Deterministic exact maximum-inner-product top-k via lossy quantization bounds.

Technique: asymmetric scalar quantization + Cauchy-Schwarz residual bound +
branch-and-bound verification. Unlike ANN recall claims, certificate is computed
against the *original* floating-point vectors; coarse scores alone are unsafe.
This prototype still evaluates ALL quantized scores (O(nd)), so fewer full
precision dot products does NOT automatically mean lower wall time.
"""
from __future__ import annotations
from dataclasses import dataclass
import time
import numpy as np

@dataclass(frozen=True)
class SearchResult:
    indices: np.ndarray
    full_precision_scores: int
    coarse_scores: int
    certificate_margin: float

class CertifiedQuantizedIndex:
    def __init__(self, vectors: np.ndarray, bits: int = 8):
        v=np.asarray(vectors, dtype=np.float64)
        if v.ndim!=2 or len(v)==0 or not np.all(np.isfinite(v)):
            raise ValueError('finite nonempty 2d index required')
        if bits not in (4, 8): raise ValueError('supported bits: 4 or 8')
        self.v=v.copy(); self.bits=bits
        levels=2**bits - 1
        self.min=v.min(axis=0); span=v.max(axis=0)-self.min
        self.step=np.where(span>0, span/levels, 1.)
        self.codes=np.clip(np.rint((v-self.min)/self.step), 0, levels).astype(np.uint8)
        self.reconstructed=self.min + self.codes.astype(np.float64)*self.step
        self.residual_norm=np.linalg.norm(self.v-self.reconstructed,axis=1)
        # Numeric tolerance is conservative in score units for double precision.
        self.eps=1.e-11
    def search(self, query:np.ndarray, k:int=10)->SearchResult:
        q=np.asarray(query,dtype=np.float64)
        if q.shape!=(self.v.shape[1],) or not np.all(np.isfinite(q)):raise ValueError('query shape/finite mismatch')
        if not 1<=k<=len(self.v):raise ValueError('invalid k')
        # Vectorized lower/upper certificates avoid a Python score-and-sort
        # loop for every provisional candidate.  The kth largest *lower*
        # bound is conservative: the true kth largest score is at least this
        # value, so any upper bound below it can safely be excluded.
        approx=self.reconstructed@q
        radius=np.linalg.norm(q)*self.residual_norm+self.eps
        lower=approx-radius
        upper=approx+radius
        kth_lower=float(np.partition(lower, len(lower)-k)[len(lower)-k])
        eligible=np.flatnonzero(upper >= kth_lower-self.eps)
        # Evaluate each eligible original vector exactly once.  It is not
        # sufficient to sort the lossy quantized scores by themselves.
        scores=self.v[eligible]@q
        ordered=np.lexsort((eligible, -scores))
        top=eligible[ordered[:k]]
        omitted=upper < kth_lower-self.eps
        if np.any(omitted):
            kth_score=float(scores[ordered[k-1]])
            margin=float(kth_score-np.max(upper[omitted]))
        else:
            margin=float('-inf')
        return SearchResult(top.astype(np.int64),len(eligible),len(self.v),margin)

def exact_mips(v,q,k):
    scores=np.asarray(v,dtype=float)@np.asarray(q,dtype=float)
    return np.lexsort((np.arange(len(scores)),-scores))[:k]

def evaluate_digits():
    from sklearn.datasets import load_digits
    from sklearn.decomposition import PCA
    from src.retrieval import ClusterIndex
    x,_=load_digits(return_X_y=True)
    projection=PCA(n_components=24,random_state=2).fit_transform(x)
    emb=projection/(np.linalg.norm(projection,axis=1,keepdims=True)+1.e-12)
    base=emb[:1400].astype(np.float64); queries=emb[1400:].astype(np.float64)
    cindex=CertifiedQuantizedIndex(base,bits=8)
    baseline=ClusterIndex(base,ncluster=32,seed=2)
    counts=[];correct=[]; old_counts=[]
    ts=time.perf_counter()
    for q in queries:
        ans=cindex.search(q,10)
        truth=exact_mips(base,q,10)
        correct.append(bool(np.array_equal(ans.indices,truth)))
        counts.append(ans.full_precision_scores)
    elapsed=time.perf_counter()-ts
    for q in queries:
        _,n,_=baseline.search(q,10,certify=True);old_counts.append(n)
    # Baseline exact vectorized matmul time (not identical algorithmic work).
    ts=time.perf_counter()
    for q in queries:exact_mips(base,q,10)
    brute_elapsed=time.perf_counter()-ts
    return {'dataset':'sklearn_digits_PCA24_proxy_not_recommendation', 'items':len(base),'queries':len(queries),
        'exact_top10_matches':sum(correct),'old_cluster_exact_dot_mean':float(np.mean(old_counts)),
        'quantized_bound_exact_dot_mean':float(np.mean(counts)),
        'quantized_all_coarse_dot_count_per_query':len(base),
        'quantized_full_precision_ratio':float(np.mean(counts)/len(base)),
        'quantized_python_elapsed_s':elapsed,'numpy_full_scan_elapsed_s':brute_elapsed,
        'quantized_code_bytes':int(cindex.codes.nbytes), 'original_float64_bytes':int(base.nbytes),
        'warning':'all coarse dot-products and sorting remain; do not claim speedup from rescoring reduction'}
if __name__=='__main__':
    import json
    print(json.dumps(evaluate_digits(),indent=2))
