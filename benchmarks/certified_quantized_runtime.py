"""Reproducible paired timing: legacy heap-like scan vs vectorized bound filter.

Both methods certify Top-K using the original float64 vectors. They are
measured against a NumPy dense full scan, not production ANN systems.
"""
from __future__ import annotations
import json
import os
import platform
import statistics
import time
from pathlib import Path
import numpy as np
from sklearn.datasets import load_digits
from sklearn.decomposition import PCA
from src.certified_quantized import CertifiedQuantizedIndex, exact_mips


def legacy(index: CertifiedQuantizedIndex, query: np.ndarray, k: int) -> tuple[np.ndarray, int]:
    approx=index.reconstructed@query
    upper=approx+np.linalg.norm(query)*index.residual_norm+index.eps
    order=np.lexsort((np.arange(len(upper)), -upper))
    candidate=[]
    count=0
    for ix in order:
        if len(candidate)>=k:
            rank=sorted(candidate,key=lambda x:(-x[0],x[1]))[:k]
            if upper[ix] < rank[-1][0]-index.eps:
                break
        score=float(index.v[ix]@query)
        count+=1
        candidate.append((score,int(ix)))
        if len(candidate)>2*k:
            candidate=sorted(candidate,key=lambda x:(-x[0],x[1]))[:k]
    rank=sorted(candidate,key=lambda x:(-x[0],x[1]))[:k]
    return np.array([i for _,i in rank],dtype=np.int64),count


def run(rounds: int = 7, k: int = 10, sample_queries: int = 397) -> dict:
    x,_=load_digits(return_X_y=True)
    z=PCA(n_components=24,random_state=2).fit_transform(x)
    z=z/(np.linalg.norm(z,axis=1,keepdims=True)+1e-12)
    base=z[:1400].astype(np.float64)
    queries=z[1400:1400+sample_queries].astype(np.float64)
    index=CertifiedQuantizedIndex(base, bits=8)
    methods={
        'previous':lambda q: legacy(index,q,k),
        'vectorized':lambda q: (lambda r:(r.indices, r.full_precision_scores))(index.search(q,k)),
        'dense_float64':lambda q:(exact_mips(base,q,k),len(base)),
    }
    truth=[exact_mips(base,q,k) for q in queries]
    # Warm each method before measuring to limit first-call overhead.
    for f in methods.values():
        for query in queries[:10]:f(query)
    timing={name:[] for name in methods};counts={name:[] for name in methods}
    rng=np.random.default_rng(47)
    for rnd in range(rounds):
        order=rng.permutation(len(queries))
        # Rotate order across rounds so no method always benefits from cache.
        names=list(methods)
        names=names[rnd%len(names):]+names[:rnd%len(names)]
        for name in names:
            start=time.perf_counter()
            total=0
            for i in order:
                result,n=methods[name](queries[i])
                if not np.array_equal(result,truth[i]):
                    raise AssertionError(f'{name} returned incorrect Top-K for query {i}')
                total+=n
            timing[name].append(time.perf_counter()-start)
            counts[name].append(total/len(queries))
    medians={key:float(statistics.median(val)) for key,val in timing.items()}
    return {
        'dataset':'sklearn_digits_PCA24_proxy_not_user_item_recommendation',
        'index_items':len(base),'queries':len(queries),'k':k,
        'rounds':rounds,'python':platform.python_version(),
        'numpy':np.__version__,'os':platform.platform(),
        'threads_env':{n:os.getenv(n) for n in ('OPENBLAS_NUM_THREADS','OMP_NUM_THREADS','MKL_NUM_THREADS')},
        'times_seconds_by_round':timing,'median_times_seconds':medians,
        'median_previous_over_vectorized':medians['previous']/medians['vectorized'],
        'median_dense_over_vectorized':medians['dense_float64']/medians['vectorized'],
        'mean_exact_scoring_count':{key:float(statistics.mean(val)) for key,val in counts.items()},
        'all_methods_match_dense_topk':True,
        'limitations':'Synthetic normalized handwritten-digit vectors. This does not establish ANN production throughput, hardware portability, or real recommender relevance.'
    }

if __name__=='__main__':
    res=run()
    target=Path('evidence')/'certified_quantized_vectorization_benchmark.json'
    target.parent.mkdir(parents=True,exist_ok=True)
    target.write_text(json.dumps(res,indent=2,ensure_ascii=False),encoding='utf8')
    print(json.dumps({k:res[k] for k in ('queries','median_times_seconds','median_previous_over_vectorized','median_dense_over_vectorized','mean_exact_scoring_count')},indent=2))
