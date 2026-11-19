"""Repeatable exact-only scoring audit of THREE historical saved model artifacts.

No ratings, FAISS, ANN comparisons, online traffic, or relevance metrics available.
"""
from __future__ import annotations
import json
from pathlib import Path
import sys
import time
import numpy as np

ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT))
from src.exact_diagnostic import ExactDiagnostic


def audit():
    results=[]
    for seed in (7,11,19):
        folder=ROOT/'models'/f'seed{seed}'
        model=ExactDiagnostic.load(folder)
        times=[]
        for user in range(1,49):
            seen={1,2,3,4,5,9}
            before=time.perf_counter_ns()
            result=model.recommend(user,seen,50)
            elapsed=(time.perf_counter_ns()-before)/1e6
            ids=[r['item'] for r in result['items']]
            assert len(ids)==10 and len(set(ids))==10 and not (set(ids)&seen)
            assert not result['approximate'] and result['backend']=='numpy_exact_no_faiss'
            assert result==model.recommend(user,seen,50)
            times.append(elapsed)
        results.append({'seed':seed,'model_schema':model.manifest['schema'],
            'queries':len(times),'budget':50,'top':10,'seen_filter':'SYNTHETIC_CONSTANT_NOT_MOVIELENS_PREFIX',
            'latency_ms':{'p50':float(np.percentile(times,50)),
                          'p95':float(np.percentile(times,95)),
                          'p99':float(np.percentile(times,99))},
            'validated':'saved_artifact_hashes+deterministic_exact_ranking+unique+seen_filter',
            'model_input_sha256':model.manifest['input_sha256']})
    return {'status':'PASS_EXACT_ONLY_HISTORICAL_ARTIFACT_AUDIT',
            'bounds':['FAISS_HNSW=NOT_RUN','NO_ORIGINAL_MOVIELENS_RATINGS_IN_PACKAGE',
                      'NOT_A_SERVICE_LATENCY_SLA','TIMING_LOCAL_CPU_ONLY',
                      'NO_GROUNDED_NDCG_OR_RECALL_MEASURED'],
            'models':results}


if __name__=='__main__':
    a=audit()
    p=ROOT/'evidence'/'r20_three_historical_models.json'
    p.write_text(json.dumps(a,indent=2,ensure_ascii=False)+'\n')
    print(a['status'],[(m['seed'],round(m['latency_ms']['p95'],2)) for m in a['models']])
