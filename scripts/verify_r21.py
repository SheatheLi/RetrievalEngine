"""Reproduce R21 source tests and saved-model IVF fidelity after archive extraction."""
from __future__ import annotations
import json
import os
from pathlib import Path
import subprocess
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT))
from competition.ivf_vs_exact import audit


def main():
    env={**os.environ,'OPENBLAS_NUM_THREADS':'1','MKL_NUM_THREADS':'1','OMP_NUM_THREADS':'1'}
    run=subprocess.run([sys.executable,'-m','pytest','-q','tests','--disable-warnings'],cwd=ROOT,env=env,stdout=subprocess.PIPE,stderr=subprocess.STDOUT,text=True,timeout=100)
    print(run.stdout)
    if run.returncode:
        raise SystemExit(f'PYTEST_FAIL={run.returncode}')
    expected=json.loads((ROOT/'evidence'/'r21_saved_embedding_ivf_comparison.json').read_text())
    actual=audit(queries=80)
    fields=('mean_candidate_recall_at_50_vs_exact','mean_reranked_overlap_at_10_vs_exact',
            'identical_rerank_top10_query_count','mean_dot_products_including_32_centers',
            'mean_actual_probed_clusters')
    assert len(expected['models'])==len(actual['models'])==3
    for e,a in zip(expected['models'],actual['models']):
        assert e['model_seed']==a['model_seed']
        assert e['original_model_manifest_sha256']==a['original_model_manifest_sha256']
        assert len(e['cases'])==len(a['cases'])==6
        for c,d in zip(e['cases'],a['cases']):
            assert c['minimum_nprobe']==d['minimum_nprobe']
            for key in fields:
                assert abs(c[key]-d[key])<1e-9,(a['model_seed'],key,c[key],d[key])
    print('R21_VERIFIED: 3 original saved models; 240 fixed users; 1440 pairings; 6 budgets')
    print('NATIVE_FAISS: NOT RUN; UPSTREAM_ORIGINAL_FIXES: 0/8')


if __name__=='__main__':main()
