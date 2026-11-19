from __future__ import annotations
from pathlib import Path
import sys
import numpy as np
import pytest

ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT))
from competition.ivf_vs_exact import IVFDiagnostic, audit, rerank
from src.exact_diagnostic import ExactDiagnostic


def test_saved_embedding_comparator_exact_oracle_real_model():
    model=ExactDiagnostic.load(ROOT/'models'/'seed7')
    idx=IVFDiagnostic(model,16,random_state=4)
    for user in (1,8,21):
        seen={1,2,3}
        exact=model._candidates(model.state['user_vectors'][user],model.state['item_vectors'],seen,50)
        approx,stats=idx.search(user,seen,50,nprobe=16)
        assert np.array_equal(exact,approx)
        assert stats['probes_actual']==16
        assert rerank(model,user,approx)==[v['item'] for v in model.recommend(user,seen,50)['items']]


def test_reject_invalid_probe_and_budget():
    model=ExactDiagnostic.load(ROOT/'models'/'seed7')
    idx=IVFDiagnostic(model,8,random_state=4)
    for probe in (0,9,-1):
        with pytest.raises(ValueError): idx.search(1,set(),50,nprobe=probe)
    for budget in (0,len(model.catalog)+1):
        with pytest.raises(ValueError): idx.search(1,set(),budget,nprobe=1)


def test_fixed_full_budget_even_when_one_cluster_too_small():
    model=ExactDiagnostic.load(ROOT/'models'/'seed7')
    idx=IVFDiagnostic(model,32,random_state=4)
    candidate,stats=idx.search(1,{1,2,3},50,nprobe=1)
    assert len(candidate)==50
    assert len(set(candidate))==50
    assert stats['probes_actual']>=1


def test_original_saved_models_no_relevance_claim():
    result=audit(seeds=(7,), queries=3,probes=(1,4,32))
    assert result['models'][0]['cases'][-1]['mean_candidate_recall_at_50_vs_exact']==1.0
    assert result['models'][0]['cases'][-1]['mean_reranked_overlap_at_10_vs_exact']==1.0
    assert any('NOT relevance' in x for x in result['scope_limits'])


def test_adaptive_budget_with_heavy_seen_filter():
    model=ExactDiagnostic.load(ROOT/'models'/'seed11')
    idx=IVFDiagnostic(model,32,random_state=9)
    allowed={int(t) for t in model.catalog[-50:]}
    seen={int(t) for t in model.catalog if int(t) not in allowed}
    found,stats=idx.search(19,seen,50,nprobe=1)
    assert set(found)==allowed
    assert stats['probes_actual']>1


def test_no_invented_relevance_metric():
    r=audit(seeds=(11,),queries=2,probes=(1,32))
    assert len(r['models'])==1
    model=r['models'][0]
    assert model['cases'][-1]['mean_candidate_recall_at_50_vs_exact']==1.0
    assert model['cases'][-1]['mean_reranked_overlap_at_10_vs_exact']==1.0
    assert 'no original ratings supplied' in r['provenance']
