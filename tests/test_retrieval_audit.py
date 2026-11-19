import numpy as np
import pytest
from src.retrieval_audit import FrozenCatalog, QueryCase, exact_topk, audit_result, summarize, paired_neighbor_delta


def fixture():
    x=FrozenCatalog(np.array([50, 10, 30, 20]), np.array([[1.,0.],[1.,0.],[0.,1.],[-1.,0.]]), 'vector-v1')
    q=QueryCase('query-1',np.array([1.,0.]),frozenset(),frozenset({10,30}))
    return x,q


def test_exact_business_id_tie_break():
    x,q=fixture()
    assert exact_topk(x,q,3)==(10,50,30)


def test_seen_filter_both_truth_and_result():
    x,q=fixture()
    q=QueryCase('q2',q.vector,frozenset({10,50}),frozenset({30}))
    assert exact_topk(x,q,2)==(30,20)
    r=audit_result(x,q,2,[30,20],'exact',12)
    assert r.neighbor_recall==1 and r.relevant_recall==1


def test_ann_fidelity_not_equal_relevance():
    x,q=fixture()
    r=audit_result(x,q,2,[10,50],'ann',20)
    assert r.neighbor_recall==1 and r.relevant_recall==.5


def test_no_future_positive_support_is_explicit():
    x,q=fixture()
    q=QueryCase('q3',q.vector,frozenset(),frozenset({999}))
    assert audit_result(x,q,2,[10,50],'exact',20).relevant_recall is None

@pytest.mark.parametrize('ranking',[[10,10],[999,50],[10],[10,50,30],[50,10.0]])
def test_reject_bad_rankings(ranking):
    x,q=fixture()
    with pytest.raises(ValueError): audit_result(x,q,2,ranking,'ann',1)


def test_all_seen_allows_empty_result():
    x,q=fixture()
    q=QueryCase('q4',q.vector,frozenset({10,20,30,50}),frozenset({30}))
    r=audit_result(x,q,2,[],'ann',1)
    assert r.neighbor_recall is None and r.relevant_recall is None


def test_catalog_copies_are_immutable():
    vec=np.array([[1.],[2.]])
    snap=FrozenCatalog(np.array([11,13]),vec,'v')
    vec[0,0]=9.
    assert snap.vectors[0,0] == 1.
    with pytest.raises(ValueError): snap.vectors[0,0]=5.


def test_dimension_and_overflow_failure():
    x,q=fixture()
    with pytest.raises(ValueError): exact_topk(x,QueryCase('bad',np.array([1,2,3]),frozenset(),frozenset()),2)
    extreme=FrozenCatalog(np.array([1,2]),np.array([[1e308],[1e308]]),'v')
    with pytest.raises((FloatingPointError,ValueError)):
        exact_topk(extreme,QueryCase('huge',np.array([1e308]),frozenset(),frozenset()),1)


def test_reject_duplicate_aggregate_query_and_mixed_method():
    x,q=fixture()
    a=audit_result(x,q,2,[10,50],'exact',100)
    b=audit_result(x,q,2,[50,10],'ann',200)
    with pytest.raises(ValueError): summarize([a,a])
    with pytest.raises(ValueError): summarize([a,b])


def test_paired_comparison_requires_same_ground_truth_and_cohort():
    x,q=fixture()
    a=audit_result(x,q,2,[10,50],'exact',10)
    b=audit_result(x,q,2,[50,30],'ann',20)
    outcome=paired_neighbor_delta([a],[b])
    assert outcome['a_minus_b_mean']==.5
    with pytest.raises(ValueError): paired_neighbor_delta([a],[])


def test_summary_boundaries_and_metric_name():
    x,q=fixture()
    r=audit_result(x,q,2,[10,50],'ann',1_500_000)
    s=summarize([r])
    assert s['mean_neighbor_recall']==1
    assert s['p95_ms']==1.5
    assert 'online CTR' in s['not_proven']


def test_no_eligible_neighbors_are_not_perfect_recall():
    x,q=fixture()
    q=QueryCase('none',q.vector,frozenset({10,20,30,50}),frozenset())
    r=audit_result(x,q,2,[],'ann',12)
    stats=summarize([r])
    assert stats['mean_neighbor_recall'] is None
    assert stats['neighbor_evaluable_queries']==0
    with pytest.raises(ValueError): paired_neighbor_delta([r],[r])
