import copy,pytest
from src.evaluation_protocol import report,paired_difference,EvaluationError

def q(qid='q',ranked=None,candidates=None):
    return dict(query_id=qid,relevant=['gold','rare'],relevance_source='human-reviewed-v1',
        candidates=['gold','other'] if candidates is None else candidates,
        ranked=['other','gold'] if ranked is None else ranked,latency_ms=4.5)

def test_stage_specific_loss_is_not_hallucinated():
    r=report([q()],k=2,iterations=101)
    x=r['per_query'][0]
    assert x['candidate_recall']==.5
    assert x['recall_at_k']==.5
    assert x['relevant_candidates_lost']==('rare',)
    assert x['relevant_ranking_lost']==()
    assert 0<x['ndcg_at_k']<1

def test_paired_identical_is_zero_not_claimed_improvement():
    b=[q('a'),q('b')];c=copy.deepcopy(b)
    r=paired_difference(b,c,k=2,iterations=101)
    assert r['summary']['ndcg_delta']['mean_delta']==0
    assert r['summary']['latency_delta_ms']['mean_delta']==0

def test_pairing_refuses_label_drift():
    before=[q()];after=copy.deepcopy(before)
    after[0]['relevant']=['other']
    with pytest.raises(EvaluationError):paired_difference(before,after,iterations=101)

@pytest.mark.parametrize('change',[
    lambda x:x.update(relevance_source='same_model'),
    lambda x:x.update(relevant=[]),
    lambda x:x.update(candidates=['gold','gold']),
    lambda x:x.update(ranked=['unknown']),
    lambda x:x.update(latency_ms=float('nan')),
    lambda x:x.update(latency_ms=-1),
])
def test_reject_unsupported_input(change):
    row=q();change(row)
    with pytest.raises(EvaluationError):report([row],iterations=101)
