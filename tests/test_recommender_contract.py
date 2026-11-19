from pathlib import Path
import numpy as np
import lightgbm as lgb
import pytest
pytest.importorskip("faiss", reason="native FAISS unavailable; run exact diagnostic separately")
from src.movielens_data import load
from src.recommender import Engine, metrics


def example_engine():
    vectors=np.asarray([[0,0],[1,0],[.8,.2],[.4,.6],[0,1]],dtype=np.float32)
    ranker=lgb.train({'objective':'regression','verbosity':-1,'num_threads':1,'min_data_in_leaf':1,
                     'min_data_in_bin':1},lgb.Dataset(np.arange(50,dtype=float).reshape(10,5),
                                                     label=np.arange(10)),num_boost_round=2)
    return Engine(np.asarray([[0,0],[1,0]],dtype=np.float32), vectors,np.arange(1,5),
                  np.ones(5),np.ones(2),np.zeros((2,5),dtype=np.float32),np.zeros((5,5),dtype=np.float32),ranker)


def test_temporal_split_keeps_equal_timestamps_on_same_side(tmp_path):
    rows=[]
    for timestamp in range(40):
        for user in range(1,4): rows.append([user, timestamp % 20 + 1, 4, timestamp])
    path=tmp_path/'ratings.tsv';np.savetxt(path,rows,fmt='%d',delimiter='\t')
    data=load(path)
    stages=[data.train,data.rank_train,data.validation,data.test]
    assert sum(map(len,stages))==len(rows)
    for left,right in zip(stages,stages[1:]):assert left[:,3].max()<right[:,3].min()
    assert set(data.catalog)==set(data.train[:,1])


def test_retrieval_respects_consumed_items_and_exhaustion():
    engine=example_engine()
    assert engine.retrieve(1,{1},2,False).tolist()==[2,3]
    assert set(engine.retrieve(1,{1},2,True)).isdisjoint({1})
    ids,scores=engine.recommend(1,{1,2,3,4},budget=4)
    assert len(ids)==len(scores)==0


def test_saved_model_prediction_and_corruption_rejection(tmp_path):
    engine=example_engine()
    expected=engine.recommend(1,{1},budget=3)
    engine.save(tmp_path,{'purpose':'contract test'})
    restored=Engine.load(tmp_path)
    actual=restored.recommend(1,{1},budget=3)
    np.testing.assert_array_equal(actual[0],expected[0])
    np.testing.assert_allclose(actual[1],expected[1],rtol=0,atol=0)
    p=tmp_path/'ranker.txt';p.write_bytes(p.read_bytes()+b'\n')
    with pytest.raises(ValueError,match='version mismatch'):Engine.load(tmp_path)


def test_heldout_metrics_are_not_self_generated_score_labels():
    recall,ndcg=metrics(np.asarray([3,5,7]),{5,7,9})
    assert recall==pytest.approx(2/3)
    assert ndcg==pytest.approx((1/np.log2(3)+1/np.log2(4))/(1+1/np.log2(3)+1/np.log2(4)))
    with pytest.raises(ValueError):metrics(np.asarray([1]),set())
