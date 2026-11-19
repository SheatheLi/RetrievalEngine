"""Native FAISS + ranker integration contracts (NOT RUN without pinned FAISS)."""
import json
import numpy as np
import pytest

pytest.importorskip('faiss', reason='native FAISS dependency unavailable')
from src.recommender import Engine


def example_engine():
    import lightgbm as lgb
    training = np.arange(50, dtype=np.float64).reshape(10, 5)
    ranker = lgb.train({'objective':'regression','verbosity':-1,'num_threads':1,
                        'min_data_in_leaf':1,'min_data_in_bin':1},
                       lgb.Dataset(training, label=np.arange(10)), num_boost_round=2)
    item = np.asarray([[0,0],[1,0],[.8,.2],[.4,.6],[0,1]], dtype=np.float32)
    history_item = np.asarray([[0,0],[1,0],[.8,.2],[.4,.6],[0,1]], dtype=np.float32)
    return Engine(np.zeros((4, 2), dtype=np.float32), item, np.arange(1, 5),
                  np.ones(5), np.ones(4), np.zeros((4, 5), dtype=np.float32),
                  np.zeros((5, 5), dtype=np.float32), ranker=ranker,
                  history_item_vectors=history_item,history_projection=np.eye(2,dtype=np.float32))


def test_history_only_user_exact_and_hnsw_never_return_consumed():
    engine = example_engine()
    exact,_ = engine.recommend(3,{1},3,'history_exact',history_positive=[1])
    near,_ = engine.recommend(3,{1},3,'history_hnsw',history_positive=[1])
    assert len(exact)==len(near)==3
    assert set(exact)==set(near)=={2,3,4}
    with pytest.raises(ValueError,match='timestamp-ordered'):
        engine.recommend(3,{1},3,'history_exact',history_positive={1})
    with pytest.raises(ValueError,match='no known positive history'):
        engine.recommend(3,{1},3,'history_exact',history_positive=[])


def test_v3_persistence_and_inductive_index_integrity(tmp_path):
    engine = example_engine()
    expected = engine.recommend(3,{1},3,'history_exact',history_positive=[1])
    manifest = engine.save(tmp_path,{'seed':81,'input_sha256':'test-fixture'})
    assert manifest['schema']=='two-stage-recommender.v3'
    assert 'history_hnsw.index' in manifest['sha256']
    actual = Engine.load(tmp_path).recommend(3,{1},3,'history_exact',history_positive=[1])
    np.testing.assert_array_equal(actual[0],expected[0])
    np.testing.assert_allclose(actual[1],expected[1])
    p=tmp_path/'history_hnsw.index';p.write_bytes(p.read_bytes()+b'tampered')
    with pytest.raises(ValueError,match='version mismatch'):
        Engine.load(tmp_path)
