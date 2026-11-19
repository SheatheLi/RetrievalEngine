import numpy as np
from src.retrieval import *

def test_temporal_split():
    d=events(users=12,items=70)
    assert d.train[:,2].max()<d.test[:,2].min()
    assert not set(map(tuple,d.train[:,:2])) & set(map(tuple,d.test[:,:2]))

def test_certified_exactness():
    rng=np.random.default_rng(3);v=rng.normal(size=(100,8));v/=np.linalg.norm(v,axis=1)[:,None]
    ix=ClusterIndex(v,ncluster=12)
    for q in rng.normal(size=(10,8)):
        true=set(np.argsort(-(v@q))[:5]);actual=set(ix.search(q,k=5,certify=True)[0]);assert true==actual

def test_model_train_and_eval():
    d=events(users=20,items=100);u,v=train(d,epochs=2)
    out=evaluate(d,u,v,k=5)
    assert out['certified_recall_at_k']==1.0
    assert out['mean_dot_products_ivf']<=d.items
