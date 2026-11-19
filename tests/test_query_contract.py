import sys
from pathlib import Path
import numpy as np
import pytest
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
from src.retrieval import ClusterIndex

def make_index():
    i=ClusterIndex.__new__(ClusterIndex)
    i.v=np.asarray([[1.,0.],[0.,1.],[-1.,0.]])
    i.centers=np.asarray([[.5,.5],[-1.,0.]])
    i.members=[np.asarray([0,1]),np.asarray([2])]
    i.radii=np.asarray([.8,0.])
    return i

@pytest.mark.parametrize('q,k,nprobe',[
    ([1.,2.,3.],1,1),([float('nan'),0.],1,1),([1.,0.],0,1),
    ([1.,0.],4,1),([1.,0.],2,0),([1.,0.],2,3),([1.,0.],True,1)
])
def test_bad_query_fails_closed(q,k,nprobe):
    with pytest.raises(ValueError):make_index().search(np.asarray(q),k,nprobe)

def test_valid_retrieval_keeps_existing_contract():
    result,visited,clusters=make_index().search(np.array([1.,0.]),k=2,nprobe=2,certify=True)
    assert len(result)==2 and set(result)=={0,1}
    assert visited==2 and clusters==1
