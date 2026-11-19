from src.scatter_compat import matrix,scatter_out_compatible
import torch

def test_pytorch_issue_compat_matrix():
    rows=matrix()
    assert len(rows)==12
    assert all(row['equal'] and row['preserves_out'] for row in rows)

def test_reject_unsupported_operation():
    import pytest
    x=torch.zeros(1,2);i=torch.zeros(1,2,dtype=torch.long)
    with pytest.raises(ValueError):scatter_out_compatible(x,i,x,x,'invalid')
