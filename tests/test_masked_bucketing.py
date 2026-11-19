import torch
from src.masked_bucketing import MaskedAttention,batch,next_bucket

def test_masked_pad_equivalence():
    torch.manual_seed(13)
    m=MaskedAttention().eval()
    x=torch.randn(2,5,24)
    with torch.no_grad():
        a=m(*batch(x,5));b=m(*batch(x,8));c=m(*batch(x,16))
    assert torch.allclose(a,b,rtol=1e-5,atol=1e-5)
    assert torch.allclose(a,c,rtol=1e-5,atol=1e-5)

def test_no_silent_truncate():
    assert next_bucket(9,(8,16))==16
    try:next_bucket(17,(8,16))
    except ValueError:pass
    else:assert False

def test_gradient_ignores_padded_positions():
    m=MaskedAttention().eval()
    x=torch.randn(2,5,24,requires_grad=True)
    loss=m(*batch(x,8)).square().mean();loss.backward()
    assert x.grad is not None and torch.isfinite(x.grad).all()
