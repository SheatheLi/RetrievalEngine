"""PyTorch training + Dynamo compilation-guard experiment on variable sequence length.

Uses PyTorch FX/Dynamo backend='eager'; no CUDA/Inductor/GPU-kernel claim.
"""
from collections import Counter
from dataclasses import dataclass
import torch
from torch import nn

class SelfAttentionBlock(nn.Module):
    """Real multi-head scaled-dot-product attention + residual/feedforward."""
    def __init__(self,d=24,heads=3):
        super().__init__();self.h=heads;self.ln=nn.LayerNorm(d)
        self.qkv=nn.Linear(d,3*d);self.out=nn.Linear(d,d)
        self.ff=nn.Sequential(nn.LayerNorm(d),nn.Linear(d,4*d),nn.GELU(),nn.Linear(4*d,d))
    def forward(self,x):
        b,t,d=x.shape
        q,k,v=self.qkv(self.ln(x)).view(b,t,3,self.h,d//self.h).permute(2,0,3,1,4).unbind(0)
        a=torch.nn.functional.scaled_dot_product_attention(q,k,v)
        x=x+self.out(a.transpose(1,2).contiguous().view(b,t,d))
        return x+self.ff(x)

class SequenceRegressor(nn.Module):
    def __init__(self, d=24, hidden=40):
        super().__init__()
        self.encoder=nn.Sequential(SelfAttentionBlock(d),SelfAttentionBlock(d))
        self.head=nn.Linear(d,1)
    def forward(self,x):
        z=self.encoder(x)
        return self.head(z.mean(dim=1)).flatten()

@dataclass
class Run:
    graphs: int
    graph_count_by_shape: dict
    max_abs_error: float
    lengths: list
    losses: list

def run(lengths=(16,24,32,24,16),dynamic=False,seed=7):
    torch._dynamo.reset()
    torch.manual_seed(seed)
    m=SequenceRegressor().eval()
    graphs=[]
    def backend(gm,example_inputs):
        graphs.append(gm)
        return gm.forward
    fn=torch.compile(m,backend=backend,dynamic=dynamic,fullgraph=True)
    err=0.0; losses=[]; counts={}
    with torch.no_grad():
        for length in lengths:
            x=torch.randn(2,length,24)
            ref=m(x)
            got=fn(x)
            err=max(err,(ref-got).abs().max().item())
            losses.append(got.sum().item())
            counts[str(length)]=len(graphs)
    return Run(len(graphs),counts,err,list(lengths),losses)

def train_small(steps=6,seed=1):
    """Genuine autograd/AdamW training, disjoint from Dynamo guard study."""
    torch.manual_seed(seed)
    model=SequenceRegressor();opt=torch.optim.AdamW(model.parameters(),lr=.004)
    x=torch.randn(24,12,24);y=x.square().mean((1,2))
    first=None
    for _ in range(steps):
        opt.zero_grad(set_to_none=True)
        loss=(model(x)-y).square().mean()
        if first is None:first=float(loss.item())
        loss.backward();opt.step()
    return dict(initial=first,final=float(loss.item()))
