"""Value-preserving shape-bucket policy for variable-length attention workloads.

Motivated by PyTorch #190631. Experiments use real torch.compile and real
scaled-dot-product multi-head attention on CPU; this is not a PyTorch patch,
nor a CUDA Graph/Inductor optimization.
"""
from __future__ import annotations
from dataclasses import dataclass,asdict
import time
import torch
from torch import nn
from torch.nn import functional as F

class MaskedAttention(nn.Module):
    def __init__(self,d=24,heads=3):
        super().__init__();self.heads=heads;self.d=d
        self.proj=nn.Linear(d,d*3);self.o=nn.Linear(d,d)
        self.ln=nn.LayerNorm(d)
        self.ff=nn.Sequential(nn.Linear(d,d*2),nn.GELU(),nn.Linear(d*2,d))
    def forward(self,x,valid):
        b,t,d=x.shape
        q,k,v=self.proj(self.ln(x)).reshape(b,t,3,self.heads,d//self.heads).permute(2,0,3,1,4).unbind(0)
        attention=F.scaled_dot_product_attention(q,k,v,attn_mask=valid[:,None,None,:])
        x=x+self.o(attention.transpose(1,2).reshape(b,t,d))
        x=x+self.ff(self.ln(x))
        return (x*valid[:,:,None]).sum(dim=1)/valid.sum(dim=1).clamp(min=1)[:,None]

def next_bucket(n:int,buckets:tuple[int,...])->int:
    if n <= 0 or sorted(set(buckets)) != list(buckets): raise ValueError('invalid length/buckets')
    for k in buckets:
        if k>=n:return k
    raise ValueError('no bucket for input length')

def batch(x:torch.Tensor,length:int)->tuple[torch.Tensor,torch.Tensor]:
    b,n,d=x.shape
    if length<n: raise ValueError('bucket smaller than sequence')
    pad=length-n
    padded=F.pad(x,(0,0,0,pad))
    valid=torch.arange(length,device=x.device)[None,:]<n
    return padded,valid.expand(b,-1)

@dataclass
class Result:
    eager_graphs:int
    bucket_graphs:int
    num_lengths:int
    max_error:float
    total_true_tokens:int
    total_bucket_tokens:int
    exact_execution_ms:float
    bucket_execution_ms:float

def experiment(lengths=(5,7,9,13,15,17,25,31,29,13),seed=7)->Result:
    torch.manual_seed(seed)
    model=MaskedAttention().eval()
    tensors=[torch.randn(2,n,24) for n in lengths]
    def compile_counter():
        g=[]
        def backend(gm, inputs):
            g.append(gm)
            return gm.forward
        return torch.compile(model,backend=backend,fullgraph=True,dynamic=False),g
    torch.compiler.reset()
    # Allow the unbucketed control to expose its complete specialization count.
    # The default guard limit raises after eight distinct lengths.
    torch._dynamo.config.recompile_limit=32
    exact_fn,exact_graphs=compile_counter()
    with torch.no_grad():
        exact_inputs=[batch(x,x.shape[1]) for x in tensors]
        exact_out=[exact_fn(a,v) for a,v in exact_inputs]
        t=time.perf_counter()
        for a,v in exact_inputs: exact_fn(a,v)
        exact_t=(time.perf_counter()-t)*1000
    exact_count=len(exact_graphs)
    torch.compiler.reset()
    bucket_fn,bucket_graphs=compile_counter()
    with torch.no_grad():
        bucket_inputs=[batch(x,next_bucket(x.shape[1],(8,16,32,64))) for x in tensors]
        bucket_out=[bucket_fn(a,v) for a,v in bucket_inputs]
        t=time.perf_counter()
        for a,v in bucket_inputs:bucket_fn(a,v)
        bucket_t=(time.perf_counter()-t)*1000
    error=max(float((a-b).abs().max()) for a,b in zip(exact_out,bucket_out))
    return Result(exact_count,len(bucket_graphs),len(lengths),error,
                  sum(x.shape[1]*2 for x in tensors),sum(x.shape[1]*2 for x,_ in bucket_inputs),exact_t,bucket_t)

if __name__=='__main__':
    import json
    print(json.dumps(asdict(experiment()),indent=2))
