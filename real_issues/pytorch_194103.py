"""Native PyTorch reproducer for #194103 on the INSTALLED release.

Issue refers to a newer nightly, so a pass here is not proof that issue was fixed.
"""
from __future__ import annotations
import json
import torch

def run():
    def f(x,index,src,out):
        return torch.scatter(x,1,index,src,out=out)
    compiled=torch.compile(f,backend='eager',fullgraph=True,dynamic=True)
    records=[]
    for n in (2,3,5):
        x=torch.zeros(1,n)
        idx=torch.arange(n).unsqueeze(0);src=torch.ones(1,n)
        before=f(x,idx,src,torch.empty_like(x))
        try:
            after=compiled(x,idx,src,torch.empty_like(x))
            records.append({'width':n,'native_compiled':'pass','equal':torch.equal(before,after)})
        except Exception as exc:
            records.append({'width':n,'native_compiled':'fail','error_type':type(exc).__name__,'message':str(exc)[:250]})
    return {'torch':torch.__version__,'issue_target':'2.15.0.dev20260819','rows':records,
            'interpretation':'version-gap study; no upstream fix implemented'}

if __name__=='__main__':print(json.dumps(run(),indent=2))

def local_workaround():
    """Extra allocation+copy workaround; not an upstream ATen/meta fix."""
    def f(x,index,src,out):
        out.copy_(torch.scatter(x,1,index,src))
        return out
    compiled=torch.compile(f,backend='eager',fullgraph=True,dynamic=True)
    rows=[]
    for n in (2,3,5):
        x=torch.zeros(1,n);idx=torch.arange(n).unsqueeze(0)
        src=torch.ones(1,n);out=torch.empty_like(x)
        result=compiled(x,idx,src,out)
        rows.append({'width':n,'correct':torch.equal(result,torch.ones_like(result)),
                     'preserved_out_identity':result is out})
    return {'rows':rows,'tradeoff':'additional intermediate allocation and copy; not a native PyTorch operator fix'}
