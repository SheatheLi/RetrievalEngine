"""Semantics-preserving callable workaround for PyTorch #194103 out-variants.

This is a USER-SPACE compatibility layer, not a native PyTorch/ATen patch.
Extra intermediate tensor allocations and copies are required.
"""
from __future__ import annotations
from typing import Literal
import torch

Kind=Literal['scatter_src','scatter_add','scatter_value']

def scatter_out_compatible(x: torch.Tensor,index: torch.Tensor,
                           src:torch.Tensor|None,out:torch.Tensor,
                           kind:Kind='scatter_src',value:float=1.)->torch.Tensor:
    if kind=='scatter_src':
        if src is None:raise ValueError('missing tensor source')
        temp=torch.scatter(x,1,index,src)
    elif kind=='scatter_add':
        if src is None:raise ValueError('missing tensor source')
        temp=torch.scatter_add(x,1,index,src)
    elif kind=='scatter_value':
        temp=torch.scatter(x,1,index,float(value))
    else:raise ValueError('unsupported op')
    out.copy_(temp)
    return out

def matrix():
    rows=[]
    for kind in ('scatter_src','scatter_add','scatter_value'):
        fn=lambda x,i,s,out:scatter_out_compatible(x,i,s,out,kind)
        compiled=torch.compile(fn,backend='eager',dynamic=True,fullgraph=True)
        for width in (2,3,5,11):
            x=torch.arange(width,dtype=torch.float32).reshape(1,width)/10
            # Test duplicate scatter indices: scatter_add semantics are nontrivial.
            idx=torch.arange(width).reshape(1,width)//2
            src=torch.full_like(x,2.)
            dest=torch.empty_like(x)
            if kind=='scatter_src': expected=torch.scatter(x,1,idx,src)
            elif kind=='scatter_add':expected=torch.scatter_add(x,1,idx,src)
            else:expected=torch.scatter(x,1,idx,1.)
            got=compiled(x,idx,src,dest)
            rows.append(dict(op=kind,width=width,equal=torch.equal(expected,got),preserves_out=got is dest))
    return rows
