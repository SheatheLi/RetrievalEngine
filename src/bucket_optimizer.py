"""Exact 1D weighted partition DP for shape-specialization budget.

Targeted objective: given a fixed number of graph shapes, reduce padded
attention tokens while never truncating a real sequence. No claims about GPU
throughput or PyTorch upstream patches.
"""
from __future__ import annotations
from dataclasses import dataclass
from collections import Counter

@dataclass(frozen=True)
class BucketPlan:
    boundaries:tuple[int,...]
    total_tokens:int
    true_tokens:int
    wasted_tokens:int
    distinct_input_lengths:int


def optimize_bucket_boundaries(lengths:list[int]|tuple[int,...],max_graph_shapes:int,
                               max_supported_length:int|None=None)->BucketPlan:
    if not lengths or max_graph_shapes<1 or any(type(x)!=int or x<=0 for x in lengths):
        raise ValueError('positive lengths and at least one graph shape required')
    if max_supported_length is not None and (type(max_supported_length)!=int or max_supported_length<max(lengths)):
        raise ValueError("max_supported_length must cover all training sequences")
    freq=Counter(lengths);values=sorted(freq)
    k=min(max_graph_shapes,len(values));n=len(values)
    pref_count=[0];pref_weight=[0]
    for v in values:
        pref_count.append(pref_count[-1]+freq[v]);pref_weight.append(pref_weight[-1]+freq[v]*v)
    # Last segment [i,j) is padded to values[j-1], with exact weighted cost.
    def cost(i,j,last):
        endpoint=max_supported_length if last and max_supported_length is not None else values[j-1]
        return endpoint*(pref_count[j]-pref_count[i])
    INF=10**30
    dp=[[INF]*(n+1) for _ in range(k+1)]
    prev=[[None]*(n+1) for _ in range(k+1)]
    dp[0][0]=0
    for seg in range(1,k+1):
        for j in range(seg,n+1):
            for i in range(seg-1,j):
                candidate=dp[seg-1][i]+cost(i,j,seg==k and j==n)
                if candidate<dp[seg][j]:dp[seg][j]=candidate;prev[seg][j]=i
    cur=n;boundaries=[]
    for seg in range(k,0,-1):
        boundaries.append(values[cur-1]);cur=prev[seg][cur]
    assert cur==0
    boundaries=list(reversed(boundaries))
    if max_supported_length is not None:boundaries[-1]=max_supported_length
    boundaries=tuple(boundaries)
    actual=sum(lengths)
    return BucketPlan(boundaries,dp[k][n],actual,dp[k][n]-actual,n)
