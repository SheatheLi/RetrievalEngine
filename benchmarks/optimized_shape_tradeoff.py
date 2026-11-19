"""Measured specialization-vs-padding envelope on real PyTorch attention.

Both strategies use the same 3 graph shape budget. CPU/Eager FX backend.
Timing is not the optimization objective and may be neutral or negative.
"""
from __future__ import annotations
import json,time,statistics
import torch
from src.masked_bucketing import MaskedAttention,batch,next_bucket
from src.bucket_optimizer import optimize_bucket_boundaries

def run(lengths=(5,7,9,13,15,17,25,31,29,13),repeats=9):
    torch.manual_seed(47);torch.set_num_threads(1)
    torch._dynamo.config.recompile_limit=32
    model=MaskedAttention().eval();inputs=[torch.randn(2,n,24) for n in lengths]
    with torch.no_grad():
        references=[model(*batch(x,x.shape[1])) for x in inputs]
    static=(8,16,32,64)
    optimized=optimize_bucket_boundaries(lengths,3,max_supported_length=32)
    outcomes=[]
    for name,bounds in [('static_power_two',static),('optimized_dp',optimized.boundaries)]:
        graph_count=[0]
        def backend(gm,inputs):
            graph_count[0]+=1
            return gm.forward
        torch.compiler.reset()
        compiled=torch.compile(model,backend=backend,fullgraph=True,dynamic=False)
        batches=[batch(x,next_bucket(x.shape[1],bounds)) for x in inputs]
        with torch.no_grad():
            actual=[compiled(a,mask) for a,mask in batches]
            errors=[float((a-b).abs().max()) for a,b in zip(actual,references)]
            times=[]
            for _ in range(repeats):
                t=time.perf_counter()
                for a,mask in batches:compiled(a,mask)
                times.append((time.perf_counter()-t)*1000)
        outcomes.append({'policy':name,'boundaries':list(bounds),
                         'graph_compiles':graph_count[0],
                         'input_tokens':sum(lengths)*2,
                         'padded_tokens':sum(b[0].shape[1]*b[0].shape[0] for b in batches),
                         'max_output_error':max(errors),
                         'cpu_eager_ms_median':statistics.median(times),
                         'cpu_eager_ms_min':min(times),'cpu_eager_ms_max':max(times)})
    return {'torch':torch.__version__,'backend':'torch.compile with FX eager backend on CPU',
            'scenario':'fixed 3 graph shapes, 10 variable lengths, 2 batch items, masked SDPA attention',
            'objective':'minimize padded tokens with identical specialization budget',
            'rows':outcomes,'boundaries_from_holdout_traffic':False,
            'deployment_cap':32,
            'limitation':'same sequence-length sample used to choose optimal buckets; must be tested on future distribution shift'}

if __name__=='__main__':print(json.dumps(run(),indent=2))
