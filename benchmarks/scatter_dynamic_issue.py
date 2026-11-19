"""Pinned installed PyTorch native-vs-compatible compile experiment.

Failure is in the installed PyTorch version; upstream Issue targets newer nightly.
Compilers, devices and alias cases not tested are deliberately excluded.
"""
import json,time,torch
from src.scatter_compat import scatter_out_compatible

def run():
    out=[]
    for kind in ('scatter_src','scatter_add','scatter_value'):
        def native(x,i,s,d):
            if kind=='scatter_src':return torch.scatter(x,1,i,s,out=d)
            if kind=='scatter_add':return torch.scatter_add(x,1,i,s,out=d)
            return torch.scatter(x,1,i,1.,out=d)
        compiled_native=torch.compile(native,backend='eager',fullgraph=True,dynamic=True)
        compiled_compat=torch.compile(lambda x,i,s,d:scatter_out_compatible(x,i,s,d,kind),backend='eager',fullgraph=True,dynamic=True)
        for width in (2,3,5,11):
            x=torch.zeros(1,width);idx=torch.arange(width).reshape(1,width)//2
            src=torch.full_like(x,2.);o=torch.empty_like(x)
            try:
                nr=compiled_native(x,idx,src,o);status='PASS';err=None
            except Exception as exc:
                status='FAIL';err=type(exc).__name__
            c=compiled_compat(x,idx,src,o)
            expected=native(x,idx,src,torch.empty_like(x))
            out.append({'kind':kind,'width':width,'native_compiled':status,'native_error':err,
                        'compat_equal_to_eager':bool(torch.equal(c,expected)),
                        'out_identity':c is o})
    return {'torch':torch.__version__,'device':'CPU','backend':'torch.compile(eager,dynamic=True,fullgraph=True)',
            'rows':out,'scope':'User-space semantic rewrite with allocation+copy; not a native PyTorch patch.'}
if __name__=='__main__':print(json.dumps(run(),indent=2))
