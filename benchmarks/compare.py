import json,numpy as np
from pathlib import Path
from src.retrieval import *
result=[]
for seed in (1,2,3):
    data=events(seed=seed);users,items=train(data,seed=seed)
    metrics=evaluate(data,users,items,nprobe=3,k=10)
    result.append({'seed':seed,**metrics})
payload={'dataset':'seeded synthetic chronological interaction events; no real user data','k':10,'results':result,
         'avg_ann_recall':float(np.mean([r['ann_recall_at_k'] for r in result])),
         'avg_certified_recall':float(np.mean([r['certified_recall_at_k'] for r in result]))}
Path('evidence/benchmark.json').write_text(json.dumps(payload,indent=2))
print(json.dumps({k:v for k,v in payload.items() if k!='results'},indent=2))
