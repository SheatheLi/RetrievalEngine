"""Real public handwritten-digit embeddings as an ANN index sanity check.
Digits are NOT user-item recommendation interactions and do not validate ranking.
"""
import json
import numpy as np
from pathlib import Path
from sklearn.datasets import load_digits
from sklearn.decomposition import PCA
from src.retrieval import ClusterIndex
x,y=load_digits(return_X_y=True)
emb=PCA(n_components=24,random_state=2).fit_transform(x)
emb=emb/(np.linalg.norm(emb,axis=1,keepdims=True)+1e-12)
base=emb[:1400];queries=emb[1400:]
ix=ClusterIndex(base,ncluster=32,seed=2)
found=[];cert=[];cost=[]
for q in queries:
    true=set(np.argsort(-(base@q))[:10]);a,v,_=ix.search(q,k=10,nprobe=4)
    c,vc,_=ix.search(q,k=10,certify=True)
    found.append(len(true & set(a))/10);cert.append(len(true & set(c))/10);cost.append(vc)
r={'source':'sklearn.datasets.load_digits (public digits proxy, not recommendation)',
   'index_items':1400,'queries':len(queries),'approx_recall_at_10':float(np.mean(found)),
   'certified_recall_at_10':float(np.mean(cert)),
   'certified_mean_dot_products':float(np.mean(cost)),'exact_dot_products':len(base)}
Path('evidence/real_digits_proxy.json').write_text(json.dumps(r,indent=2));print(json.dumps(r,indent=2))
