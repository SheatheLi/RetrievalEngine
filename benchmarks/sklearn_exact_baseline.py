"""Comparative independent API baseline; no ANN production claim."""
import os,sys,time,json,platform,statistics, pathlib
import numpy as np
from sklearn.datasets import load_digits
from sklearn.decomposition import PCA
from sklearn.neighbors import NearestNeighbors
ROOT=pathlib.Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT));sys.path.insert(0,str(ROOT/'src'))
from src.certified_quantized import CertifiedQuantizedIndex,exact_mips
x,_=load_digits(return_X_y=True)
z=PCA(n_components=24,random_state=2).fit_transform(x)
z=z/(np.linalg.norm(z,axis=1,keepdims=True)+1e-12)
base=z[:1400].astype('float64'); queries=z[1400:1797].astype('float64');k=10
index=CertifiedQuantizedIndex(base,bits=8)
reference=[exact_mips(base,q,k) for q in queries]
sk=NearestNeighbors(n_neighbors=10,algorithm='brute',metric='euclidean');t=time.perf_counter();sk.fit(base);fit=time.perf_counter()-t
idxs=sk.kneighbors(queries,return_distance=False)
sk_match=sum(bool(np.array_equal(r,z)) for r,z in zip(idxs,reference))
assert sk_match==len(queries),sk_match
models={
 'certified_vectorized_single':lambda i:index.search(queries[i],k).indices,
 'numpy_dense_single':lambda i:exact_mips(base,queries[i],k),
 'sklearn_exact_brute_single':lambda i:sk.kneighbors(queries[i:i+1],return_distance=False)[0]
}
results={s:[] for s in models};correct={s:0 for s in models}
for j in range(7):
 rnd=np.random.default_rng(531+j).permutation(len(queries));names=list(models); names=names[j%3:]+names[:j%3]
 for name in names:
  start=time.perf_counter()
  for i in rnd:
   ans=models[name](i)
   if not np.array_equal(ans,reference[i]):raise AssertionError(name+str(i))
  results[name].append(time.perf_counter()-start)
# batch optimum external exact API
batch=[]
for j in range(7):
 ids=np.random.default_rng(632+j).permutation(len(queries));t=time.perf_counter();rs=sk.kneighbors(queries[ids],return_distance=False);batch.append(time.perf_counter()-t)
 for row,i in zip(rs,ids): assert np.array_equal(row,reference[i])
out={'scope':'397 queries × 1400 float64 PCA24 normalized digits vectors, k=10, 7 repeat rounds; external sklearn nearest neighbors exact brute; toy public dataset, not production', 'env':{'python':platform.python_version(),'numpy':np.__version__,'sklearn':__import__('sklearn').__version__,'openblas':os.environ.get('OPENBLAS_NUM_THREADS'),'os':platform.platform()},'baselines':['project certified quantization','NumPy full exact scan','scikit-learn NearestNeighbors brute single-query','scikit-learn NearestNeighbors brute batched'], 'sklearn_fit_seconds':fit,'exact_match_count':len(queries),'seconds_by_round':results,'batch_sklearn_seconds_by_round':batch,'median_seconds':{k:statistics.median(v) for k,v in results.items()}|{'sklearn_exact_brute_batch':statistics.median(batch)},'notes':'All methods returned exactly equal ordered indices under this dataset. sklearn batch amortizes per-request overhead; batch API is not same latency contract as single-query. No claim to beat FAISS/HNSW/online service.'}
p=ROOT/'evidence'/'sklearn_external_api_comparison_20261112.json';p.write_text(json.dumps(out,ensure_ascii=False,indent=2),encoding='utf8');print(json.dumps(out['median_seconds'],indent=2));print(p)
