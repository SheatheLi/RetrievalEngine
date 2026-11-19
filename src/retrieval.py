"""Temporal two-tower retrieval and certified inverted-cluster query.
On synthetic user-item events, all held-out positives are excluded from training.
The exactness certificate upper-bounds scores in unvisited clusters.
"""
from __future__ import annotations
from dataclasses import dataclass
import time
import numpy as np
import torch
from sklearn.cluster import KMeans

@dataclass
class Events:
    train: np.ndarray
    test: np.ndarray
    users:int
    items:int

def events(seed=1,users=120,items=400,per_user=16)->Events:
    rng=np.random.default_rng(seed)
    u=rng.normal(size=(users,12));v=rng.normal(size=(items,12))
    # Time-trending user preferences, strictly increasing timestamps per user.
    v/=np.linalg.norm(v,axis=1,keepdims=True)
    rows=[]
    for person in range(users):
        already=set()
        for t in range(per_user):
            latent=u[person]+.16*(t/per_user)*np.roll(u[person],1)
            scores=v@latent + rng.gumbel(size=items)*.35
            for ix in np.argsort(scores)[::-1]:
                if int(ix) not in already:break
            already.add(int(ix));rows.append((person,int(ix),t))
    log=np.array(rows,dtype=np.int64)
    return Events(log[log[:,2]<per_user-2],log[log[:,2]>=per_user-2],users,items)

class Towers(torch.nn.Module):
    def __init__(self,users,items,dim=24):
        super().__init__();self.user=torch.nn.Embedding(users,dim);self.item=torch.nn.Embedding(items,dim)
        torch.nn.init.normal_(self.user.weight,std=.08);torch.nn.init.normal_(self.item.weight,std=.08)
    def forward(self,u,i):
        return torch.nn.functional.normalize(self.user(u),dim=1), torch.nn.functional.normalize(self.item(i),dim=1)

def train(data:Events,seed=1,epochs=12)->tuple[np.ndarray,np.ndarray]:
    torch.manual_seed(seed);torch.set_num_threads(1)
    model=Towers(data.users,data.items);opt=torch.optim.AdamW(model.parameters(),lr=.032,weight_decay=.002)
    rng=np.random.default_rng(seed)
    for epoch in range(epochs):
        perm=rng.permutation(len(data.train))
        for idx in np.array_split(perm,max(1,int(np.ceil(len(perm)/96)))):
            batch=data.train[idx];u=torch.tensor(batch[:,0]);i=torch.tensor(batch[:,1]);negative=torch.tensor(rng.integers(0,data.items,len(idx)))
            uv,iv=model(u,i);_,nv=model(u,negative)
            positive=(uv*iv).sum(1);neg=(uv*nv).sum(1)
            loss=-torch.nn.functional.logsigmoid((positive-neg)*8).mean()
            opt.zero_grad();loss.backward();opt.step()
    with torch.no_grad():
        u=torch.nn.functional.normalize(model.user.weight,dim=1).numpy().copy()
        i=torch.nn.functional.normalize(model.item.weight,dim=1).numpy().copy()
    return u,i

class ClusterIndex:
    def __init__(self,vectors:np.ndarray,ncluster=24,seed=1):
        self.v=vectors.astype(np.float64)
        clusters=KMeans(n_clusters=ncluster,random_state=seed,n_init=3).fit(self.v)
        self.centers=clusters.cluster_centers_;self.members=[np.where(clusters.labels_==i)[0] for i in range(ncluster)]
        self.radii=np.array([np.linalg.norm(self.v[m]-self.centers[i],axis=1).max() for i,m in enumerate(self.members)])
    def search(self,query:np.ndarray,k=10,nprobe=3,certify=False):
        q=np.asarray(query, dtype=np.float64)
        if q.ndim != 1 or q.shape[0] != self.v.shape[1] or not np.isfinite(q).all():
            raise ValueError('query must be a finite vector of index dimension')
        if isinstance(k,bool) or not isinstance(k,(int,np.integer)) or not 1 <= k <= len(self.v):
            raise ValueError('k must be in [1, index size]')
        if isinstance(nprobe,bool) or not isinstance(nprobe,(int,np.integer)) or not 1 <= nprobe <= len(self.members):
            raise ValueError('nprobe must be in [1, cluster count]')
        order=np.argsort(-(self.centers@q))
        scores=np.full(len(self.v),-np.inf);visited=0;cluster_count=0
        for ix in order:
            members=self.members[int(ix)];scores[members]=self.v[members]@q
            visited+=len(members);cluster_count+=1
            if certify:
                if cluster_count>=len(order):break
                kth=np.partition(scores,-k)[-k]
                remaining=order[cluster_count:]
                # Cauchy-Schwarz: max(q·x) <= q·centroid + ||q|| radius.
                bound=np.max(self.centers[remaining]@q + np.linalg.norm(q)*self.radii[remaining])
                if kth>bound+1e-10:break
            elif cluster_count>=nprobe:break
        top=np.argsort(-scores,kind='stable')[:k]
        return top,visited,cluster_count

def evaluate(data:Events,u:np.ndarray,v:np.ndarray,nprobe=3,k=10)->dict:
    idx=ClusterIndex(v,seed=1);exact_hits=[];approx_hits=[];cert_hits=[];exact_vs_approx=[];exact_vs_cert=[];visits=[];cvisits=[];times=[];cert_times=[];exact_times=[]
    # Evaluate one last event per user; training has never used it.
    targets={int(row[0]):int(row[1]) for row in data.test if row[2]==data.test[:,2].max()}
    for user,item in targets.items():
        query=u[user];t_exact=time.perf_counter_ns();ground=np.argsort(-(v@query),kind='stable')[:k]
        exact_times.append((time.perf_counter_ns()-t_exact)/1e6)
        begin=time.perf_counter_ns();approx,visited,_=idx.search(query,k,nprobe,False)
        duration=(time.perf_counter_ns()-begin)/1e6
        t_cert=time.perf_counter_ns();cert,cvisited,_=idx.search(query,k,nprobe,True)
        cert_times.append((time.perf_counter_ns()-t_cert)/1e6)
        exact_hits.append(item in ground);approx_hits.append(item in approx);cert_hits.append(item in cert)
        exact_vs_approx.append(len(set(ground)&set(approx))/k)
        exact_vs_cert.append(len(set(ground)&set(cert))/k)
        visits.append(visited);cvisits.append(cvisited);times.append(duration)
    return {'queries':len(targets),'heldout_hit_exact':float(np.mean(exact_hits)),
            'heldout_hit_ivf':float(np.mean(approx_hits)),'heldout_hit_certified':float(np.mean(cert_hits)),
            'ann_recall_at_k':float(np.mean(exact_vs_approx)),
            'certified_recall_at_k':float(np.mean(exact_vs_cert)),
            'mean_dot_products_ivf':float(np.mean(visits)),
            'mean_dot_products_certified':float(np.mean(cvisits)),
            'exact_dot_products':len(v),'ivf_query_p95_ms':float(np.percentile(times,95)),
            'certified_query_p95_ms':float(np.percentile(cert_times,95)),
            'exact_query_p95_ms':float(np.percentile(exact_times,95))}
