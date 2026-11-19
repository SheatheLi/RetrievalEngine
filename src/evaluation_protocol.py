"""Paired, query-level retrieval evaluation with explicit label and scope rules.

Neither candidate recall nor nDCG is user satisfaction. Both require external
relevance judgments. The bootstrap interval describes query sampling
uncertainty in the *supplied workload*, not causal product lift. This module
preserves every query's label provenance, candidate loss and ranking loss.
"""
from __future__ import annotations
import argparse,json,math,pathlib,random
from dataclasses import dataclass,asdict

class EvaluationError(ValueError):pass

@dataclass(frozen=True)
class Query:
    query_id:str
    relevant:tuple[str,...]
    candidates:tuple[str,...]
    ranked:tuple[str,...]
    relevance_source:str
    latency_ms:float

@dataclass(frozen=True)
class QueryMetrics:
    query_id:str
    candidate_recall:float
    recall_at_k:float
    ndcg_at_k:float
    mrr_at_k:float
    relevant_candidates_lost:tuple[str,...]
    relevant_ranking_lost:tuple[str,...]
    latency_ms:float

    def serial(self):return asdict(self)

def _ids(row,key,*,allow_empty=True):
    value=row.get(key)
    if not isinstance(value,list) or (not allow_empty and not value):
        raise EvaluationError(f'{key} must be a list'+(' with elements' if not allow_empty else ''))
    if any(not isinstance(x,str) or not x for x in value) or len(set(value))!=len(value):
        raise EvaluationError(f'{key} contains invalid/duplicate IDs')
    return tuple(value)

def parse_query(row)->Query:
    if not isinstance(row,dict):raise EvaluationError('query must be object')
    ident=row.get('query_id');provenance=row.get('relevance_source')
    if not isinstance(ident,str) or not ident:raise EvaluationError('query_id required')
    if not isinstance(provenance,str) or not provenance.strip() or provenance=='same_model':
        raise EvaluationError('independent relevance provenance required')
    latency=row.get('latency_ms')
    if type(latency) not in (float,int) or not math.isfinite(latency) or latency<0:
        raise EvaluationError('latency_ms must be nonnegative finite')
    rel=_ids(row,'relevant',allow_empty=False)
    candidates=_ids(row,'candidates')
    ranked=_ids(row,'ranked')
    if not set(ranked).issubset(candidates):
        raise EvaluationError('ranked items absent from candidate stage')
    return Query(ident,rel,candidates,ranked,provenance,float(latency))

def evaluate(query:Query,k:int)->QueryMetrics:
    if isinstance(k,bool) or not isinstance(k,int) or k<1:
        raise EvaluationError('k must be a positive integer')
    truth=set(query.relevant);candidate=set(query.candidates)
    top=query.ranked[:k]
    hits=[1 if item in truth else 0 for item in top]
    dcg=sum(hit/math.log2(rank+2) for rank,hit in enumerate(hits))
    ideal=sum(1/math.log2(rank+2) for rank in range(min(k,len(truth))))
    score=dcg/ideal if ideal else 0.
    reciprocal=next((1/(i+1) for i,hit in enumerate(hits) if hit),0.)
    return QueryMetrics(query.query_id,len(candidate&truth)/len(truth),
        sum(hits)/len(truth),score,reciprocal,
        tuple(sorted(truth-candidate)),tuple(sorted((truth&candidate)-set(top))),
        query.latency_ms)

def _mean(values):return sum(values)/len(values)

def _ci(values,iterations,seed):
    if len(values)==1:return [values[0],values[0]]
    rng=random.Random(seed);n=len(values)
    samples=sorted(_mean([values[rng.randrange(n)] for _ in range(n)]) for _ in range(iterations))
    return [samples[int(.025*(iterations-1))],samples[int(.975*(iterations-1))]]

def report(rows:list[dict],*,k:int=10,iterations:int=1000,seed:int=17)->dict:
    if not isinstance(rows,list) or not rows:raise EvaluationError('nonempty query set required')
    if type(iterations) is not int or iterations<100:raise EvaluationError('at least 100 bootstrap replicates')
    queries=[parse_query(x) for x in rows]
    if len(set(q.query_id for q in queries))!=len(queries):raise EvaluationError('duplicate query_id')
    metrics=[evaluate(q,k) for q in queries]
    dimensions=('candidate_recall','recall_at_k','ndcg_at_k','mrr_at_k','latency_ms')
    return {'k':k,'queries':len(metrics),'label_sources':sorted(set(q.relevance_source for q in queries)),
      'summary':{name:{'mean':_mean([getattr(x,name) for x in metrics]),
                       'query_bootstrap_95_interval':_ci([getattr(x,name) for x in metrics],iterations,seed)}
                 for name in dimensions},
      'per_query':[x.serial() for x in metrics],
      'claim_boundary':'supplied independent labels and sampled queries only; no global ANN superiority or online lift'}

def paired_difference(baseline:list[dict],candidate:list[dict],*,k=10,iterations=1000,seed=17):
    b={q.query_id:q for q in map(parse_query,baseline)}
    c={q.query_id:q for q in map(parse_query,candidate)}
    if set(b)!=set(c) or len(b)!=len(baseline) or len(c)!=len(candidate):
        raise EvaluationError('same unique query IDs required for a paired comparison')
    diffs=[]
    for qid in sorted(b):
        if b[qid].relevant!=c[qid].relevant or b[qid].relevance_source!=c[qid].relevance_source:
            raise EvaluationError('relevance judgments or provenance differ between systems')
        bm=evaluate(b[qid],k);cm=evaluate(c[qid],k)
        diffs.append({'query_id':qid,'ndcg_delta':cm.ndcg_at_k-bm.ndcg_at_k,
                      'latency_delta_ms':cm.latency_ms-bm.latency_ms,
                      'candidate_recall_delta':cm.candidate_recall-bm.candidate_recall})
    return {'queries':len(diffs),'summary':{
        field:{'mean_delta':_mean([x[field] for x in diffs]),
               'query_bootstrap_95_interval':_ci([x[field] for x in diffs],iterations,seed)}
        for field in ('ndcg_delta','latency_delta_ms','candidate_recall_delta')},
        'per_query':diffs,
        'claim_boundary':'paired descriptive comparison in frozen workload; query-bootstrap is not an online A/B test'}

def main():
    cli=argparse.ArgumentParser(description=__doc__)
    cli.add_argument('queries',type=pathlib.Path)
    cli.add_argument('--baseline',type=pathlib.Path)
    cli.add_argument('--k',type=int,default=10)
    args=cli.parse_args()
    rows=json.loads(args.queries.read_text(encoding='utf8'))
    result=paired_difference(json.loads(args.baseline.read_text(encoding='utf8')),rows,k=args.k) if args.baseline else report(rows,k=args.k)
    print(json.dumps(result,indent=2,ensure_ascii=False,allow_nan=False))
if __name__=='__main__':main()
