from src.bucket_optimizer import optimize_bucket_boundaries
from src.masked_bucketing import next_bucket
import itertools,random


def test_optimized_3_buckets_reduce_padding_without_graph_budget_growth():
    lengths=(5,7,9,13,15,17,25,31,29,13)
    r=optimize_bucket_boundaries(lengths,3)
    static=sum(next_bucket(n,(8,16,32,64)) for n in lengths)
    assert len(r.boundaries)==3
    assert r.total_tokens<static
    assert r.total_tokens==sum(next_bucket(n,r.boundaries) for n in lengths)
    assert r.wasted_tokens==r.total_tokens-r.true_tokens


def test_dynamic_program_matches_exhaustive_partition_25_seeds():
    for seed in range(25):
        random.seed(seed)
        values=sorted(random.sample(range(2,33),8))
        lengths=[v for v in values for _ in range(random.randint(1,5))]
        r=optimize_bucket_boundaries(tuple(lengths),3)
        exhaustive=min(sum(next_bucket(x,bounds) for x in lengths)
                       for bounds in itertools.combinations(values[:-1],2)
                       if True for bounds in [(*bounds,values[-1])])
        assert r.total_tokens==exhaustive


def test_legal_inputs_and_no_truncation():
    for lengths in [(1,1,1),(1,2,2,2,7),(2,10,3,5)]:
        for k in (1,2,5):
            r=optimize_bucket_boundaries(lengths,k)
            assert r.total_tokens>=r.true_tokens
            assert all(next_bucket(x,r.boundaries)>=x for x in lengths)


def test_bounded_future_shape_32_and_drift_failure():
    train=(5,7,9,13,15,17,25,31,29,13)
    r=optimize_bucket_boundaries(train,3,max_supported_length=32)
    assert r.boundaries[-1]==32
    assert next_bucket(32,r.boundaries)==32
    assert r.total_tokens==sum(next_bucket(n,r.boundaries) for n in train)
    # A distribution shift toward len=8 can make the trained DP policy
    # worse than the fixed power-of-two baseline. This is not a universal win.
    future=[8]*50
    assert sum(next_bucket(n,r.boundaries) for n in future) > 8*len(future)
