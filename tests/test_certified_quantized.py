import numpy as np
from src.certified_quantized import CertifiedQuantizedIndex, exact_mips

def test_exhaustive_random_and_near_ties():
    for seed in range(12):
        rng=np.random.default_rng(seed)
        v=rng.normal(size=(75,11))
        v[2]=v[1]+1.e-10
        v[3]=v[1]
        ix=CertifiedQuantizedIndex(v,bits=4 if seed%2 else 8)
        for j in range(18):
            q=rng.normal(size=11)
            for k in (1,5,20):
                np.testing.assert_array_equal(ix.search(q,k).indices,exact_mips(v,q,k))

def test_invalid():
    try:CertifiedQuantizedIndex(np.array([[float('nan')]]))
    except ValueError:pass
    else:raise AssertionError('NaN accepted')

def test_certificate_vectorized_edge_cases():
    """Conservative bounds must preserve exact ties, zero query, and full k."""
    rng=np.random.default_rng(239)
    v=rng.normal(size=(80, 9)).astype(np.float64)
    v[9]=v[8]
    v[10]=v[8]+1.e-13
    index=CertifiedQuantizedIndex(v,bits=4)
    for q in (np.zeros(9),rng.normal(size=9),v[8],-v[8]):
        for k in (1, 7, 60, len(v)):
            res=index.search(q,k)
            np.testing.assert_array_equal(res.indices,exact_mips(v,q,k))
            assert res.full_precision_scores <= len(v)
            assert res.full_precision_scores >= k
