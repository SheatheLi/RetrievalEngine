from real_issues.pytorch_194103 import local_workaround

def test_workaround_preserves_out_and_correctness():
    result=local_workaround()
    assert len(result['rows'])==3
    assert all(r['correct'] and r['preserved_out_identity'] for r in result['rows'])
