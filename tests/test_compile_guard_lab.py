from src.compile_guard_lab import run,train_small

def test_dynamo_captures_real_torch_graph():
    a=run([8,12,8],dynamic=False)
    b=run([8,12,8],dynamic=True)
    assert a.max_abs_error <1e-5 and b.max_abs_error<1e-5
    assert a.graphs >= b.graphs, (a,b)

def test_autograd_training():
    res=train_small(steps=15)
    assert res['final']<res['initial']
