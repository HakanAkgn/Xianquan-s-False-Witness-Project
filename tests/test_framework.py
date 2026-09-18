import numpy as np
import pytest
import torch
from fw.data import Task,draw,encode,equal_function_pair,relevant_change
from fw.model import Transformer
from fw.evaluate import pack_outcomes,unpack_outcomes,paired_stats,qkv_factorial
from fw.theory import theory_checks

@pytest.mark.parametrize('relations',[1,2])
@pytest.mark.parametrize('count',[0,1,2,4,7])
def test_invariants(relations,count):
    t=Task(relations=relations)
    a,b,rows=equal_function_pair(t,257,torch.Generator().manual_seed(19),count)
    assert torch.equal(a.all_country_answers(),b.all_country_answers())
    assert torch.equal(a.labels(),b.labels())
    assert torch.equal(a.intermediate(),b.intermediate())
    indices=torch.arange(a.batch)
    f0=a.f[indices,a.relation]; f1=b.f[indices,b.relation]
    assert ((f0==a.hint[:,None]).sum(1)==0).all()
    assert ((f1==a.hint[:,None]).sum(1)==count).all()
    assert (a.labels()!=a.g[indices,a.hint]).all()
    if relations==2:
        assert torch.equal(encode(a,t).sort(1).values,encode(b,t).sort(1).values)
    assert torch.all(relevant_change(a,t).labels()!=a.labels())

@pytest.mark.parametrize('relations',[1,2])
@pytest.mark.parametrize('layout',['legacy','fixed'])
def test_tokens(relations,layout):
    t=Task(relations=relations,layout=layout)
    w=draw(t,300,torch.Generator().manual_seed(41))
    x=encode(w,t)
    assert x.shape==(300,t.length)
    assert x.min()>=0 and x.max()<t.vocab
    assert (x[:,:t.n]==w.g).all()
    assert (x[:,t.n:(1+relations)*t.n]==w.f.flatten(1)).all()
    if layout=='fixed': assert torch.equal(x[:,-3],w.x)
    else: assert torch.equal(x[~w.present,-1],w.x[~w.present])


def test_no_change_without_swaps():
    t=Task(relations=2)
    a,b,_=equal_function_pair(t,100,torch.Generator().manual_seed(5),0)
    assert torch.equal(encode(a,t),encode(b,t))


def test_patch_identity():
    torch.set_num_threads(1)
    t=Task()
    a,b,_=equal_function_pair(t,16,torch.Generator().manual_seed(5),1)
    r=qkv_factorial(Transformer(t).eval(),a,b,size=16)
    assert r['identity_patch_max_error']<1e-6


def test_outcomes_roundtrip():
    x=np.random.default_rng(5).integers(0,8,(5,4096),dtype=np.uint8)
    assert np.array_equal(unpack_outcomes(pack_outcomes(x)),x)


def test_stats():
    r=paired_stats([0,1,1,0],[1,1,0,0])
    assert r['effect']==0 and r['became_wrong']==1 and r['became_not_wrong']==1


def test_theory():
    r=theory_checks(trials=200)
    assert r['max_gradient_finite_difference_error']<1e-8
    assert r['max_softmax_mass_identity_error']<1e-14
    assert r['max_factorial_identity_error']<1e-14


def test_reproducible_draw():
    t=Task(relations=2)
    a=draw(t,200,torch.Generator().manual_seed(19))
    b=draw(t,200,torch.Generator().manual_seed(19))
    assert torch.equal(encode(a,t),encode(b,t))


def test_one_step_is_not_false_invariance_claim():
    t=Task(relations=2)
    a,b,rows=equal_function_pair(t,100,torch.Generator().manual_seed(19),1)
    a.x=rows[:,0]; b.x=rows[:,0]
    a.two_step.fill_(False); b.two_step.fill_(False)
    assert torch.all(a.labels()!=b.labels())


def test_js_properties_and_gradient():
    from fw.repair import js_divergence
    x=torch.randn(17,8,requires_grad=True)
    y=torch.randn(17,8,requires_grad=True)
    assert abs(float(js_divergence(x,x).detach()))<1e-6
    d=js_divergence(x,y)
    assert float(d.detach())>=0
    assert torch.allclose(d,js_divergence(y,x))
    d.backward()
    assert torch.isfinite(x.grad).all() and torch.isfinite(y.grad).all()


def test_paired_stats_rejects_unpaired_or_single_samples():
    with pytest.raises(ValueError):paired_stats([True],[False])
    with pytest.raises(ValueError):paired_stats([True,False],[False])
