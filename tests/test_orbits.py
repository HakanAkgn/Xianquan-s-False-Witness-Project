import numpy as np
import torch
from fw.data import Task,encode
from fw.orbits import cube_worlds,walsh,cube_statistics


def test_whole_cube_has_identical_answers_and_tokens():
    worlds=list(cube_worlds(Task(relations=2,layout='fixed'),79,4,13))
    _,base=worlds[0]
    tokens=encode(base,Task(relations=2,layout='fixed')).sort(1).values
    answers=base.all_country_answers()
    for mask,w in worlds:
        assert torch.equal(tokens,encode(w,Task(relations=2,layout='fixed')).sort(1).values)
        assert torch.equal(answers,w.all_country_answers())
        selected=w.f[torch.arange(w.batch),w.relation]
        assert ((selected==w.hint[:,None]).sum(1)==mask.bit_count()).all()


def test_walsh_parseval_and_inverse():
    x=np.random.default_rng(9).normal(size=(16,90))
    c=walsh(x)
    assert np.max(abs(16*walsh(c)-x))<1e-12
    assert np.max(abs(x.var(0)-(c[1:]**2).sum(0)))<1e-12


def test_pure_two_body_interaction():
    z=np.arange(16)
    x=(((-1.)**((z&1)!=0))*((-1.)**((z&2)!=0)))[:,None]
    c=walsh(x)
    assert abs(c[3,0]-1)<1e-12
    assert np.count_nonzero(abs(c)>1e-12)==1


def test_singletons_can_miss_failure():
    # Every isolated edit has zero effect, but two simultaneous edits change it.
    f=np.array([0.,0.,0.,1.])
    assert f[1]-f[0]==0 and f[2]-f[0]==0
    assert f[3]-f[0]==1


def test_group_average_jensen_and_poincare():
    rng=np.random.default_rng(11)
    logits=rng.normal(size=(16,99,8))
    p=np.exp(logits-logits.max(-1,keepdims=True));p/=p.sum(-1,keepdims=True)
    labels=np.zeros(99,dtype=int);wrong=np.ones(99,dtype=int)
    d=logits[:,:,1]-logits[:,:,0]
    r=cube_statistics(d,p,labels,wrong)
    assert r['jensen_gain']>=-1e-12
    assert r['parseval_max_error']<1e-12
    assert r['poincare_lower_min_slack']>=-1e-12
    assert r['poincare_upper_min_slack']>=-1e-12


def test_original_compensated_cube():
    t=Task()
    worlds=list(cube_worlds(t,79,4,13))
    _,base=worlds[0]
    tokens=encode(base,t).sort(1).values
    for mask,w in worlds:
        assert torch.equal(tokens,encode(w,t).sort(1).values)
        assert torch.equal(base.all_country_answers(),w.all_country_answers())
        assert torch.equal(base.intermediate(),w.intermediate())


def test_compensation_only_leaves_answers_and_model_format():
    from fw.data import balanced_original_pair
    t=Task()
    a,b,rows,d=balanced_original_pair(t,100,torch.Generator().manual_seed(44))
    comp=a.copy();comp.g=b.g.clone()
    alone=a.copy();alone.f=b.f.clone()
    assert torch.equal(a.all_country_answers(),comp.all_country_answers())
    assert torch.equal(a.all_country_answers(),alone.all_country_answers())
    assert encode(a,t).shape[1]==19
    ix=torch.arange(a.batch)
    assert torch.all(a.g[ix,d[:,0]]!=a.labels())
    assert torch.all(b.g[ix,d[:,0]]!=a.labels())
    wrong=a.g[ix,a.hint]
    assert torch.all(a.g[ix,d[:,0]]!=wrong)
    assert torch.all(b.g[ix,d[:,0]]!=wrong)


def test_one_dimensional_cube_has_no_heldout_combinations():
    p=np.full((2,4,8),1/8)
    r=cube_statistics(np.zeros((2,4)),p,np.zeros(4,dtype=int),np.ones(4,dtype=int))
    assert r['heldout_masks']==[]
    assert r['additive_singleton_heldout_margin_rmse'] is None
    assert r['higher_order_fraction_nonconstant_margin_energy'] is None
