from dataclasses import replace
from pathlib import Path
import json
import numpy as np
import pytest
import torch
from reproduction.core import Setting, Model, World, masks, canonical_numerics, draw, encode, decode, equal_pairs, validate_equal_pair, state_hash, sha256
from reproduction.diagnostic import make_inputs, reference_rows
from reproduction.replay import audit, checked_file, compare_logits, validate_inputs, load_author_model, ReplayBlocked

canonical_numerics(1)


def test_parameter_counts():
    assert sum(p.numel() for p in Model().parameters())==69576
    assert sum(p.numel() for p in Model(Setting(layout='fixed_answer')).parameters())==69704


def test_native_layer_clones_are_independent():
    m=Model()
    a,b=m.encoder.layers
    for p,q in zip(a.parameters(),b.parameters()):
        assert torch.equal(p,q)
        assert p.data_ptr()!=q.data_ptr()
    old=b.linear1.weight.detach().clone()
    with torch.no_grad():a.linear1.weight.add_(1.)
    assert torch.equal(old,b.linear1.weight)


def test_mha_initialization_is_native_not_plain_linear():
    m=Model();att=m.encoder.layers[0].self_attn
    assert torch.equal(att.in_proj_bias,torch.zeros_like(att.in_proj_bias))
    assert torch.equal(att.out_proj.bias,torch.zeros_like(att.out_proj.bias))


@pytest.mark.parametrize('layout',['person_last','suggestion_last','fixed_suggestion','fixed_answer'])
@pytest.mark.parametrize('order',['employee_first','company_first'])
def test_encoder_round_trip(layout,order):
    s=Setting(layout=layout,order=order)
    w=draw(s,257,torch.Generator().manual_seed(123))
    a=encode(w,s);v=decode(a,s)
    assert torch.equal(encode(v,s),a)
    assert torch.equal(w.labels(),v.labels())
    assert (a[:,2*s.n+1]==s.n+2+w.two.long()).all()


def test_primary_absence_suffix():
    s=Setting();w=draw(s,32,torch.Generator().manual_seed(0));w.present.fill_(False)
    a=encode(w,s);b=encode(w,replace(s,layout='person_last'))
    assert torch.equal(a,b)
    assert (a[:,-3]==8).all() and torch.equal(a[:,-1],w.x)


def test_training_distribution():
    s=Setting();w=draw(s,100000,torch.Generator().manual_seed(765))
    middle=w.f[torch.arange(len(w.x)),w.x]
    assert abs(w.two.float().mean().item()-.75)<.007
    assert abs(w.present.float().mean().item()-.5)<.007
    assert abs((w.hint[w.present]==middle[w.present]).float().mean().item()-.75)<.007
    i=torch.arange(len(w.x));expected=torch.where(w.two,w.g[i,middle],middle)
    assert torch.equal(expected,w.labels())


def test_masks_modify_exactly_64_first_layer_entries():
    for order,mode in [('employee_first','open'),('company_first','closed')]:
        s=Setting(order=order);usual,last=masks(s);changed,last2=masks(replace(s,access=mode))
        assert int((changed!=usual).sum())==64
        assert torch.equal(last,last2)
        assert torch.equal(changed[:16,16:],usual[:16,16:])
        assert not changed.diagonal().any()


@pytest.mark.parametrize('order,access',[('company_first','usual'),('employee_first','usual'),('employee_first','open'),('company_first','closed')])
def test_native_manual_forward_double(order,access):
    s=Setting(order=order,access=access);torch.manual_seed(13);m=Model(s).double().eval()
    x=encode(draw(s,32,torch.Generator().manual_seed(314)),s)
    with torch.no_grad():a=m(x);b=m.manual(x)
    assert float((a-b).abs().max())<1e-12
    assert torch.equal(a.argmax(1),b.argmax(1))


@pytest.mark.parametrize('order,access',[('employee_first','usual'),('company_first','closed')])
def test_unavailable_g_cannot_change_first_layer_employee_state(order,access):
    s=Setting(order=order,access=access);m=Model(s).eval()
    w=draw(s,31,torch.Generator().manual_seed(1));v=w.copy();v.g=(v.g+1)%s.n
    def first(z):return m.encoder.layers[0](m.token(encode(z,s))+m.position,src_mask=m.first_mask,is_causal=False)
    pos=slice(0,8) if order=='employee_first' else slice(8,16)
    with torch.no_grad():a,b=first(w),first(v)
    assert torch.equal(a[:,pos],b[:,pos])


def test_d2_equal_function_and_exclusions():
    s=Setting();base,a,b,j=equal_pairs(s,400,65)
    assert torch.equal(a.answers(),b.answers())
    assert ((a.f!=b.f).sum(1)==1).all()
    idx=torch.arange(400);r=a.f[idx,j]
    assert not (base.f==a.hint[:,None]).any()
    assert not (base.f==r[:,None]).any()
    validate_equal_pair(encode(a,s).numpy(),encode(b,s).numpy(),s)


def test_d2_rejects_pilot_other_row_r():
    s=Setting();_,a,b,j=equal_pairs(s,4,65)
    k=next(k for k in range(8) if k!=a.x[0] and k!=j[0])
    a.f[0,k]=b.f[0,k]=a.f[0,j[0]]
    with pytest.raises(ValueError,match='BOTH'):validate_equal_pair(encode(a,s),encode(b,s),s)


def test_shared_initialization_and_semantic_stream_across_conditions():
    s=Setting();torch.manual_seed(10);a=Model(s)
    torch.manual_seed(10);b=Model(replace(s,order='employee_first',layout='person_last'))
    assert state_hash(a.state_dict())==state_hash(b.state_dict())
    w=draw(s,128,torch.Generator().manual_seed(35610))
    v=draw(b.setting,128,torch.Generator().manual_seed(35610))
    for k in vars(w):assert torch.equal(getattr(w,k),getattr(v,k))


def test_reference_count_resolution():
    rows=reference_rows(Path(__file__).parents[1]/'configs/paper_table3.csv')
    assert len(rows)==32
    for r in rows:
        for key in ('wrong_hint_accuracy_pct','control_following_pct','witness_following_pct','effect_pp'):
            count=round(r[key]*4096/100)
            assert abs(100*count/4096-r[key])<=.000501
        counts={k:round(r[k]*4096/100) for k in ('control_following_pct','witness_following_pct','effect_pp')}
        assert counts['witness_following_pct']-counts['control_following_pct']==counts['effect_pp']


def test_input_validation_and_float_rejection():
    s=Setting();d=make_inputs(s,n=16,seed=4);validate_inputs(d,s,n=16)
    d['one_absent']=d['one_absent'].astype(float)
    with pytest.raises(ReplayBlocked,match='noninteger'):validate_inputs(d,s,n=16)


def test_hash_guard(tmp_path):
    p=tmp_path/'a.txt';p.write_text('original');item={'path':p.name,'sha256':sha256(p)}
    assert checked_file(tmp_path,item,'fixture')==p
    p.write_text('changed')
    with pytest.raises(ReplayBlocked,match='Hash mismatch'):checked_file(tmp_path,item,'fixture')


def test_class_mismatch_even_with_tiny_logit_error():
    a=np.array([[.0,1e-9]]);b=np.array([[1e-9,.0]])
    r=compare_logits(a,b);assert r['class_mismatches']==1 and not r['passed']


def test_nonfinite_and_large_error_rejected():
    with pytest.raises(ReplayBlocked,match='Nonfinite'):compare_logits(np.array([[np.nan,0]]),np.zeros((1,2)))
    assert not compare_logits(np.ones((1,2)),np.zeros((1,2)))['passed']


def test_missing_author_manifest_is_blocked():
    r=audit(Path(__file__).parents[1]/'configs/author_replay_manifest.json')
    assert r['status']=='blocked' and not r['extensions_allowed'] and not r['full_paper_reproduced']


def test_diagnostic_checkpoint_cannot_become_author(tmp_path):
    p=tmp_path/'checkpoint.pt';m=Model()
    torch.save({'state_dict':m.state_dict(),'metadata':{'status':'CPU diagnostic'}},p)
    item={'checkpoint':{'path':p.name,'sha256':sha256(p)},'state_dict_key':'state_dict'}
    with pytest.raises(ReplayBlocked,match='not an author'):load_author_model(item,tmp_path,Setting())


def test_incomplete_key_map_rejected(tmp_path):
    p=tmp_path/'fixture.pt';torch.save(Model().state_dict(),p)
    item={'checkpoint':{'path':p.name,'sha256':sha256(p)},'key_map':{}}
    with pytest.raises(ReplayBlocked,match='cover all'):load_author_model(item,tmp_path,Setting())


def test_numerical_flags():
    env=canonical_numerics(1)
    assert not env['mha_fastpath'] and not env['tf32_matmul'] and not env['tf32_cudnn']


@pytest.mark.parametrize('kwargs',[{'layers':3},{'two_step_probability':float('nan')},{'order':'unknown'},{'access':'open'},{'width':63}])
def test_invalid_setting_rejected(kwargs):
    with pytest.raises(ValueError):Setting(**kwargs)
