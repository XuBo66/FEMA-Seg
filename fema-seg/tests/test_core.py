import numpy as np
import torch
from fema.model import FEMASeg,BandSplit,SGDU,SelectiveSSM
from fema.losses import objective
from fema.metrics import Metrics
from scripts.split import stratified
from scripts.pseudo import merge

torch.set_num_threads(2)


def tiny():
    return FEMASeg(classes=3,dims=[8,12,16,24],depths=[2,2,2,2],state=2,context_dim=8,context_depth=3,heads=2,bridges=[1,2,3],backend='reference')


def test_backward_odd_shape():
    model=tiny();x=torch.randn(2,3,35,47);y=torch.randint(0,3,(2,35,47));y[:,0]=255
    out=model(x);assert out['logits'].shape==(2,3,35,47)
    loss=objective(out,y,torch.ones_like(y,dtype=torch.float32),.5);loss.backward()
    assert torch.isfinite(loss)
    assert model.stages[0][0].radii.grad.abs().sum()>0
    assert model.decoder[0].up.shift.weight.grad.abs().sum()>0
    assert model.decoder[0].cls.weight.grad.abs().sum()>0


def test_ignore_loss():
    model=tiny().eval();out=model(torch.randn(1,3,32,32));y=torch.full((1,32,32),255)
    loss=objective(out,y,torch.ones_like(y,dtype=torch.float32),.5)
    assert loss.item()==0


def test_metrics_perfect_and_void():
    y=np.zeros((32,32),dtype=np.int64);y[8:24,8:24]=1;y[:2]=255
    m=Metrics(3,[1]);m.update(np.where(y==255,0,y),y);r=m.result()
    assert r['mIoU']==1 and r['mBIoU']==1 and r['GVI_RMSE_fraction']==0
    m=Metrics(3,[1]);m.update(np.zeros_like(y),np.full_like(y,255));assert m.result()['mIoU'] is None


def test_sgdu_identity():
    op=SGDU(4,2).eval();x=torch.randn(1,4,5,7)
    assert torch.allclose(op(x),op.up(x),atol=2e-6)


def test_split_and_competition():
    rows=[{'id':str(i),'stratum':str(i%3)} for i in range(100)]
    a=stratified(rows,.05,42);assert a==stratified(rows,.05,42) and len(a['selected_ids'])==5
    ds=[{'class_id':1,'score':.5},{'class_id':2,'score':.9}]
    y,w=merge((3,3),ds,[np.ones((3,3),bool)]*2,255)
    assert (y==2).all() and np.allclose(w,.9)


def test_scan_cuda_equivalence():
    import pytest
    if not torch.cuda.is_available():pytest.skip('CUDA unavailable')
    pytest.importorskip('mamba_ssm')
    ref=SelectiveSSM(4,3,'reference').cuda();fast=SelectiveSSM(4,3,'cuda').cuda();fast.load_state_dict(ref.state_dict())
    x=torch.randn(2,4,13,device='cuda',requires_grad=True);z=x.detach().clone().requires_grad_()
    a,b=ref(x),fast(z);assert torch.allclose(a,b,atol=2e-4,rtol=2e-4)
    a.sum().backward();b.sum().backward();assert torch.allclose(x.grad,z.grad,atol=2e-4,rtol=2e-4)
