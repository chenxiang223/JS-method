import numpy as np
import torch
import pytest
from jev.data import make_splits, prepare_scene, validate_splits, SplitCoverageError, PatchDataset
from jev.calibration import fit_temperature, fit_reliability, coverage_threshold
from jev.metrics import evaluate, risk_coverage


def test_train_only_normalization_and_cache(tmp_path):
    rng=np.random.default_rng(0); cube=rng.normal(size=(30,30,5)).astype('float32')
    gt=np.tile(np.arange(1,4),(30,10))
    scene=prepare_scene(cube,gt,'random',0,5,tmp_path)
    train=cube.reshape(-1,5)[scene.splits['train']]
    np.testing.assert_allclose(scene.mean,train.mean(0),atol=1e-6)
    changed=cube.copy(); changed.reshape(-1,5)[scene.splits['test']]+=100
    other=prepare_scene(changed,gt,'random',0,5,tmp_path/'other')
    np.testing.assert_array_equal(scene.mean,other.mean)
    with pytest.raises(ValueError): prepare_scene(changed,gt,'random',0,5,tmp_path)
    assert PatchDataset(scene,'train',5)[0][0].shape==(5,5,5)


def test_spatial_support_and_small_class_failure():
    rng=np.random.default_rng(0); gt=rng.integers(1,4,size=(128,128))
    s,l,m=make_splits(gt,'spatial',1,5,16)
    validate_splits(s,gt.shape,5,True)
    with pytest.raises(ValueError): validate_splits({'a':np.array([100]),'b':np.array([101])},gt.shape,5,True)
    gt=np.zeros((40,40),int); gt[0,0]=1; gt[1,1]=2
    with pytest.raises(SplitCoverageError): make_splits(gt,'random',0)


def test_calibration_and_metrics():
    x=torch.tensor([[8.,0.],[8.,0.],[0.,8.],[0.,8.]])
    y=torch.tensor([0,1,1,0]); t,info=fit_temperature(x,y)
    assert t>1 and info['nll_after']<=info['nll_before']
    a,b,info=fit_reliability(torch.ones(4),torch.ones(4))
    assert info['status']=='degenerate_single_correctness_label' and (a,b)==(1,0)
    threshold,info=coverage_threshold(torch.tensor([.2,.5,.5,.9]),.75)
    assert threshold==.5 and info['calibration_coverage']==.75
    p=np.array([[.8,.2],[.3,.7],[.8,.2],[.3,.7]])
    scores=np.array([.9,.8,.7,.6]); metrics=evaluate(p,np.array([0,1,1,0]),scores,.75)
    assert metrics['oa']==.5 and metrics['coverage']==.5 and metrics['selective_risk']==0
    assert metrics['error_auroc']==1.
    _,r=risk_coverage(np.ones(4),np.array([0,0,1,1]))
    np.testing.assert_allclose(r,.5)


def test_constrained_spatial_assignment_certificate():
    from jev.spatial import constrained_assignment
    gt=np.tile(np.arange(1,4),(96,32))
    regions,metadata=constrained_assignment(gt,np.array([1,2,3]),16,5,0)
    assert regions is not None and metadata['verified_feasible']
    assert set(np.unique(regions))==set(range(5))
