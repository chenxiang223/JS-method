import copy
import torch
import pytest
from models import SPARCNet
from jev.model import JevSPARC, STATE_NAMES
from jev.losses import classification_loss, evidential_mse
from jev.stress import perturb, KINDS

torch.set_num_threads(2)


def test_legacy_compatibility_and_actual_branches():
    torch.manual_seed(3)
    model = SPARCNet(10, base_channels=16, adapter_channels=16, num_classes=3).eval()
    keys=set(model.state_dict())
    x=torch.randn(2,10,15,15)
    with torch.no_grad():
        original=model(x); aux=model(x,return_aux=True); states=model.forward_states(x)
        real,imag=model.cvoca(x); adapted=model.adapter(real,imag)
        spa=model.feinfn.spa_branch(adapted.feature_map)
        fre=model.feinfn.fre_branch(adapted.feature_map, adapted.z_spe,model.raw_hint_proj(x))
    assert set(states)==set(STATE_NAMES)
    torch.testing.assert_close(spa,states['spatial'])
    torch.testing.assert_close(fre,states['frequency'])
    torch.testing.assert_close(states['fused'],aux['mid_feature'])
    torch.testing.assert_close(states['final'],aux['feature'])
    torch.testing.assert_close(original,aux['logits'])
    assert set(model.state_dict())==keys
    clone=copy.deepcopy(model); clone.load_state_dict(model.state_dict(),strict=True)
    torch.testing.assert_close(clone(x),original)


@pytest.mark.parametrize('variant',['sparc','pooled_linear','query','full','no_adaptive','no_evidence','no_stress'])
def test_variants_gradient_and_roundtrip(variant,tmp_path):
    torch.manual_seed(2)
    model=JevSPARC(10,3,variant=variant,channels=16)
    x=torch.randn(2,10,15,15); y=torch.tensor([0,2])
    out=model(x,diagnostics=True)
    assert out.logits.shape==(2,3)
    if variant!='sparc': assert out.state_tokens.shape==(2,6,128)
    assert torch.isfinite(out.reliability).all()
    torch.testing.assert_close(out.probabilities.sum(1),torch.ones(2))
    loss=classification_loss(out,y,0,model.has_evidence); loss.backward()
    for branch in (model.backbone.feinfn.spa_branch,model.backbone.feinfn.fre_branch):
        grads=[p.grad for p in branch.parameters() if p.grad is not None]
        assert grads and all(torch.isfinite(g).all() for g in grads)
        assert sum(g.abs().sum() for g in grads)>0
    if model.has_evidence: assert model.evidence.weight.grad.abs().sum()>0
    model.eval()
    path=tmp_path/'model.pt'; torch.save(model.state_dict(),path)
    clone=JevSPARC(10,3,variant=variant,channels=16).eval()
    clone.load_state_dict(torch.load(path,weights_only=True))
    torch.testing.assert_close(model(x).probabilities,clone(x).probabilities)


def test_reliability_does_not_change_classifier_or_batchnorm():
    model=JevSPARC(8,3,channels=16)
    model.freeze_for_reliability()
    before=copy.deepcopy(model.backbone.state_dict())
    out=model(torch.randn(4,8,15,15))
    torch.nn.functional.binary_cross_entropy_with_logits(out.reliability_logits,torch.tensor([0.,1.,0.,1.])).backward()
    assert all(p.grad is None for p in model.backbone.parameters())
    assert model.reliability_head[0].weight.grad.abs().sum()>0
    for k,v in before.items(): torch.testing.assert_close(v,model.backbone.state_dict()[k])


def test_stress_reproducible_and_spatial_center_intact():
    x=torch.ones(4,10,15,15)
    for kind in KINDS:
        a=perturb(x,kind,3,torch.Generator().manual_seed(4))
        b=perturb(x,kind,3,torch.Generator().manual_seed(4))
        torch.testing.assert_close(a,b)
        assert torch.isfinite(a).all()
        if kind=='spatial_occlusion': torch.testing.assert_close(a[:,:,7,7],x[:,:,7,7])
    assert (x==1).all()


def test_edl_large_evidence_finite():
    e=torch.full((4,3),1e5,requires_grad=True)
    loss=evidential_mse(e,torch.tensor([0,1,2,1])); loss.backward()
    assert torch.isfinite(loss) and torch.isfinite(e.grad).all()
