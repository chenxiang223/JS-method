import time
import numpy as np
import torch
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from .model import STATE_NAMES


def linear_cka(x,y):
    x=x-x.mean(0); y=y-y.mean(0)
    numerator=np.linalg.norm(x.T@y,'fro')**2
    denominator=np.linalg.norm(x.T@x,'fro')*np.linalg.norm(y.T@y,'fro')
    return float(numerator/denominator) if denominator>0 else None


@torch.no_grad()
def state_diagnostics(model, train_loader, test_loader, device, max_test=2048):
    def collect(loader, limit=None):
        chunks={k:[] for k in STATE_NAMES}; labels=[]; seen=0
        for x,y in loader:
            if limit is not None:
                x,y=x[:limit-seen],y[:limit-seen]
            states=model.backbone.forward_states(x.to(device))
            for k,v in states.items(): chunks[k].append(v.mean((-2,-1)).cpu().numpy())
            labels.append(y.numpy()); seen+=len(y)
            if limit is not None and seen>=limit: break
        return {k:np.concatenate(v) for k,v in chunks.items()},np.concatenate(labels)
    model.eval()
    train,yt=collect(train_loader); test,y=collect(test_loader,max_test)
    predictions={}; probes={}
    for k in STATE_NAMES:
        probe=make_pipeline(StandardScaler(),LogisticRegression(C=1.,max_iter=1000,random_state=0))
        probe.fit(train[k],yt); pred=probe.predict(test[k]); predictions[k]=pred
        probes[k]={'accuracy':float((pred==y).mean()), 'per_class_accuracy':{
            str(int(c)):float((pred[y==c]==c).mean()) for c in np.unique(y)}}
    pairs={}
    for i,a in enumerate(STATE_NAMES):
        for b in STATE_NAMES[i+1:]:
            pa,pb=predictions[a],predictions[b]
            pairs[a+'__'+b]={'cka':linear_cka(test[a].astype(float),test[b].astype(float)),
                'disagreement':float((pa!=pb).mean()),
                'a_only_correct':float(((pa==y)&(pb!=y)).mean()),
                'b_only_correct':float(((pb==y)&(pa!=y)).mean()),
                'oracle_either_correct':float(((pa==y)|(pb==y)).mean())}
    return {'n_train':len(yt),'n_test':len(y),'probe':'standardized logistic regression, C=1, train only',
            'probes':probes,'pairs':pairs, 'note':'Descriptive diagnostics, never used for checkpoint selection'}


@torch.no_grad()
def benchmark(model,x,device,repeats=30):
    model.eval(); x=x.to(device)
    cuda=str(device).startswith('cuda')
    if cuda: torch.cuda.reset_peak_memory_stats(device)
    for _ in range(5): model(x)
    if cuda: torch.cuda.synchronize(device)
    start=time.perf_counter()
    for _ in range(repeats): model(x)
    if cuda: torch.cuda.synchronize(device)
    seconds=(time.perf_counter()-start)/repeats
    return {'parameters':sum(p.numel() for p in model.parameters()),
            'backbone_parameters':sum(p.numel() for p in model.backbone.parameters()),
            'batch_size':len(x),'batch_ms':seconds*1000,'per_sample_ms':seconds*1000/len(x),
            'peak_cuda_bytes':torch.cuda.max_memory_allocated(device) if cuda else None,
            'device':str(device),'gpu_name':torch.cuda.get_device_name(device) if cuda else None}
