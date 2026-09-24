"""One immutable experiment: train -> reliability -> held-out calibration -> test."""
import argparse
from dataclasses import asdict
from pathlib import Path
import hashlib
import json
import os
os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", ":4096:8")
import platform
import random
import sys
import time
import traceback
import numpy as np
import torch
from torch.nn import functional as F
from torch.utils.data import DataLoader, Subset
from .data import load_scene, prepare_scene, PatchDataset, SplitCoverageError, SPLITS
from .model import JevSPARC
from .losses import classification_loss
from .stress import perturb, KINDS
from .calibration import fit_temperature, fit_reliability, coverage_threshold
from .metrics import evaluate
from .diagnostics import state_diagnostics, benchmark

ROOT=Path(__file__).resolve().parents[1]


def write_json(path, obj):
    path=Path(path); temp=path.with_suffix(path.suffix+'.tmp')
    temp.write_text(json.dumps(obj,indent=2,ensure_ascii=False,allow_nan=False),encoding='utf-8')
    temp.replace(path)


def seed_all(seed):
    random.seed(seed); np.random.seed(seed); torch.manual_seed(seed)
    if torch.cuda.is_available(): torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.benchmark=False
    torch.backends.cudnn.deterministic=True
    torch.use_deterministic_algorithms(True,warn_only=True)


def provenance():
    source={str(p.relative_to(ROOT)):hashlib.sha256(p.read_bytes()).hexdigest()
            for sub in ('jev','models','datasets') for p in sorted((ROOT/sub).rglob('*.py'))}
    return {'source_sha256':source,'python':sys.version,'torch':torch.__version__,
            'numpy':np.__version__,'platform':platform.platform(),
            'cuda_runtime':torch.version.cuda, 'argv':sys.argv,
            'determinism':'seeded, cudnn deterministic, torch deterministic warn_only; cross-device equality not promised'}


@torch.no_grad()
def collect(model, loader, device, stress=None, seed=0):
    model.eval()
    generator=torch.Generator(device=device).manual_seed(seed)
    rows={k:[] for k in ('logits','probabilities','prediction','reliability','reliability_logits','class_evidence','uncertainty','labels')}
    for x,y in loader:
        x=x.to(device)
        if stress is not None: x=perturb(x,*stress,generator)
        o=model(x)
        for k in rows:
            rows[k].append(y.cpu() if k=='labels' else getattr(o,k).detach().cpu())
    if not rows['labels']: raise ValueError('Empty evaluation split')
    return {k:torch.cat(v) for k,v in rows.items()}


def scores(rows, threshold):
    return evaluate(rows['probabilities'].numpy(),rows['labels'].numpy(),rows['reliability'].numpy(),threshold)


def evaluate_nll(model,loader,device):
    model.eval(); total=0.; n=0
    with torch.no_grad():
        for x,y in loader:
            o=model(x.to(device)); total+=F.cross_entropy(o.logits,y.to(device),reduction='sum').item(); n+=len(y)
    return total/n


def run(args):
    torch.set_num_threads(args.threads)
    seed_all(args.seed)
    device=torch.device(args.device)
    outdir=Path(args.output)/args.dataset/args.protocol/f'seed{args.seed}'/args.variant
    outdir.mkdir(parents=True,exist_ok=True)
    config=vars(args).copy()
    if (outdir/'result.json').exists():
        previous=json.loads((outdir/'config.json').read_text(encoding='utf-8'))
        if previous != config: raise ValueError('Completed run has a different configuration')
        print(f'Already complete: {outdir}',flush=True); return
    if (outdir/'config.json').exists():
        raise RuntimeError(f'Incomplete run exists: {outdir}. Preserve it and choose a new --output for an explicit retry.')
    write_json(outdir/'config.json',config); write_json(outdir/'provenance.json',provenance())
    started=time.time()
    def status(phase,**extra):
        record={'phase':phase,'updated_unix':time.time(),'elapsed_seconds':time.time()-started,**extra}
        write_json(outdir/'status.json',record)
        print(json.dumps(record),flush=True)
    try:
        status('data')
        cube,gt,source=load_scene(args.data_root,args.dataset)
        write_json(outdir/'data_source.json',source)
        split_dir=Path(args.output)/'_splits'/args.dataset/args.protocol/f'seed{args.seed}'
        scene=prepare_scene(cube,gt,args.protocol,args.seed,args.patch_size,split_dir,args.block_size)
        np.savez_compressed(outdir/'preprocessing.npz',mean=scene.mean,std=scene.std,labels=scene.labels)
        classes=len(scene.labels)
        datasets={k:PatchDataset(scene,k,args.patch_size,augment=k=='train') for k in SPLITS}
        loaders={k:DataLoader(v,batch_size=args.batch_size if k=='train' else args.eval_batch_size,shuffle=k=='train',num_workers=0,
                    generator=torch.Generator().manual_seed(args.seed+31),pin_memory=device.type=='cuda')
                 for k,v in datasets.items()}
        plain_train=DataLoader(PatchDataset(scene,'train',args.patch_size),batch_size=args.batch_size,shuffle=False)
        counts=np.bincount(datasets['train'].labels,minlength=classes).tolist()
        model_kwargs={'in_channels':cube.shape[-1], 'num_classes':classes, 'variant':args.variant,
                      'channels':args.channels,'class_counts':counts}
        model=JevSPARC(**model_kwargs).to(device)
        opt=torch.optim.AdamW(model.parameters(),lr=args.lr,weight_decay=1e-4)
        scheduler=torch.optim.lr_scheduler.CosineAnnealingLR(opt,args.epochs)
        best=float('inf'); history=[]; seed_all(args.seed+100)
        classification_start=time.time()
        for epoch in range(args.epochs):
            model.train(); summed=0.; n=0
            for x,y in loaders['train']:
                x,y=x.to(device),y.to(device); opt.zero_grad(set_to_none=True)
                o=model(x)
                loss=classification_loss(o,y,epoch,model.has_evidence,
                                         brier_weight=0. if args.variant=='sparc' else .1)
                if not torch.isfinite(loss): raise FloatingPointError('Nonfinite classification loss')
                loss.backward(); torch.nn.utils.clip_grad_norm_(model.parameters(),5.); opt.step()
                summed+=loss.item()*len(y); n+=len(y)
            val=evaluate_nll(model,loaders['val'],device)
            if not np.isfinite(val): raise FloatingPointError('Nonfinite validation NLL')
            if val<best:
                best=val
                torch.save({'model':model.state_dict(),'model_kwargs':model_kwargs,'epoch':epoch,'val_nll':val},outdir/'best_classifier.pt')
            scheduler.step()
            history.append({'epoch':epoch+1,'train_loss':summed/n,'val_nll':val,'best_nll':best})
            write_json(outdir/'classification_history.json',history)
            status('classification',epoch=epoch+1,total_epochs=args.epochs,val_nll=val,best_nll=best)
        classification_seconds=time.time()-classification_start
        checkpoint=torch.load(outdir/'best_classifier.pt',map_location=device,weights_only=False)
        model.load_state_dict(checkpoint['model'])
        reliability_history=[]; reliability_start=time.time()
        if model.has_reliability:
            model.freeze_for_reliability()
            relopt=torch.optim.AdamW(model.reliability_head.parameters(),lr=1e-3,weight_decay=1e-4)
            gen=torch.Generator(device=device).manual_seed(args.seed+200)
            control=random.Random(args.seed+200)
            for epoch in range(args.reliability_epochs):
                loss_sum=0.; n=0; correct_count=0
                for x,y in plain_train:
                    x,y=x.to(device),y.to(device)
                    if args.variant!='no_stress':
                        stressed=perturb(x,control.choice(KINDS),control.choice((1,2,3)),gen)
                        x,y=torch.cat((x,stressed)),torch.cat((y,y))
                    relopt.zero_grad(set_to_none=True)
                    o=model(x); target=(o.prediction==y).float().detach()
                    loss=F.binary_cross_entropy_with_logits(o.reliability_logits,target)
                    if not torch.isfinite(loss): raise FloatingPointError('Nonfinite reliability loss')
                    loss.backward(); relopt.step()
                    loss_sum+=loss.item()*len(y); n+=len(y); correct_count+=int(target.sum())
                reliability_history.append({'epoch':epoch+1,'bce':loss_sum/n,'correct':correct_count,'n':n})
                write_json(outdir/'reliability_history.json',reliability_history)
                status('reliability',epoch=epoch+1,total_epochs=args.reliability_epochs,bce=loss_sum/n,correct=correct_count,n=n)
        reliability_seconds=time.time()-reliability_start
        model.eval(); status('calibration')
        fit=collect(model,loaders['cal_fit'],device)
        temperature,t_info=fit_temperature(fit['logits'],fit['labels'])
        model.temperature.fill_(temperature)
        calibration={'temperature':temperature,'temperature_fit':t_info}
        if model.has_reliability:
            a,b,info=fit_reliability(fit['reliability_logits'],fit['prediction']==fit['labels'])
            model.reliability_scale.fill_(a); model.reliability_bias.fill_(b)
            calibration.update(reliability_scale=a,reliability_bias=b,reliability_fit=info)
        threshold_rows=collect(model,loaders['cal_threshold'],device)
        threshold,threshold_info=coverage_threshold(threshold_rows['reliability'],args.coverage)
        model.abstain_threshold.fill_(threshold)
        calibration['threshold']=threshold; calibration['threshold_selection']=threshold_info
        write_json(outdir/'calibration.json',calibration)
        torch.save({'model':model.state_dict(),'model_kwargs':model_kwargs,'config':config,
                    'calibration':calibration,'label_values':scene.labels.tolist(),
                    'mean':torch.from_numpy(scene.mean),'std':torch.from_numpy(scene.std),
                    'split_sha256':hashlib.sha256((split_dir/'split.npz').read_bytes()).hexdigest()},outdir/'model.pt')
        status('test')
        test=collect(model,loaders['test'],device)
        result={'dataset':args.dataset,'protocol':args.protocol,'seed':args.seed,'variant':args.variant,
                'status':'completed','best_epoch':checkpoint['epoch']+1,'split':scene.metadata,
                'classification_seconds':classification_seconds,'reliability_seconds':reliability_seconds,
                'calibrated':scores(test,threshold),'stress':{},
                'evidence_semantics':'Dirichlet vacuity' if model.has_evidence else 'disabled; zero evidence is a placeholder',
                'baseline_protocol':'original SPARC architecture, common single-stage CE training; not the original decoupled recipe'}
        np.savez_compressed(outdir/'predictions.npz',**{k:v.numpy() for k,v in test.items()},centre_ids=scene.splits['test'])
        # Pre-calibration metrics reuse identical predictions, without another forward.
        raw={k:v.clone() for k,v in test.items()}
        raw['probabilities']=raw['logits'].softmax(-1)
        raw['reliability']=raw['reliability_logits'].sigmoid() if model.has_reliability else raw['probabilities'].max(-1).values
        raw_threshold_scores=threshold_rows['reliability_logits'].sigmoid() if model.has_reliability else threshold_rows['logits'].softmax(-1).max(-1).values
        raw_threshold,_=coverage_threshold(raw_threshold_scores,args.coverage)
        result['uncalibrated']=scores(raw,raw_threshold)
        # Fixed stratified test subset for all stress comparisons, never for tuning.
        ytest=datasets['test'].labels; rng=np.random.default_rng(args.seed+400)
        ids=[]
        for c in range(classes):
            candidates=np.flatnonzero(ytest==c)
            limit=max(1,int(args.stress_samples*len(candidates)/len(ytest)))
            ids.extend(rng.permutation(candidates)[:limit].tolist())
        ids=np.asarray(ids,dtype=int)
        stress_loader=DataLoader(Subset(datasets['test'],ids),batch_size=args.eval_batch_size)
        np.save(outdir/'stress_centre_ids.npy',scene.splits['test'][ids])
        clean_subset=collect(model,stress_loader,device)
        result['stress_clean_subset']=scores(clean_subset,threshold)
        for ki,kind in enumerate(KINDS):
            for severity in (1,2,3):
                status('stress_test',kind=kind,severity=severity)
                rows=collect(model,stress_loader,device,(kind,severity),args.seed+1000+ki*10+severity)
                result['stress'][f'{kind}_{severity}']=scores(rows,threshold)
                np.savez_compressed(outdir/f'stress_{kind}_{severity}.npz',**{k:v.numpy() for k,v in rows.items()})
        x,_=next(iter(plain_train)); result['compute']=benchmark(model,x,device)
        if args.variant in ('sparc','full'):
            status('state_diagnostics')
            diagnostics=state_diagnostics(model,plain_train,stress_loader,device)
            write_json(outdir/'state_diagnostics.json',diagnostics)
        result['elapsed_seconds']=time.time()-started
        write_json(outdir/'result.json',result); status('completed')
    except SplitCoverageError as error:
        write_json(outdir/'blocked_split.json',error.report)
        status('blocked_split',reason=str(error))
    except Exception as error:
        (outdir/'traceback.txt').write_text(traceback.format_exc(),encoding='utf-8')
        status('failed',reason=str(error)); raise


def parser():
    p=argparse.ArgumentParser()
    p.add_argument('--data-root',required=True); p.add_argument('--output',required=True)
    p.add_argument('--dataset',choices=['IndianPines','PaviaU','Salinas'],required=True)
    p.add_argument('--protocol',choices=['random','spatial'],default='random')
    p.add_argument('--variant',choices=['sparc','pooled_linear','query','full','no_adaptive','no_evidence','no_stress'],default='full')
    p.add_argument('--seed',type=int,default=0); p.add_argument('--epochs',type=int,default=150)
    p.add_argument('--reliability-epochs',type=int,default=20)
    p.add_argument('--eval-batch-size',type=int,default=256)
    p.add_argument('--batch-size',type=int,default=64); p.add_argument('--channels',type=int,default=64)
    p.add_argument('--patch-size',type=int,default=15); p.add_argument('--block-size',type=int,default=32)
    p.add_argument('--lr',type=float,default=3e-4); p.add_argument('--coverage',type=float,default=.9)
    p.add_argument('--stress-samples',type=int,default=4096)
    p.add_argument('--device',default='cuda:0'); p.add_argument('--threads',type=int,default=4)
    return p

if __name__=='__main__':
    run(parser().parse_args())
