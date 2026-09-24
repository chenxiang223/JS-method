"""Bounded sequential worker; partition whole dataset/protocol/seed groups across GPUs."""
import argparse
import itertools
import json
import subprocess
import sys
import time
from pathlib import Path
from .data import load_scene, prepare_scene, SplitCoverageError
from .experiment import write_json

VARIANTS=('sparc','pooled_linear','query','full','no_adaptive','no_evidence','no_stress')

def main():
    p=argparse.ArgumentParser()
    p.add_argument('--data-root',required=True);p.add_argument('--output',required=True)
    p.add_argument('--worker',type=int,default=0);p.add_argument('--workers',type=int,default=2)
    p.add_argument('--device',default='cuda:0');p.add_argument('--epochs',type=int,default=150)
    p.add_argument('--datasets',nargs='+',default=['IndianPines','PaviaU','Salinas'])
    p.add_argument('--protocols',nargs='+',default=['random','spatial'])
    p.add_argument('--seeds',nargs='+',type=int,default=[0,1,2])
    p.add_argument('--variants',nargs='+',default=list(VARIANTS))
    a=p.parse_args(); root=Path(a.output);root.mkdir(parents=True,exist_ok=True)
    tasks=list(itertools.product(a.datasets,a.protocols,a.seeds))
    manifest=[];start=time.time()
    for i,(dataset,protocol,seed) in enumerate(tasks):
        if i%a.workers!=a.worker: continue
        split_dir=root/'_splits'/dataset/protocol/f'seed{seed}'
        cube,gt,_=load_scene(a.data_root,dataset)
        try:
            prepare_scene(cube,gt,protocol,seed,15,split_dir)
        except SplitCoverageError as e:
            split_dir.mkdir(parents=True,exist_ok=True)
            write_json(split_dir/'blocked_split.json',e.report)
            manifest.append({'dataset':dataset,'protocol':protocol,'seed':seed,'status':'blocked_split','reason':e.report})
            write_json(root/f'worker{a.worker}.json',manifest); continue
        del cube,gt
        for variant in a.variants:
            run_dir=root/dataset/protocol/f'seed{seed}'/variant
            cmd=[sys.executable,'-u','-m','jev.experiment','--data-root',a.data_root,'--output',a.output,
                 '--dataset',dataset,'--protocol',protocol,'--seed',str(seed),'--variant',variant,
                 '--device',a.device,'--epochs',str(a.epochs)]
            logdir=root/'logs'; logdir.mkdir(exist_ok=True)
            log=logdir/f'{dataset}_{protocol}_{seed}_{variant}.log'
            write_json(root/f'worker{a.worker}_status.json',{'status':'running','command':cmd,'log':str(log),'started':time.time()})
            with log.open('a',encoding='utf-8') as f:
                process=subprocess.Popen(cmd,stdout=f,stderr=subprocess.STDOUT)
                while process.poll() is None:
                    write_json(root/f'worker{a.worker}_status.json',{'status':'running','pid':process.pid,'command':cmd,'log':str(log),
                               'heartbeat':time.time(),'elapsed_total_seconds':time.time()-start})
                    time.sleep(10)
            manifest.append({'dataset':dataset,'protocol':protocol,'seed':seed,'variant':variant,'exit_code':process.returncode,
                             'status':'completed' if (run_dir/'result.json').exists() else 'failed'})
            write_json(root/f'worker{a.worker}.json',manifest)
            if process.returncode:
                write_json(root/f'worker{a.worker}_status.json',{'status':'failed','log':str(log),'exit_code':process.returncode})
                raise SystemExit(process.returncode)
    write_json(root/f'worker{a.worker}_status.json',{'status':'completed','elapsed_seconds':time.time()-start})

if __name__=='__main__': main()
