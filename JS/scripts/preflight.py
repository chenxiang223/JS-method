import argparse,json,platform
from pathlib import Path
import numpy as np
import torch
from jev.data import load_scene, make_splits, prepare_scene, SplitCoverageError
p=argparse.ArgumentParser();p.add_argument('--data-root',required=True);p.add_argument('--output',required=True);p.add_argument('--split-output');p.add_argument('--datasets',nargs='+',default=['IndianPines','PaviaU','Salinas']);a=p.parse_args()
report={'torch':torch.__version__,'numpy':np.__version__,'python':platform.python_version(),
        'gpus':[torch.cuda.get_device_name(i) for i in range(torch.cuda.device_count())],'datasets':{}}
for name in a.datasets:
    cube,gt,source=load_scene(a.data_root,name)
    item={'shape':list(cube.shape),'classes':int(len(np.unique(gt[gt>0]))),'source':source,'spatial':{}}
    for seed in (0,1,2):
        try:
            s,labels,meta=make_splits(gt,'spatial',seed)
            if a.split_output:
                directory=Path(a.split_output)/'_splits'/name/'spatial'/f'seed{seed}'
                prepare_scene(cube,gt,'spatial',seed,15,directory)
                previous=directory/'blocked_split.json'
                if previous.exists(): previous.rename(directory/'initial_search_blocked.json')
            item['spatial'][str(seed)]={'status':'valid',**meta}
        except SplitCoverageError as e:
            item['spatial'][str(seed)]={'status':'blocked',**e.report}
    report['datasets'][name]=item
    print(name,{k:v['status'] for k,v in item['spatial'].items()},flush=True)
Path(a.output).write_text(json.dumps(report,indent=2),encoding='utf-8')
