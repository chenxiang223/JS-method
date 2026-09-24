from pathlib import Path
import json
root=Path('outputs/benchmark')
completed=list(root.glob('*/*/seed*/*/result.json'))
print('Completed:',len(completed),flush=True)
for p in sorted(root.glob('worker*_status.json')):
    v=json.loads(p.read_text())
    cmd=v.get('command',[])
    if cmd:
        get=lambda k:cmd[cmd.index(k)+1]
        key=[get(k) for k in ('--dataset','--protocol','--seed','--variant')]
        status=root/key[0]/key[1]/('seed'+key[2])/key[3]/'status.json'
        print(p.stem,key,json.loads(status.read_text()) if status.exists() else v,flush=True)
    else:print(p.stem,v,flush=True)
