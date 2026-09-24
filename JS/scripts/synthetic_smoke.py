"""No claims about HSI accuracy: only learns a small separable synthetic batch."""
import argparse
import json
from pathlib import Path
import torch
from jev.model import JevSPARC
from jev.losses import classification_loss

p=argparse.ArgumentParser(); p.add_argument('--device',default='cpu'); p.add_argument('--output',default='outputs/synthetic_smoke.json')
a=p.parse_args(); torch.set_num_threads(2); torch.manual_seed(5)
y=torch.arange(3).repeat_interleave(4).to(a.device)
x=torch.randn(12,12,15,15,device=a.device)*.03
for c in range(3): x[y==c,c*4:(c+1)*4]+=2
m=JevSPARC(12,3,channels=16).to(a.device)
opt=torch.optim.AdamW(m.parameters(),lr=.002)
losses=[]
for i in range(60):
    m.train(); opt.zero_grad(); o=m(x)
    loss=classification_loss(o,y,i); loss.backward(); opt.step(); losses.append(float(loss.detach()))
m.eval()
with torch.no_grad(): accuracy=float((m(x).prediction==y).float().mean())
result={'initial_loss':losses[0],'final_loss':losses[-1],'eval_accuracy':accuracy,'steps':60,'device':a.device}
assert accuracy>=.95 and losses[-1]<losses[0]*.3,result
Path(a.output).parent.mkdir(parents=True,exist_ok=True)
Path(a.output).write_text(json.dumps(result,indent=2))
print(result)
