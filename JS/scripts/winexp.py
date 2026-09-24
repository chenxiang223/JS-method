"""Local SSH launcher. Launch keeps SSH alive; no Windows detached-process assumption."""
import argparse
import base64
import subprocess
from pathlib import Path

p=argparse.ArgumentParser()
p.add_argument('action',choices=['launch','status','fetch'])
p.add_argument('--host',default='winexp')
p.add_argument('--remote-root',default='D:/experiments/jev_sparc_20260924')
p.add_argument('--python',default='E:/anaconda/envs/LDX/python.exe')
p.add_argument('--data-root',default='C:/Users/PC/Desktop/cvoca-feinfn/cvoca-feinfn/datasets')
p.add_argument('--worker',type=int,default=0)
p.add_argument('--local-output',default='outputs/winexp')
a=p.parse_args()
def quote(s):return "'"+s.replace("'","''")+"'"
def ssh(script):
    encoded=base64.b64encode(script.encode('utf-16le')).decode()
    return subprocess.call(['ssh','-o','ConnectTimeout=10','-o','ServerAliveInterval=30','-o','ServerAliveCountMax=20',
                            a.host,'powershell -NoProfile -EncodedCommand '+encoded])
base="$ProgressPreference='SilentlyContinue'\nSet-Location "+quote(a.remote_root)+"\n$env:PYTHONPATH=(Get-Location).Path\n"
if a.action=='launch':
    command=(base+"& "+quote(a.python)+" -u -m jev.suite --data-root "+quote(a.data_root)+
             " --output outputs/benchmark --worker "+str(a.worker)+" --workers 2 --device cuda:"+str(a.worker)+"\nexit $LASTEXITCODE")
    raise SystemExit(ssh(command))
if a.action=='status':
    script=base+"& "+quote(a.python)+" scripts/compact_status.py"
    raise SystemExit(ssh(script))
if a.action=='fetch':
    rc=ssh(base+"& "+quote(a.python)+" -m jev.report outputs/benchmark\ntar -czf results.tar.gz outputs\nexit $LASTEXITCODE")
    if rc:raise SystemExit(rc)
    destination=Path(a.local_output); destination.mkdir(parents=True,exist_ok=True)
    subprocess.run(['scp',a.host+':'+a.remote_root+'/results.tar.gz',str(destination/'results.tar.gz')],check=True)
    # Archive comes from this task's controlled output path. Reject unsafe members.
    import tarfile
    with tarfile.open(destination/'results.tar.gz') as tar:
        root=destination.resolve()
        for m in tar.getmembers():
            resolved=(root/m.name).resolve()
            if not resolved.is_relative_to(root) or m.issym() or m.islnk():raise ValueError('Unsafe archive member')
        tar.extractall(destination)
