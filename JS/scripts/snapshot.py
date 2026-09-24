"""Package small, completed result artifacts while workers continue training."""
from pathlib import Path
import tarfile
root=Path('outputs/benchmark')
with tarfile.open('snapshot.tar.gz','w:gz') as tar:
    for p in root.rglob('*.json'):
        tar.add(p,arcname=str(p))
print('Snapshot ready')
