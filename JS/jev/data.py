"""Five disjoint centre-index splits; spatial mode also separates patch support."""
from pathlib import Path
from dataclasses import dataclass
import hashlib
import json
import numpy as np
from scipy.io import loadmat
from scipy.ndimage import minimum_filter, maximum_filter
import torch
from torch.utils.data import Dataset

SPLITS = ('train', 'val', 'cal_fit', 'cal_threshold', 'test')
PRESETS = {
 'IndianPines': ('Indian_pines_corrected', 'Indian_pines_gt', 'indian_pines_corrected', 'indian_pines_gt'),
 'PaviaU': ('PaviaU', 'PaviaU_gt', 'paviaU', 'paviaU_gt'),
 'Salinas': ('Salinas_corrected', 'Salinas_gt', 'salinas_corrected', 'salinas_gt'),
}

class SplitCoverageError(ValueError):
    def __init__(self, report):
        self.report = report
        super().__init__('Insufficient class coverage: '+json.dumps(report))


def load_array(path, key=None):
    path = Path(path)
    if path.suffix.lower() == '.npy':
        return np.load(path, allow_pickle=False)
    if path.suffix.lower() != '.mat':
        raise ValueError(f'Unsupported array file: {path}')
    try:
        arrays = {k:v for k,v in loadmat(path).items() if not k.startswith('__')}
    except (NotImplementedError, ValueError):
        import h5py
        with h5py.File(path, 'r') as f:
            arrays = {k:np.array(v).T for k,v in f.items() if isinstance(v, h5py.Dataset)}
    if key is not None:
        if key not in arrays:
            raise KeyError(f'{path}: expected {key}; available {list(arrays)}')
        return arrays[key]
    if len(arrays) != 1:
        raise ValueError(f'{path}: specify a key from {list(arrays)}')
    return next(iter(arrays.values()))


def locate_dataset(root, name):
    a,b,ak,bk = PRESETS[name]
    root = Path(root)
    def find(stem, key):
        for ext in ('.npy', '.mat'):
            candidates = sorted(p for p in root.rglob('*'+ext) if p.stem.lower() == stem.lower())
            if len(candidates) > 1:
                raise ValueError(f'Ambiguous data: {candidates}; narrow --data-root')
            if candidates:
                p = candidates[0]
                return p, None if ext == '.npy' else key
        raise FileNotFoundError(f'{stem} under {root}')
    return (*find(a, ak), *find(b, bk))


def load_scene(root, name):
    cp, ck, gp, gk = locate_dataset(root, name)
    cube, gt = np.asarray(load_array(cp, ck), dtype=np.float32), np.squeeze(load_array(gp, gk))
    if cube.ndim != 3 or gt.ndim != 2 or cube.shape[:2] != gt.shape:
        raise ValueError(f'Expected HWC cube and HW labels: {cube.shape}, {gt.shape}')
    if not np.isfinite(cube).all() or not np.isfinite(gt).all():
        raise ValueError('Nonfinite source data; do not silently impute test data')
    if not np.all(gt == gt.astype(np.int64)):
        raise ValueError('Labels must be integers')
    return cube, gt.astype(np.int64), {'cube_path':str(cp), 'cube_key':ck, 'gt_path':str(gp), 'gt_key':gk}


def _counts(n):
    # Tiny classes retain all five roles, never silently drop a class.
    if n < 7:
        raise SplitCoverageError({'class_size':n, 'minimum_required':7})
    counts = [max(2, int(np.floor(n*.01))), max(1, int(np.floor(n*.05))),
              max(1, int(np.floor(n*.025))), max(1, int(np.floor(n*.025)))]
    if sum(counts) >= n:
        raise SplitCoverageError({'class_size':n, 'requested':counts})
    return counts


def make_splits(gt, protocol, seed, patch_size=15, block_size=32):
    if patch_size % 2 != 1 or patch_size < 1:
        raise ValueError('patch_size must be positive and odd')
    labels = np.array(sorted(set(np.unique(gt))- {0}), dtype=np.int64)
    if len(labels) < 2:
        raise ValueError('At least two labeled classes required')
    rng = np.random.default_rng(seed)
    out = {k:[] for k in SPLITS}
    flat = gt.ravel()
    if protocol == 'random':
        for label in labels:
            ids = rng.permutation(np.flatnonzero(flat == label))
            a,b,c,d = _counts(len(ids))
            parts = np.split(ids, np.cumsum([a,b,c,d]))
            for k, part in zip(SPLITS, parts):
                out[k].extend(part.tolist())
        meta = {'protocol':protocol, 'patch_overlap_allowed':True}
    elif protocol == 'spatial':
        if block_size < patch_size:
            raise ValueError('block_size must be >= patch_size')
        h,w = gt.shape
        # Blocks are assigned as whole regions. Erosion excludes crossing patches.
        yy,xx = np.indices((h,w))
        blocks = (yy//block_size)*int(np.ceil(w/block_size)) + xx//block_size
        nb = int(blocks.max())+1
        best = None
        for attempt in range(256):
            assignment = rng.choice(5, nb, p=[.25,.15,.10,.10,.40])
            regions = assignment[blocks]
            lo = minimum_filter(regions, size=patch_size, mode='constant', cval=-1)
            hi = maximum_filter(regions, size=patch_size, mode='constant', cval=5)
            valid = lo == hi
            counts = np.array([[np.sum(valid & (regions==i) & (gt==l)) for l in labels] for i in range(5)])
            score = (int((counts > 0).sum()), float(np.minimum(counts, 5).sum()))
            if best is None or score > best[0]:
                best = (score, regions.copy(), valid.copy(), counts.copy(), attempt)
            if (counts > 0).all() and (counts[0] >= 2).all():
                break
        _,regions,valid,counts,attempt = best
        solver_meta = None
        if not ((counts > 0).all() and (counts[0] >= 2).all()):
            from .spatial import constrained_assignment
            solved, solver_meta = constrained_assignment(gt, labels, block_size, patch_size, seed)
            if solved is not None:
                regions = solved
                valid = (minimum_filter(regions, size=patch_size, mode='constant', cval=-1)
                         == maximum_filter(regions, size=patch_size, mode='constant', cval=5))
                counts = np.array([[np.sum(valid & (regions==i) & (gt==l)) for l in labels] for i in range(5)])
        missing = {k:labels[counts[i] < (2 if i==0 else 1)].tolist() for i,k in enumerate(SPLITS)}
        meta = {'protocol':protocol, 'patch_overlap_allowed':False, 'block_size':block_size,
                'assignment_attempt':attempt, 'available_counts':counts.tolist(), 'missing_classes':missing, 'constraint_solver':solver_meta}
        if any(missing.values()):
            raise SplitCoverageError(meta)
        for j,label in enumerate(labels):
            desired = _counts(int(np.sum(gt==label)))
            for i,k in enumerate(SPLITS):
                ids = np.flatnonzero((valid & (regions==i) & (gt==label)).ravel())
                if i < 4:
                    ids = rng.permutation(ids)[:min(len(ids), desired[i])]
                out[k].extend(ids.tolist())
    else:
        raise ValueError(protocol)
    arrays = {k:np.sort(np.asarray(v, dtype=np.int64)) for k,v in out.items()}
    validate_splits(arrays, gt.shape, patch_size, protocol == 'spatial')
    meta.update(seed=seed, patch_size=patch_size, label_values=labels.tolist(),
                counts={k:[int(np.sum(flat[v]==l)) for l in labels] for k,v in arrays.items()})
    return arrays, labels, meta


def validate_splits(splits, shape, patch_size, spatial=False):
    all_ids = np.concatenate(list(splits.values()))
    if all_ids.size != np.unique(all_ids).size:
        raise ValueError('Duplicate centres within or across splits')
    if np.any(all_ids < 0) or np.any(all_ids >= np.prod(shape)):
        raise ValueError('Out-of-bounds split indices')
    if spatial:
        from scipy.ndimage import maximum_filter
        occupied = np.zeros(shape, bool)
        for ids in splits.values():
            mask = np.zeros(shape, bool)
            mask.ravel()[ids] = True
            support = maximum_filter(mask, size=patch_size, mode='constant')
            if np.any(support & occupied):
                raise ValueError('Overlapping spatial patch support')
            occupied |= support


def fingerprint(cube, gt):
    h = hashlib.sha256()
    for x in (cube, gt):
        h.update(str(x.shape).encode()); h.update(str(x.dtype).encode())
        h.update(np.ascontiguousarray(x).data)
    return h.hexdigest()


@dataclass
class Scene:
    normalized: np.ndarray
    gt: np.ndarray
    splits: dict
    labels: np.ndarray
    mean: np.ndarray
    std: np.ndarray
    metadata: dict


def prepare_scene(cube, gt, protocol, seed, patch_size, split_dir, block_size=32):
    split_dir = Path(split_dir)
    split_dir.mkdir(parents=True, exist_ok=True)
    cache = split_dir/'split.npz'
    digest = fingerprint(cube, gt)
    if cache.exists():
        z = np.load(cache, allow_pickle=False)
        meta = json.loads(str(z['metadata']))
        expected = (digest, protocol, seed, patch_size, block_size)
        actual = (meta['data_sha256'], meta['protocol'], meta['seed'], meta['patch_size'], meta['requested_block_size'])
        if actual != expected:
            raise ValueError('Split cache differs from source/config; use a new output directory')
        splits, labels = {k:z[k] for k in SPLITS}, z['labels']
    else:
        splits, labels, meta = make_splits(gt, protocol, seed, patch_size, block_size)
        meta.update(data_sha256=digest, requested_block_size=block_size)
        np.savez_compressed(cache, **splits, labels=labels, metadata=json.dumps(meta))
        (split_dir/'split.json').write_text(json.dumps(meta, indent=2), encoding='utf-8')
    validate_splits(splits, gt.shape, patch_size, protocol=='spatial')
    # Fit only on labeled training centres, not the entire scene or test patches.
    train = cube.reshape(-1,cube.shape[-1])[splits['train']].astype(np.float64)
    mean, std = train.mean(0).astype(np.float32), train.std(0).astype(np.float32)
    std[std < 1e-6] = 1.
    normalized = ((cube-mean)/std).astype(np.float32)
    return Scene(normalized, gt, splits, labels, mean, std, meta)


class PatchDataset(Dataset):
    def __init__(self, scene, split, patch_size=15, augment=False):
        self.ids = scene.splits[split]
        self.coords = np.column_stack(np.unravel_index(self.ids, scene.gt.shape))
        self.labels = np.searchsorted(scene.labels, scene.gt.ravel()[self.ids]).astype(np.int64)
        self.patch_size, self.augment = patch_size, augment
        r = patch_size//2
        self.cube = np.pad(scene.normalized, ((r,r),(r,r),(0,0)), mode='reflect')
    def __len__(self):
        return len(self.ids)
    def __getitem__(self, i):
        y,x = self.coords[i]
        p = self.cube[y:y+self.patch_size, x:x+self.patch_size].transpose(2,0,1).copy()
        t = torch.from_numpy(p)
        if self.augment:
            if torch.rand(()) < .5: t = t.flip(-1)
            if torch.rand(()) < .5: t = t.flip(-2)
        return t, int(self.labels[i])
