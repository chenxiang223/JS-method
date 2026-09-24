"""Label-only constrained fallback for fixed spatial blocks; never sees predictions."""
import numpy as np
from scipy.ndimage import minimum_filter, maximum_filter
from scipy.optimize import milp, Bounds, LinearConstraint
from scipy.sparse import lil_matrix


def constrained_assignment(gt, labels, block_size, patch_size, seed):
    h,w=gt.shape; yy,xx=np.indices(gt.shape)
    blocks=(yy//block_size)*int(np.ceil(w/block_size))+xx//block_size
    nb=int(blocks.max())+1
    interior=(minimum_filter(blocks,size=patch_size,mode='constant',cval=-1)
              == maximum_filter(blocks,size=patch_size,mode='constant',cval=nb))
    counts=np.zeros((nb,len(labels)),dtype=int)
    for j,label in enumerate(labels):
        counts[:,j]=np.bincount(blocks[interior&(gt==label)],minlength=nb)
    support=(counts>0).sum(0)
    # Necessary condition for this conservative per-block-interior certificate.
    # Adjacent same-region blocks could admit additional centres, so this is not
    # a proof of impossibility for every conceivable geographic partition.
    if (support<5).any():
        return None,{'method':'block_interior_milp','status':'insufficient_interior_block_support',
                     'support_blocks_by_label':dict(zip(map(str,labels.tolist()),support.tolist()))}
    ncon=nb+5*len(labels); matrix=lil_matrix((ncon,nb*5),dtype=float)
    low=np.ones(ncon); high=np.full(ncon,np.inf)
    for b in range(nb):matrix[b,b*5:b*5+5]=1;high[b]=1
    for region in range(5):
        for j in range(len(labels)):
            row=nb+region*len(labels)+j
            matrix[row,np.arange(nb)*5+region]=counts[:,j]
            low[row]=2 if region==0 else 1
    rng=np.random.default_rng(seed+923)
    preferred=rng.choice(5,nb,p=[.25,.15,.1,.1,.4])
    objective=np.ones((nb,5))+rng.random((nb,5))*.001
    objective[np.arange(nb),preferred]-=1
    result=milp(objective.ravel(),integrality=np.ones(nb*5),bounds=Bounds(0,1),
                constraints=LinearConstraint(matrix.tocsr(),low,high),options={'time_limit':30.})
    meta={'method':'block_interior_milp','status':int(result.status),'message':str(result.message),
          'support_blocks_by_label':dict(zip(map(str,labels.tolist()),support.tolist()))}
    if result.x is None:return None,meta
    assignment=result.x.reshape(nb,5).argmax(1)
    certificate=np.eye(5)[assignment].ravel()
    actual=matrix.tocsr()@certificate
    if not (np.all(actual>=low-1e-6) and np.all(actual<=high+1e-6)):
        meta['status']='no_verified_feasible_assignment';return None,meta
    meta['verified_feasible']=True
    return assignment[blocks],meta
