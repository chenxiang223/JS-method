"""Perturb normalized inputs only. All random draws use a caller-owned generator."""
import torch

KINDS = ('noise', 'band_dropout', 'band_mask', 'spatial_occlusion')


def perturb(x, kind, severity, generator):
    if kind not in KINDS or severity not in (1, 2, 3):
        raise ValueError((kind, severity))
    z = x.clone()
    b, c, h, w = z.shape
    def rand(*shape):
        return torch.rand(shape, generator=generator, device=x.device)
    if kind == 'noise':
        z += torch.randn(z.shape, generator=generator, device=z.device, dtype=z.dtype) * (.05, .15, .3)[severity-1]
    elif kind == 'band_dropout':
        z *= (rand(b, c, 1, 1) >= (.1, .25, .4)[severity-1]).to(z.dtype)
    elif kind == 'band_mask':
        length = max(1, int(c*(.1, .25, .4)[severity-1]))
        starts = (rand(b)*(c-length+1)).long()
        bands = torch.arange(c, device=x.device)[None, :]
        keep = ~((bands >= starts[:, None]) & (bands < starts[:, None]+length))
        z *= keep[:, :, None, None]
    else:
        side = max(1, int(min(h,w)*(.2, .35, .5)[severity-1]))
        starts_h, starts_w = (rand(b)*(h-side+1)).long(), (rand(b)*(w-side+1)).long()
        rows = torch.arange(h, device=x.device)[None, :, None]
        cols = torch.arange(w, device=x.device)[None, None, :]
        mask = ((rows >= starts_h[:,None,None]) & (rows < starts_h[:,None,None]+side)
                & (cols >= starts_w[:,None,None]) & (cols < starts_w[:,None,None]+side))
        mask[:, h//2, w//2] = False
        z.masked_fill_(mask[:,None], 0)
    return z
