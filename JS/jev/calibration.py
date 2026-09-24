"""Post-hoc fitting only; callers must supply the held-out cal_fit split."""
import math
import torch
from torch.nn import functional as F


def fit_temperature(logits, labels):
    x,y = logits.detach().double().cpu(), labels.detach().cpu()
    log_t = torch.zeros((), dtype=torch.double, requires_grad=True)
    opt = torch.optim.LBFGS([log_t], lr=.5, max_iter=100, line_search_fn='strong_wolfe')
    before = F.cross_entropy(x,y).item()
    def closure():
        opt.zero_grad()
        loss = F.cross_entropy(x / log_t.clamp(-4,4).exp(), y)
        loss.backward()
        return loss
    opt.step(closure)
    t = float(log_t.detach().clamp(-4,4).exp())
    after = F.cross_entropy(x/t,y).item()
    if not math.isfinite(after) or after > before:
        t,after = 1.,before
    return t, {'nll_before':before, 'nll_after':after, 'n':len(y)}


def fit_reliability(logits, correctness):
    x,y = logits.detach().double().cpu(), correctness.detach().double().cpu()
    if y.unique().numel() < 2:
        return 1.,0.,{'status':'degenerate_single_correctness_label', 'n':len(y), 'correct':int(y.sum())}
    log_a = torch.zeros((), dtype=torch.double, requires_grad=True)
    bias = torch.zeros((), dtype=torch.double, requires_grad=True)
    opt = torch.optim.LBFGS([log_a,bias], lr=.5, max_iter=100, line_search_fn='strong_wolfe')
    before = F.binary_cross_entropy_with_logits(x,y).item()
    def closure():
        opt.zero_grad()
        loss = F.binary_cross_entropy_with_logits(log_a.clamp(-4,4).exp()*x + bias, y)
        loss.backward()
        return loss
    opt.step(closure)
    a,b = float(log_a.detach().clamp(-4,4).exp()),float(bias.detach())
    after = F.binary_cross_entropy_with_logits(a*x+b,y).item()
    if not math.isfinite(after) or after > before:
        a,b,after = 1.,0.,before
    return a,b,{'status':'fitted', 'bce_before':before, 'bce_after':after, 'n':len(y), 'correct':int(y.sum())}


def coverage_threshold(reliability, target=.9):
    if not 0 < target <= 1 or reliability.numel()==0:
        raise ValueError('Nonempty scores and coverage in (0,1] required')
    scores = reliability.detach().flatten().cpu().sort(descending=True).values
    # Retain all boundary ties: achieved calibration coverage can exceed target.
    threshold = float(scores[math.ceil(target*len(scores))-1])
    return threshold, {'target_coverage':target, 'calibration_coverage':float((scores>=threshold).float().mean()),
                       'n':len(scores), 'ties':'retain_all_at_threshold'}
