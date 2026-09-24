import torch
from torch.nn import functional as F


def evidential_mse(evidence, labels, anneal=1.):
    """Sensoy et al. 2018 Eq. 5 + annealed KL to uniform Dirichlet."""
    alpha = evidence.float() + 1
    c = alpha.shape[-1]
    y = F.one_hot(labels, c).float()
    strength = alpha.sum(-1, keepdim=True)
    mean = alpha / strength
    mse = ((y-mean).square() + alpha*(strength-alpha)/(strength.square()*(strength+1))).sum(-1)
    adjusted = y + (1-y)*alpha
    total = adjusted.sum(-1, keepdim=True)
    kl = (torch.lgamma(total).squeeze(-1) - torch.lgamma(adjusted).sum(-1)
          - torch.lgamma(alpha.new_tensor(float(c)))
          + ((adjusted-1)*(torch.digamma(adjusted)-torch.digamma(total))).sum(-1))
    return (mse + float(anneal)*kl).mean()


def classification_loss(out, labels, epoch, has_evidence=True, brier_weight=.1):
    p = out.logits.softmax(-1)
    brier = (p-F.one_hot(labels, p.shape[-1])).square().sum(-1).mean()
    loss = F.cross_entropy(out.logits, labels) + brier_weight*brier
    if has_evidence:
        loss = loss + .1*evidential_mse(out.class_evidence, labels, min(1., (epoch+1)/10))
    return loss
