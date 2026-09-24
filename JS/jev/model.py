"""Structured parallel decisions. References and departures: ../RESEARCH_NOTES.md."""
from dataclasses import dataclass
from typing import Optional
import torch
from torch import nn
from torch.nn import functional as F
from models import SPARCNet

STATE_NAMES = ('raw', 'early', 'spatial', 'frequency', 'fused', 'final')

@dataclass
class DecisionOutput:
    logits: torch.Tensor
    probabilities: torch.Tensor
    prediction: torch.Tensor
    class_evidence: torch.Tensor
    reliability: torch.Tensor
    uncertainty: torch.Tensor
    abstain: torch.Tensor
    reliability_logits: torch.Tensor
    decision_tokens: torch.Tensor
    state_tokens: torch.Tensor
    attention: Optional[torch.Tensor] = None

class DecisionStateCompiler(nn.Module):
    def __init__(self, channels=64, dim=128, adaptive=True):
        super().__init__()
        self.adaptive = adaptive
        self.project = nn.ModuleDict({k: nn.Conv2d(channels, dim, 1) for k in STATE_NAMES})
        self.pool = nn.ModuleDict({k: nn.Conv2d(dim, 1, 1) for k in STATE_NAMES}) if adaptive else nn.ModuleDict()
        self.norm = nn.ModuleDict({k: nn.LayerNorm(dim) for k in STATE_NAMES})
        self.type_embedding = nn.Parameter(torch.randn(6, dim) * 0.02)

    def forward(self, states):
        if set(states) != set(STATE_NAMES):
            raise ValueError(f'Expected six states {STATE_NAMES}, got {tuple(states)}')
        tokens = []
        for i, k in enumerate(STATE_NAMES):
            x = self.project[k](states[k])
            if self.adaptive:
                # Normalized spatial weights; a deliberate variant of TokenLearner.
                w = self.pool[k](F.gelu(x)).flatten(2).softmax(-1)
                t = (x.flatten(2) * w).sum(-1)
            else:
                t = x.mean((-2, -1))
            tokens.append(self.norm[k](t) + self.type_embedding[i])
        return torch.stack(tokens, 1)

class QueryLayer(nn.Module):
    def __init__(self, dim, heads, dropout):
        super().__init__()
        self.qnorm, self.snorm, self.fnorm = (nn.LayerNorm(dim) for _ in range(3))
        self.attn = nn.MultiheadAttention(dim, heads, dropout=dropout, batch_first=True)
        self.ffn = nn.Sequential(nn.Linear(dim, 4*dim), nn.GELU(), nn.Dropout(dropout), nn.Linear(4*dim, dim))
        self.drop = nn.Dropout(dropout)

    def forward(self, q, states, diagnostics=False):
        s = self.snorm(states)
        out, weights = self.attn(self.qnorm(q), s, s, need_weights=diagnostics, average_attn_weights=False)
        q = q + self.drop(out)
        return q + self.drop(self.ffn(self.fnorm(q))), weights

class ParallelDecisionCore(nn.Module):
    def __init__(self, classes, dim=128, heads=4, depth=2, dropout=0.1):
        super().__init__()
        self.queries = nn.Parameter(torch.randn(classes, dim)*0.02)
        self.layers = nn.ModuleList([QueryLayer(dim, heads, dropout) for _ in range(depth)])
        self.norm = nn.LayerNorm(dim)

    def forward(self, states, diagnostics=False):
        q = self.queries.unsqueeze(0).expand(states.shape[0], -1, -1)
        weights = None
        for layer in self.layers:
            q, weights = layer(q, states, diagnostics)
        return self.norm(q), weights

class JevSPARC(nn.Module):
    """Variant names are predeclared; no test-driven selection of variants."""
    def __init__(self, in_channels, num_classes, variant='full', channels=64,
                 dim=128, class_counts=None, backbone_kwargs=None):
        super().__init__()
        variants = ('sparc', 'pooled_linear', 'query', 'full', 'no_adaptive', 'no_evidence', 'no_stress')
        if variant not in variants:
            raise ValueError(variant)
        self.variant, self.num_classes = variant, num_classes
        self.has_reliability = variant in ('full', 'no_adaptive', 'no_evidence', 'no_stress')
        self.has_evidence = variant in ('full', 'no_adaptive', 'no_stress')
        kw = dict(base_channels=channels, adapter_channels=channels, token_dim=96,
                  analytic_init='hybrid', use_auxiliary_heads=False)
        kw.update(backbone_kwargs or {})
        self.backbone = SPARCNet(in_channels, num_classes=num_classes if variant == 'sparc' else None,
                                 class_counts=class_counts, **kw)
        if variant != 'sparc':
            self.compiler = DecisionStateCompiler(channels, dim, adaptive=variant not in ('pooled_linear', 'no_adaptive'))
            if variant == 'pooled_linear':
                self.linear = nn.Linear(6*dim, num_classes)
            else:
                self.core = ParallelDecisionCore(num_classes, dim)
                self.choice = nn.Linear(dim, 1)
        if self.has_evidence:
            self.evidence = nn.Linear(dim, 1)
        if self.has_reliability:
            # Chosen token, mean token and four scalar summaries.
            self.reliability_head = nn.Sequential(nn.Linear(2*dim+4, dim), nn.GELU(), nn.Linear(dim, 1))
        self.register_buffer('temperature', torch.tensor(1.))
        self.register_buffer('reliability_scale', torch.tensor(1.))
        self.register_buffer('reliability_bias', torch.tensor(0.))
        self.register_buffer('abstain_threshold', torch.tensor(0.))

    def forward(self, x, diagnostics=False):
        weights = None
        if self.variant == 'sparc':
            logits = self.backbone(x)
            states = x.new_empty(x.shape[0], 0, 0)
            decisions = x.new_empty(x.shape[0], self.num_classes, 0)
        else:
            states = self.compiler(self.backbone.forward_states(x))
            if self.variant == 'pooled_linear':
                logits = self.linear(states.flatten(1))
                decisions = states.mean(1, keepdim=True).expand(-1, self.num_classes, -1)
            else:
                decisions, weights = self.core(states, diagnostics)
                logits = self.choice(decisions).squeeze(-1)
        probs = (logits / self.temperature.clamp_min(1e-3)).softmax(-1)
        pred = probs.argmax(-1)
        e = F.softplus(self.evidence(decisions).squeeze(-1)) if self.has_evidence else torch.zeros_like(logits)
        u = self.num_classes / (e + 1).sum(-1)
        if self.has_reliability:
            # A confidence loss must not learn to change its own correctness target.
            p = logits.detach().softmax(-1)
            top = p.topk(2, dim=-1).values
            entropy = -(p * p.clamp_min(1e-8).log()).sum(-1) / torch.log(p.new_tensor(self.num_classes))
            chosen = decisions[torch.arange(len(x), device=x.device), pred].detach()
            scalars = torch.stack((top[:, 0], top[:, 0]-top[:, 1], entropy, u.detach()), -1)
            rel_input = torch.cat((chosen, decisions.detach().mean(1), scalars), -1)
            rel_logit = self.reliability_head(rel_input).squeeze(-1)
            r = torch.sigmoid(self.reliability_scale * rel_logit + self.reliability_bias)
        else:
            r = probs.max(-1).values
            rel_logit = torch.logit(r.clamp(1e-6, 1-1e-6))
        return DecisionOutput(logits, probs, pred, e, r, u, r < self.abstain_threshold,
                              rel_logit, decisions, states, weights)

    def freeze_for_reliability(self):
        if not self.has_reliability:
            raise ValueError('This variant has no reliability head')
        self.eval()
        self.requires_grad_(False)
        self.reliability_head.requires_grad_(True)
        self.reliability_head.train()
