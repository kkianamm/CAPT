"""
capt_modules.py
================
Building blocks for CAPT (Confusion-Aware Prompt Tuning, CVPR 2026) adapted to
BiomedCLIP (PubMedBERT + ViT-B/16) on top of the BiomedCoOp codebase.

The four ideas from the paper are implemented here:

  * ConfusionBank       -> stores base-model predictions/features and the
                           inter-class confusion count matrix (Sec. 3.2).
  * SemanticConfusion   -> SEM: pseudo-GT, confusion score (Eq. 7), and
                           commonality / difference prompt embeddings.
  * DiffMannerAdapter   -> the global (ViT attention) + local (depthwise conv)
                           adapter of SAM (Eq. 9-10) with dynamic alpha (Eq. 13).
  * MGDE                -> Multi-Granularity Discrepancy Expert, a small MoE
                           over commonality / difference / sample experts (Eq. 11).

Design notes / honest deviations from the original repo
-------------------------------------------------------
* The original CAPT builds its Confusion Bank from a PromptKD teacher. There is
  no public ViT-L BiomedCLIP teacher, so we build the bank from a *base
  prompt-tuned BiomedCLIP model* (BiomedCoOp or CoOp). The CAPT appendix (their
  Table 9) shows the bank can be built from CoOp/MaPLe/CLIP/TAC as well, so this
  is a supported instantiation rather than a hack.
* The original CAPT generates commonality/difference prompts with a live LLM
  (chain-of-thought). We instead reuse the GPT-4 prompt ensemble that BiomedCoOp
  already ships (``BIOMEDCOOP_TEMPLATES``, 50 clinical prompts / class) and form
  commonality = norm(e_a + e_b) and difference = norm(e_a - e_b). This keeps the
  semantics biomedical and the pipeline fully offline.
* At inference the Confusion Bank is NOT queried (the paper states inference
  needs no indexing); the adapter runs on the query's own tokens. This is what
  lets the CAPT modules trained on base classes transfer to novel classes.

All tensors are kept in the 512-d BiomedCLIP shared space unless noted; the
Diff-Manner Adapter works in the 768-d ViT token space and projects to 512-d.
"""

import math
import torch
import torch.nn as nn
import torch.nn.functional as F


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------
@torch.no_grad()
def encode_classwise_template_embeddings(biomedclip_model, tokenizer,
                                         templates_per_class, classnames, device):
    """Mean template embedding per class, L2-normalized -> (n_cls, d).

    ``templates_per_class`` is BiomedCoOp's ``BIOMEDCOOP_TEMPLATES`` dict
    (classname -> list of prompt strings). Falls back to "a photo of a {c}."
    for any class that is missing.
    """
    embs = []
    for c in classnames:
        key = c.replace("_", " ")
        prompts = templates_per_class.get(key, [f"a photo of a {key}."])
        toks = torch.cat([tokenizer(p) for p in prompts]).to(device)
        feats = biomedclip_model.encode_text(toks)          # (P, d)
        feats = F.normalize(feats, dim=-1).mean(0)
        embs.append(F.normalize(feats, dim=-1))
    return torch.stack(embs, 0)                             # (n_cls, d)


# ---------------------------------------------------------------------------
# Confusion Bank (Sec. 3.2)
# ---------------------------------------------------------------------------
class ConfusionBank:
    """Holds base-model image features, (true, pseudo-GT) labels and the
    inter-class confusion count matrix. Built once, before CAPT training."""

    def __init__(self, n_cls):
        self.n_cls = n_cls
        self.feats = None                       # (N, d) L2-normalized, cpu
        self.true = None                        # (N,)
        self.pred = None                        # (N,) pseudo-GT (argmax base logit)
        self.count = torch.zeros(n_cls, n_cls)  # count[t, p]
        self.into = {}                          # class -> indices that leaked into it
        self.n_into = torch.zeros(n_cls)        # #misclassified samples per target class

    @torch.no_grad()
    def build(self, feats, true, pred):
        self.feats = F.normalize(feats.float(), dim=-1).cpu()
        self.true = true.cpu().long()
        self.pred = pred.cpu().long()
        self.count.zero_()
        for t, p in zip(self.true.tolist(), self.pred.tolist()):
            self.count[t, p] += 1
        for c in range(self.n_cls):
            self.into[c] = ((self.pred == c) & (self.true != c)).nonzero(as_tuple=True)[0]
        self.n_into = torch.tensor(
            [len(self.into[c]) for c in range(self.n_cls)], dtype=torch.float
        )
        return self

    def to(self, device):
        if self.feats is not None:
            self.feats = self.feats.to(device)
        self.count = self.count.to(device)
        self.n_into = self.n_into.to(device)
        return self

    @torch.no_grad()
    def representative(self, query_feats, pair_classes):
        """For each query and each of its confusion-pair classes, return the
        most representative confusing sample (max cosine sim) from the bank.

        query_feats : (B, d) L2-normalized
        pair_classes: (B, k) long, confusion-pair class ids
        returns
            conf_feat : (B, d)  mean of the retrieved representatives
            max_sim   : (B,)    similarity of the best representative (for alpha)
        """
        B, k = pair_classes.shape
        d = query_feats.shape[1]
        device = query_feats.device
        conf = torch.zeros(B, d, device=device)
        best = torch.zeros(B, device=device)
        counts = torch.zeros(B, device=device)
        for b in range(B):
            acc = torch.zeros(d, device=device)
            n = 0
            for j in range(k):
                c = int(pair_classes[b, j])
                idx = self.into.get(c, None)
                if idx is None or len(idx) == 0:
                    continue
                cand = self.feats[idx].to(device)               # (m, d)
                sims = cand @ query_feats[b]                    # (m,)
                m = int(sims.argmax())
                acc += cand[m]
                best[b] = max(best[b], float(sims[m]))
                n += 1
            if n > 0:
                conf[b] = acc / n
                counts[b] = n
        return conf, best


# ---------------------------------------------------------------------------
# Semantic Confusion Miner (SEM, Sec. 3.2)
# ---------------------------------------------------------------------------
class SemanticConfusion(nn.Module):
    """Computes confusion scores, top-k confusion pairs and the commonality /
    difference semantic features for a batch. Non-parametric; the learnable
    part lives in MGDE."""

    def __init__(self, n_cls, class_emb, k_pairs=3):
        super().__init__()
        self.n_cls = n_cls
        self.k = min(k_pairs, n_cls - 1)
        # class_emb: (n_cls, d) frozen template embeddings
        self.register_buffer("class_emb", class_emb)

    @torch.no_grad()
    def pairs_and_scores(self, confidence, count=None):
        """confidence: (B, n_cls) softmax probs of the base/current model.
        count: optional (n_cls, n_cls) confusion counts. Returns pseudo-GT (B,),
        confusion-pair classes (B, k)."""
        pseudo = confidence.argmax(1)                          # (B,)
        # confusion weight per class i for a query with pseudo-GT p:
        #   w_i = 1 + count[i, p] / sum_t count[t, p]     (Eq. 7)
        if count is not None:
            col = count[:, pseudo].t()                         # (B, n_cls): count[i, p]
            denom = col.sum(1, keepdim=True).clamp_min(1.0)
            w = 1.0 + col / denom
        else:
            w = torch.ones_like(confidence)
        score = w * confidence                                 # (B, n_cls)
        # exclude the pseudo-GT itself, then take top-k
        score = score.scatter(1, pseudo.unsqueeze(1), float("-inf"))
        pairs = score.topk(self.k, dim=1).indices              # (B, k)
        return pseudo, pairs

    def semantic_feature(self, pseudo, pairs):
        """Mean commonality+difference embedding over the query's confusion
        pairs (pseudo, q). Returns (B, d)."""
        e = self.class_emb                                     # (n_cls, d)
        ep = e[pseudo]                                         # (B, d)
        eq = e[pairs]                                          # (B, k, d)
        ep_ = ep.unsqueeze(1)
        comm = F.normalize(ep_ + eq, dim=-1)                   # (B, k, d)
        diff = F.normalize(ep_ - eq, dim=-1)                   # (B, k, d)
        feat = 0.5 * (comm + diff).mean(1)                     # (B, d)
        return F.normalize(feat, dim=-1)


# ---------------------------------------------------------------------------
# Diff-Manner Adapter (SAM, Eq. 9-10, 13)
# ---------------------------------------------------------------------------
class DiffMannerAdapter(nn.Module):
    """Global ViT-attention branch + local depthwise-conv branch with a
    confusion-driven dynamic weight alpha. Operates on ViT patch tokens."""

    def __init__(self, dim=768, grid=14, n_heads=8, s=5.0, gamma=0.5):
        super().__init__()
        self.grid = grid
        self.ln = nn.LayerNorm(dim)
        self.attn = nn.MultiheadAttention(dim, n_heads, batch_first=True)
        self.dwconv = nn.Conv2d(dim, dim, kernel_size=3, padding=1, groups=dim)
        self.s = s
        self.gamma = gamma

    def dynamic_alpha(self, sim):
        # alpha = s * c^gamma, c = confusion intensity in [0, 1]   (Eq. 13)
        c = sim.clamp(min=1e-4, max=1.0)
        return (self.s * c.pow(self.gamma)).unsqueeze(1)          # (B, 1)

    def forward(self, cls, patches, sim):
        """cls: (B, D), patches: (B, N, D), sim: (B,) confusion intensity."""
        B, N, D = patches.shape
        q = self.ln(cls).unsqueeze(1)                            # (B, 1, D)
        kv = self.ln(patches)                                    # (B, N, D)
        glob, _ = self.attn(q, kv, kv)                           # (B, 1, D) Eq. 9
        x = (patches + glob).transpose(1, 2).reshape(B, D, self.grid, self.grid)
        local = self.dwconv(x).flatten(2).transpose(1, 2)        # (B, N, D)
        alpha = self.dynamic_alpha(sim)                          # (B, 1)
        tokens = patches + alpha.unsqueeze(1) * local            # Eq. 10
        return tokens.mean(1)                                    # (B, D) pooled


# ---------------------------------------------------------------------------
# Multi-Granularity Discrepancy Expert (MGDE, Eq. 11)
# ---------------------------------------------------------------------------
class MGDE(nn.Module):
    """Small Mixture-of-Experts fusing commonality / difference / sample
    granularities with a top-k router."""

    def __init__(self, dim=512, topk=2):
        super().__init__()
        self.experts = nn.ModuleList([nn.Linear(dim, dim) for _ in range(3)])
        self.router = nn.Linear(dim, 3, bias=False)
        nn.init.normal_(self.router.weight, std=0.02)
        self.topk = min(topk, 3)

    @torch.no_grad()
    def init_semantic_experts(self, comm_emb, diff_emb):
        """Warm-start the two semantic experts from prompt statistics so they
        start near the commonality / difference subspaces (CAPT initializes the
        semantic experts from the difference/commonality prompt embeddings)."""
        if comm_emb is not None:
            self.experts[0].bias.copy_(comm_emb.mean(0))
        if diff_emb is not None:
            self.experts[1].bias.copy_(diff_emb.mean(0))

    def forward(self, f):
        logits = self.router(f)                                 # (B, 3)
        masked = torch.full_like(logits, float("-inf"))
        val, idx = logits.topk(self.topk, dim=-1)
        masked.scatter_(-1, idx, val)
        w = F.softmax(masked, dim=-1)                           # (B, 3)
        outs = torch.stack([e(f) for e in self.experts], dim=1)  # (B, 3, d)
        return (w.unsqueeze(-1) * outs).sum(1)                   # (B, d)


# ---------------------------------------------------------------------------
# Confusion InfoNCE loss (Eq. 3-5)
# ---------------------------------------------------------------------------
def confusion_infonce(img_feat, text_feat, label, pairs, logit_scale):
    """Symmetric InfoNCE restricted to each sample's {true class} U {confusion
    pairs}. Encourages separating confusable classes at the instance level.

    img_feat : (B, d) L2-normalized refined image features
    text_feat: (C, d) L2-normalized class text features
    label    : (B,)   ground-truth class ids
    pairs    : (B, k)  confusion-pair class ids
    """
    B = img_feat.shape[0]
    device = img_feat.device
    scale = logit_scale.exp()
    loss = img_feat.new_zeros(())
    n = 0
    for b in range(B):
        cls_ids = torch.cat([label[b].view(1), pairs[b]]).unique()
        if cls_ids.numel() < 2:
            continue
        sub = text_feat[cls_ids]                                # (c, d)
        tgt = (cls_ids == label[b]).float().argmax()            # index of true class
        # image -> text
        logit_i2t = scale * (img_feat[b] @ sub.t())            # (c,)
        loss = loss + F.cross_entropy(logit_i2t.unsqueeze(0), tgt.view(1))
        # text -> image (symmetric term, positive is the same sample)
        logit_t2i = scale * (sub[tgt] @ img_feat.t())          # (B,)
        loss = loss + F.cross_entropy(logit_t2i.unsqueeze(0),
                                      torch.tensor([b], device=device))
        n += 1
    return loss / max(n, 1)
