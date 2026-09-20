"""
capt_biomedclip.py
==================
CAPT trainer for BiomedCLIP, registered with Dassl as ``CAPT_BiomedCLIP``.

Two-stage method (mirrors the original CAPT):
  Stage 1 (before_train): load a base prompt-tuned BiomedCLIP model (BiomedCoOp
           or CoOp checkpoint) and build the Confusion Bank over the few-shot
           training set.
  Stage 2 (train):        keep the whole backbone AND the base context frozen;
           train only the Diff-Manner Adapter (SAM) and the MGDE module.

The BiomedCLIP wrapping (TextEncoder, PromptLearner, encode_text signature,
768-d ctx / 512-d shared space) is identical to BiomedCoOp's so the base
checkpoint loads cleanly.
"""

import os.path as osp
import warnings

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.cuda.amp import GradScaler, autocast

from dassl.engine import TRAINER_REGISTRY, TrainerX
from dassl.metrics import compute_accuracy
from dassl.utils import load_pretrained_weights, load_checkpoint
from dassl.optim import build_optimizer, build_lr_scheduler

from open_clip.src.open_clip import create_model_from_pretrained, get_tokenizer
from trainers.prompt_templates import BIOMEDCOOP_TEMPLATES
from trainers.CAPT.capt_modules import (
    ConfusionBank, SemanticConfusion, DiffMannerAdapter, MGDE,
    encode_classwise_template_embeddings, confusion_infonce,
)

warnings.filterwarnings("ignore")

BIOMEDCLIP_HF = "hf-hub:microsoft/BiomedCLIP-PubMedBERT_256-vit_base_patch16_224"


# ---------------------------------------------------------------------------
# BiomedCLIP text side (same as BiomedCoOp)
# ---------------------------------------------------------------------------
class TextEncoder(nn.Module):
    def __init__(self, biomedclip_model):
        super().__init__()
        self.model = biomedclip_model
        self.dtype = biomedclip_model.text.transformer.dtype

    def forward(self, prompts, tokenized_prompts):
        return self.model.encode_text(prompts, True, tokenized_prompts)


class PromptLearner(nn.Module):
    """Identical structure to BiomedCoOp/CoOp so a base ctx checkpoint loads."""

    def __init__(self, cfg, classnames, biomedclip_model):
        super().__init__()
        n_cls = len(classnames)
        n_ctx = cfg.TRAINER.CAPT.N_CTX
        ctx_init = cfg.TRAINER.CAPT.CTX_INIT
        dtype = biomedclip_model.text.transformer.dtype
        ctx_dim = 768
        self.tokenizer = get_tokenizer(BIOMEDCLIP_HF)

        if ctx_init and n_ctx == 4:
            ctx_init = ctx_init.replace("_", " ")
            prompt = self.tokenizer(ctx_init)
            with torch.no_grad():
                device = biomedclip_model.text.transformer.embeddings.word_embeddings.weight.device
                prompt = prompt.to(device)
                
                embedding = biomedclip_model.text.transformer.embeddings.word_embeddings(prompt).type(dtype)
            ctx_vectors = embedding[0, 1: 1 + n_ctx, :]
            prompt_prefix = ctx_init
        else:
            if cfg.TRAINER.CAPT.CSC:
                ctx_vectors = torch.empty(n_cls, n_ctx, ctx_dim, dtype=dtype)
            else:
                ctx_vectors = torch.empty(n_ctx, ctx_dim, dtype=dtype)
            nn.init.normal_(ctx_vectors, std=0.02)
            prompt_prefix = " ".join(["X"] * n_ctx)

        print(f'Initial context: "{prompt_prefix}"  (n_ctx={n_ctx})')
        self.ctx = nn.Parameter(ctx_vectors)                    # frozen in CAPT

        classnames = [name.replace("_", " ") for name in classnames]
        self.name_lens = [len(self.tokenizer(name)) for name in classnames]
        prompts = [prompt_prefix + " " + name + "." for name in classnames]
        tokenized_prompts = torch.cat([self.tokenizer(p) for p in prompts])
        with torch.no_grad():
            embedding = biomedclip_model.text.transformer.embeddings.word_embeddings(tokenized_prompts).type(dtype)

        self.register_buffer("token_prefix", embedding[:, :1, :])
        self.register_buffer("token_suffix", embedding[:, 1 + n_ctx:, :])
        self.n_cls = n_cls
        self.n_ctx = n_ctx
        self.tokenized_prompts = tokenized_prompts
        self.class_token_position = cfg.TRAINER.CAPT.CLASS_TOKEN_POSITION

    def forward(self):
        ctx = self.ctx
        if ctx.dim() == 2:
            ctx = ctx.unsqueeze(0).expand(self.n_cls, -1, -1)
        prefix, suffix = self.token_prefix, self.token_suffix
        if self.class_token_position == "end":
            prompts = torch.cat([prefix, ctx, suffix], dim=1)
        else:
            raise ValueError("CAPT config uses CLASS_TOKEN_POSITION=end")
        return prompts


# ---------------------------------------------------------------------------
# CAPT model
# ---------------------------------------------------------------------------
class CustomCLIP(nn.Module):
    def __init__(self, cfg, classnames, biomedclip_model):
        super().__init__()
        self.cfg = cfg
        self.n_cls = len(classnames)
        self.prompt_learner = PromptLearner(cfg, classnames, biomedclip_model)
        self.tokenized_prompts = self.prompt_learner.tokenized_prompts
        self.image_encoder = biomedclip_model.visual
        self.text_encoder = TextEncoder(biomedclip_model)
        self.logit_scale = biomedclip_model.logit_scale
        self.dtype = biomedclip_model.text.transformer.dtype

        device = next(biomedclip_model.parameters()).device
        tokenizer = get_tokenizer(BIOMEDCLIP_HF)
        class_emb = encode_classwise_template_embeddings(
            biomedclip_model, tokenizer, BIOMEDCOOP_TEMPLATES, classnames, device
        )                                                        # (n_cls, d) frozen
        d = class_emb.shape[1]

        # ----- CAPT trainable modules -----
        self.sem = SemanticConfusion(self.n_cls, class_emb, k_pairs=cfg.TRAINER.CAPT.K_PAIRS)
        self.sam = DiffMannerAdapter(dim=768, grid=14,
                                     s=cfg.TRAINER.CAPT.ALPHA_S,
                                     gamma=cfg.TRAINER.CAPT.ALPHA_GAMMA)
        self.sam_proj = nn.Linear(768, d)                       # 768-d tokens -> 512-d
        self.fuse = nn.Linear(3 * d, d)                         # img + sample + semantic
        self.mgde = MGDE(dim=d, topk=cfg.TRAINER.CAPT.MGDE_TOPK)
        self.beta = cfg.TRAINER.CAPT.BETA                       # residual strength

        # warm-start semantic experts from prompt statistics
        comm = F.normalize(class_emb.mean(0, keepdim=True), dim=-1).squeeze(0)
        self.mgde.init_semantic_experts(comm.unsqueeze(0), comm.unsqueeze(0))

        self.bank = None                                        # set in before_train

    # -- ViT patch tokens from the frozen timm trunk --
    @torch.no_grad()
    def _tokens(self, image):
        image = image.type(self.dtype)
        try:
            tok = self.image_encoder.trunk.forward_features(image)   # (B, 197, 768)
        except Exception:
            # fallback: no tokens available -> fabricate a 1-token sequence
            pooled = self.image_encoder(image)
            tok = pooled.unsqueeze(1)
        cls, patches = tok[:, 0], tok[:, 1:]
        return cls, patches

    def encode_image(self, image):
        feat = self.image_encoder(image.type(self.dtype))
        return F.normalize(feat, dim=-1)

    def text_features(self):
        prompts = self.prompt_learner()
        tf = self.text_encoder(prompts, self.tokenized_prompts)
        return F.normalize(tf, dim=-1)

    def forward(self, image, label=None):
        logit_scale = self.logit_scale
        image = image.type(self.dtype)

        text_feat = self.text_features()                         # (C, d)
        img_feat = self.encode_image(image)                      # (B, d)
        base_logits = logit_scale.exp() * img_feat @ text_feat.t()
        conf = base_logits.softmax(1)

        # --- SEM: pseudo-GT + confusion pairs ---
        use_stats = self.training and (self.bank is not None)
        count = self.bank.count if use_stats else None
        pseudo, pairs = self.sem.pairs_and_scores(conf, count)
        sem_feat = self.sem.semantic_feature(pseudo, pairs)      # (B, d)

        # --- SAM: representative confusing samples + Diff-Manner Adapter ---
        if self.training and self.bank is not None:
            conf_feat, sim = self.bank.representative(img_feat.detach(), pairs)
        else:
            # inference: no bank indexing (paper: inference needs no bank)
            conf_feat = torch.zeros_like(img_feat)
            sim = torch.full((img_feat.shape[0],), 0.5, device=img_feat.device)

        cls_tok, patch_tok = self._tokens(image)
        sample_feat = self.sam(cls_tok, patch_tok, sim)          # (B, 768)
        sample_feat = F.normalize(self.sam_proj(sample_feat), dim=-1)
        sample_feat = F.normalize(sample_feat + conf_feat, dim=-1)

        # --- MGDE: fuse granularities -> residual on the image feature ---
        fused = self.fuse(torch.cat([img_feat, sample_feat, sem_feat], dim=1))
        residual = self.mgde(F.normalize(fused, dim=-1))         # (B, d)
        refined = F.normalize(img_feat + self.beta * residual, dim=-1)

        logits = logit_scale.exp() * refined @ text_feat.t()

        if self.training:
            loss_ce = F.cross_entropy(logits, label)
            loss_conf = confusion_infonce(refined, text_feat, label, pairs, logit_scale)
            loss = loss_ce + self.cfg.TRAINER.CAPT.CONF_LAMBDA * loss_conf
            return logits, loss
        return logits


@TRAINER_REGISTRY.register()
class CAPT_BiomedCLIP(TrainerX):

    def check_cfg(self, cfg):
        assert cfg.TRAINER.CAPT.PREC in ["fp16", "fp32", "amp"]

    def build_model(self):
        cfg = self.cfg
        classnames = self.dm.dataset.classnames

        print(f"Loading BiomedCLIP (backbone: {cfg.MODEL.BACKBONE.NAME})")
        biomedclip_model, _ = create_model_from_pretrained(BIOMEDCLIP_HF)
        if cfg.TRAINER.CAPT.PREC in ("fp32", "amp"):
            biomedclip_model.float()
        biomedclip_model = biomedclip_model.to(self.device).eval()

        print("Building CAPT model")
        self.model = CustomCLIP(cfg, classnames, biomedclip_model)

        # Load the base prompt context (Stage 1 model) if given, then freeze it.
        base_ckpt = cfg.TRAINER.CAPT.BASE_CKPT
        if base_ckpt:
            self._load_base_ctx(base_ckpt, cfg.TRAINER.CAPT.BASE_EPOCH)

        print("Freezing backbone + base context; training CAPT modules only")
        trainable = ("sam", "sam_proj", "fuse", "mgde")
        for name, param in self.model.named_parameters():
            param.requires_grad_(name.split(".")[0] in trainable)
        enabled = [n for n, p in self.model.named_parameters() if p.requires_grad]
        print(f"Trainable parameter groups: {sorted(set(n.split('.')[0] for n in enabled))}")

        if cfg.MODEL.INIT_WEIGHTS:
            load_pretrained_weights(self.model, cfg.MODEL.INIT_WEIGHTS)

        self.model.to(self.device)

        capt_params = nn.ModuleDict({
            "sam": self.model.sam, "sam_proj": self.model.sam_proj,
            "fuse": self.model.fuse, "mgde": self.model.mgde,
        })
        self.optim = build_optimizer(capt_params, cfg.OPTIM)
        self.sched = build_lr_scheduler(self.optim, cfg.OPTIM)
        self.register_model("CAPT", capt_params, self.optim, self.sched)
        self.scaler = GradScaler() if cfg.TRAINER.CAPT.PREC == "amp" else None

        if torch.cuda.device_count() > 1:
            print(f"Multiple GPUs detected ({torch.cuda.device_count()}); using DataParallel")
            self.model = nn.DataParallel(self.model)

    # -----------------------------------------------------------------
    # Stage 1: build the Confusion Bank from the base model
    # -----------------------------------------------------------------
    def before_train(self):
        super().before_train()
        model = self.model.module if isinstance(self.model, nn.DataParallel) else self.model
        n_cls = model.n_cls
        print("Building Confusion Bank from the base model over the training set ...")
        feats, trues, preds = [], [], []
        model.eval()
        with torch.no_grad():
            for batch in self.train_loader_x:
                image = batch["img"].to(self.device)
                label = batch["label"].to(self.device)
                img_feat = model.encode_image(image)
                text_feat = model.text_features()
                logits = model.logit_scale.exp() * img_feat @ text_feat.t()
                feats.append(img_feat.cpu())
                trues.append(label.cpu())
                preds.append(logits.argmax(1).cpu())
        feats = torch.cat(feats); trues = torch.cat(trues); preds = torch.cat(preds)
        bank = ConfusionBank(n_cls).build(feats, trues, preds).to(self.device)
        model.bank = bank
        acc = (preds == trues).float().mean().item() * 100
        print(f"Confusion Bank: {len(trues)} samples, base train acc {acc:.2f}%, "
              f"{int(bank.n_into.sum())} confused samples recorded.")
        model.train()

    def forward_backward(self, batch):
        image, label = self.parse_batch_train(batch)
        prec = self.cfg.TRAINER.CAPT.PREC
        if prec == "amp":
            with autocast():
                logits, loss = self.model(image, label)
            self.optim.zero_grad()
            self.scaler.scale(loss).backward()
            self.scaler.step(self.optim)
            self.scaler.update()
        else:
            logits, loss = self.model(image, label)
            self.model_backward_and_update(loss)

        loss_summary = {"loss": loss.item(),
                        "acc": compute_accuracy(logits, label)[0].item()}
        if (self.batch_idx + 1) == self.num_batches:
            self.update_lr()
        return loss_summary

    def parse_batch_train(self, batch):
        return batch["img"].to(self.device), batch["label"].to(self.device)

    # -----------------------------------------------------------------
    def _load_base_ctx(self, directory, epoch):
        model_file = "model-best.pth.tar" if epoch is None else f"model.pth.tar-{epoch}"
        # BiomedCoOp/CoOp save under a "prompt_learner" sub-dir
        for sub in ("prompt_learner", "CAPT", ""):
            path = osp.join(directory, sub, model_file) if sub else osp.join(directory, model_file)
            if osp.exists(path):
                break
        else:
            print(f"[CAPT] base checkpoint not found in {directory}; using zero-shot ctx.")
            return
        state = load_checkpoint(path)["state_dict"]
        ctx = {k: v for k, v in state.items() if k.endswith("ctx") or k == "ctx"}
        m = self.model
        with torch.no_grad():
            for k, v in ctx.items():
                if m.prompt_learner.ctx.shape == v.shape:
                    m.prompt_learner.ctx.copy_(v.to(self.device))
                    print(f"[CAPT] loaded base context from {path}")
                    return
        print(f"[CAPT] base ctx shape mismatch; using zero-shot ctx.")

    def load_model(self, directory, epoch=None):
        if not directory:
            print("Note that load_model() is skipped as no pretrained model is given")
            return
        model_file = "model-best.pth.tar" if epoch is None else f"model.pth.tar-{epoch}"
        for name in self.get_model_names():
            path = osp.join(directory, name, model_file)
            if not osp.exists(path):
                raise FileNotFoundError(f'Model not found at "{path}"')
            checkpoint = load_checkpoint(path)
            state_dict = checkpoint["state_dict"]
            print(f"Loading weights to {name} from {path} (epoch={checkpoint['epoch']})")
            self._models[name].load_state_dict(state_dict, strict=False)
