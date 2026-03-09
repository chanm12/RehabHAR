"""
RehabHAR – Loss Functions for Contrastive and Triplet Training
Based on: LanHAR models/model.py (loss functions section)

Provides:
  - SimpleModel: L2-normalisation wrapper
  - TripletLoss: triplet margin loss
  - symmetric_cross_entropy / custom_loss: symmetric CE helpers
  - clip_loss: standard CLIP-style contrastive loss
  - clip_loss_multipos: multi-positive CLIP loss with soft label support
"""

import torch
import torch.nn as nn
import torch.nn.functional as F


class SimpleModel(nn.Module):
    """L2-normalise input embeddings."""
    def __init__(self):
        super().__init__()

    def forward(self, x):
        return F.normalize(x, p=2, dim=1)


class TripletLoss(nn.Module):
    def __init__(self, margin=1.0):
        super().__init__()
        self.loss_fn = nn.TripletMarginLoss(margin=margin, p=2)

    def forward(self, anchor, positive, negative):
        return self.loss_fn(anchor, positive, negative)


# ── Symmetric Cross-Entropy helpers ─────────────────────────────────────

def symmetric_cross_entropy(logits, labels):
    loss_i = F.cross_entropy(logits, labels)
    logits_t = logits.transpose(0, 1)
    loss_t = F.cross_entropy(logits_t, labels)
    return (loss_i + loss_t) / 2


def symmetric_cross_entropy1(logits, labels):
    return F.cross_entropy(logits, labels)


def custom_loss(similarity_matrix, t):
    labels = torch.arange(similarity_matrix.size(0), device=similarity_matrix.device)
    return symmetric_cross_entropy(similarity_matrix, labels)


def custom_loss1(similarity_matrix, labels):
    labels = labels.to(similarity_matrix.device)
    return symmetric_cross_entropy1(similarity_matrix, labels)


# ── CLIP-style contrastive losses ───────────────────────────────────────

def clip_loss(image_features, text_features, logit_scale=None, temperature=None):
    if logit_scale is None:
        assert temperature is not None, "pass logit_scale or temperature"
        scale = 1.0 / max(temperature, 1e-6)
    else:
        scale = logit_scale.exp().clamp(max=100.0)

    logits_per_image = (image_features @ text_features.T) * scale
    logits_per_text = (text_features @ image_features.T) * scale
    labels = torch.arange(image_features.size(0), device=image_features.device)
    loss_img = F.cross_entropy(logits_per_image, labels)
    loss_txt = F.cross_entropy(logits_per_text, labels)
    return 0.5 * (loss_img + loss_txt)


def clip_loss_multipos(
    z_a, z_b, labels, logit_scale=None, temperature=None, eps=1e-12
):
    dev = z_a.device
    z_b = z_b.to(dev)
    labels = labels.to(dev)

    if logit_scale is None:
        assert temperature is not None
        scale = 1.0 / max(float(temperature), 1e-6)
        scale = torch.tensor(scale, device=dev)
    else:
        scale = logit_scale.to(dev).exp().clamp(max=100.0)

    logits_ab = (z_a @ z_b.t()) * scale      # (B,B)
    logits_ba = (z_b @ z_a.t()) * scale      # (B,B)

    same = labels.unsqueeze(1).eq(labels.unsqueeze(0))  # (B,B) bool
    target_ab = same.float()
    target_ba = same.t().float()

    target_ab = target_ab / (target_ab.sum(dim=1, keepdim=True) + eps)
    target_ba = target_ba / (target_ba.sum(dim=1, keepdim=True) + eps)

    log_q_ab = logits_ab - torch.logsumexp(logits_ab, dim=1, keepdim=True)
    log_q_ba = logits_ba - torch.logsumexp(logits_ba, dim=1, keepdim=True)

    loss_ab = -(target_ab * log_q_ab).sum(dim=1).mean()
    loss_ba = -(target_ba * log_q_ba).sum(dim=1).mean()
    return 0.5 * (loss_ab + loss_ba)
