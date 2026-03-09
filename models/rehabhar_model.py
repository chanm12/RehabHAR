"""
RehabHAR – Main Model: Multi-Modal Sensor–Text Alignment
Based on: LanHAR models/model.py (lanhar class)

Combines TextEncoder (BERT) and TimeSeriesTransformer (sensor encoder)
with projection heads and contrastive alignment. Renamed from `lanhar`
to `RehabHARModel`.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from transformers import AutoTokenizer, AutoModel

from models.sensor_encoder import TimeSeriesTransformer
from models.losses import TripletLoss, SimpleModel


class RehabHARModel(nn.Module):
    """
    Multi-modal model that embeds sensor time-series and text descriptions
    into a shared latent space for contrastive learning.

    Forward pass takes two text inputs (analysis + label semantics),
    a sensor time-series, four auxiliary text inputs for triplet sampling,
    and a label tensor.
    """

    def __init__(self, bert_model="allenai/scibert_scivocab_uncased",
                 max_len=512, stride=128, pool="mean",
                 embed_dim=128, input_dim=6, d_model=128,
                 nhead=2, num_layers=2, dim_feedforward=256,
                 dropout=0.1, pad_id=0):
        super().__init__()

        # ── Text backbone ────────────────────────────────────────────────
        self.bert = AutoModel.from_pretrained(bert_model)
        self.d_bert = self.bert.config.hidden_size
        self.max_len = max_len
        self.stride = stride
        self.pool = pool
        self.pad_id = pad_id

        # ── Sensor backbone ──────────────────────────────────────────────
        self.sensor_encoder = TimeSeriesTransformer(
            input_dim=input_dim, d_model=d_model, nhead=nhead,
            num_encoder_layers=num_layers,
            dim_feedforward=dim_feedforward, dropout=dropout
        )

        # ── Projection heads ────────────────────────────────────────────
        self.txt_proj = nn.Linear(self.d_bert, embed_dim)
        self.sen_proj = nn.Linear(d_model, embed_dim)

        # ── Learnable temperature ────────────────────────────────────────
        self.logit_scale = nn.Parameter(torch.ones([]) * 2.6592)

    # ── BERT helpers ─────────────────────────────────────────────────────

    def _pool_hidden(self, last_hidden_state, attention_mask):
        if self.pool == "cls":
            return last_hidden_state[:, 0, :]
        mask = attention_mask.unsqueeze(-1).type_as(last_hidden_state)
        summed = (last_hidden_state * mask).sum(dim=1)
        denom = mask.sum(dim=1).clamp(min=1e-6)
        return summed / denom

    def _bert_embed_long(self, input_ids, attention_mask):
        B, L = input_ids.size()
        device = input_ids.device
        if L <= self.max_len:
            out = self.bert(input_ids=input_ids, attention_mask=attention_mask)
            return self._pool_hidden(out.last_hidden_state, attention_mask)

        chunk_embs = []
        step = self.max_len - self.stride if self.max_len > self.stride else self.max_len
        for b in range(B):
            ids_b = input_ids[b]
            mask_b = attention_mask[b]
            parts, weights = [], []
            start = 0
            while start < L:
                end = min(start + self.max_len, L)
                ids_chunk = ids_b[start:end]
                mask_chunk = mask_b[start:end]
                valid_len = mask_chunk.sum().item()
                pad_len = self.max_len - (end - start)
                if pad_len > 0:
                    ids_chunk = torch.cat([ids_chunk, torch.full((pad_len,), self.pad_id, dtype=ids_chunk.dtype, device=device)])
                    mask_chunk = torch.cat([mask_chunk, torch.zeros(pad_len, dtype=mask_chunk.dtype, device=device)])
                ids_chunk = ids_chunk.unsqueeze(0)
                mask_chunk = mask_chunk.unsqueeze(0)
                out = self.bert(input_ids=ids_chunk, attention_mask=mask_chunk)
                emb = self._pool_hidden(out.last_hidden_state, mask_chunk).squeeze(0)
                parts.append(emb)
                weights.append(valid_len)
                if end == L:
                    break
                start += step
            parts = torch.stack(parts, dim=0)
            weights = torch.tensor(weights, device=device, dtype=parts.dtype)
            weights = (weights / weights.sum().clamp(min=1e-6)).unsqueeze(-1)
            doc_emb = (parts * weights).sum(dim=0)
            chunk_embs.append(doc_emb)
        return torch.stack(chunk_embs, dim=0)

    # ── Forward ──────────────────────────────────────────────────────────

    def forward(self,
                input_ids1, attention_mask1,   # text 1 (analysis prompt)
                input_ids2, attention_mask2,   # text 2 (label semantics)
                time_series,                    # (B,T,6)
                input_ids3, attention_mask3,   # triplet anchor text
                input_ids4, attention_mask4,   # triplet positive text
                input_ids5, attention_mask5,   # triplet negative text 1
                input_ids6, attention_mask6,   # triplet negative text 2
                labels):

        # Text embeddings (BERT)
        text_emb1 = self._bert_embed_long(input_ids1, attention_mask1)  # (B, H)
        text_emb2 = self._bert_embed_long(input_ids2, attention_mask2)  # (B, H)

        # Sensor embeddings
        sensor_emb = self.sensor_encoder(time_series)  # (B, T, D)

        # Triplet text embeddings
        anchor_emb_1 = self._bert_embed_long(input_ids3, attention_mask3)
        positive_emb_1 = self._bert_embed_long(input_ids4, attention_mask4)
        negative_emb_1 = self._bert_embed_long(input_ids5, attention_mask5)

        anchor_emb_2 = self._bert_embed_long(input_ids3, attention_mask3)
        positive_emb_2 = self._bert_embed_long(input_ids4, attention_mask4)
        negative_emb_2 = self._bert_embed_long(input_ids6, attention_mask6)

        # Projections for contrastive alignment
        text_vec = F.normalize(self.txt_proj(text_emb1), dim=-1)           # (B, E)
        sensor_vec = F.normalize(self.sen_proj(sensor_emb.sum(dim=1)), dim=-1)  # (B, E)

        return (
            text_emb1, text_emb2,
            sensor_emb,
            anchor_emb_1, positive_emb_1, negative_emb_1,
            anchor_emb_2, positive_emb_2, negative_emb_2,
            labels,
            text_vec, sensor_vec, self.logit_scale
        )
