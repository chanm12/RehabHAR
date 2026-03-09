"""
RehabHAR – Text Encoder (BERT-based Long-Document Embedder)
Based on: LanHAR models/model.py (BERT embedding logic)

Extracts the BERT long-text embedding logic from the original monolithic
model class into a standalone TextEncoder module. Supports sliding-window
chunked encoding for documents longer than max_len.
"""

import torch
import torch.nn as nn
from transformers import AutoModel


class TextEncoder(nn.Module):
    """BERT-based text encoder with sliding-window support for long texts.

    For sequences shorter than max_len, runs a single BERT forward pass.
    For longer sequences, splits into overlapping chunks and produces a
    weighted-average embedding.
    """

    def __init__(self, bert_model="allenai/scibert_scivocab_uncased",
                 max_len=512, stride=128, pool="mean", pad_id=0):
        super().__init__()
        self.bert = AutoModel.from_pretrained(bert_model)
        self.d_model = self.bert.config.hidden_size
        self.max_len = max_len
        self.stride = stride
        self.pool = pool
        self.pad_id = pad_id

    def _pool_hidden(self, last_hidden_state: torch.Tensor, attention_mask: torch.Tensor):
        if self.pool == "cls":
            return last_hidden_state[:, 0, :]
        mask = attention_mask.unsqueeze(-1).type_as(last_hidden_state)  # (B,L,1)
        summed = (last_hidden_state * mask).sum(dim=1)                  # (B,H)
        denom = mask.sum(dim=1).clamp(min=1e-6)                         # (B,1)
        return summed / denom

    def _bert_embed_long(self, input_ids: torch.Tensor, attention_mask: torch.Tensor):
        B, L = input_ids.size()
        device = input_ids.device
        if L <= self.max_len:
            out = self.bert(input_ids=input_ids, attention_mask=attention_mask)
            return self._pool_hidden(out.last_hidden_state, attention_mask)  # (B,H)

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
                    ids_chunk = torch.cat([ids_chunk, torch.full((pad_len,), self.pad_id, dtype=ids_chunk.dtype, device=device)], dim=0)
                    mask_chunk = torch.cat([mask_chunk, torch.zeros(pad_len, dtype=mask_chunk.dtype, device=device)], dim=0)

                ids_chunk = ids_chunk.unsqueeze(0)
                mask_chunk = mask_chunk.unsqueeze(0)

                out = self.bert(input_ids=ids_chunk, attention_mask=mask_chunk)
                emb = self._pool_hidden(out.last_hidden_state, mask_chunk).squeeze(0)  # (H,)
                parts.append(emb)
                weights.append(valid_len)

                if end == L:
                    break
                start += step

            parts = torch.stack(parts, dim=0)                                # (C,H)
            weights = torch.tensor(weights, device=device, dtype=parts.dtype)  # (C,)
            weights = (weights / weights.sum().clamp(min=1e-6)).unsqueeze(-1)  # (C,1)
            doc_emb = (parts * weights).sum(dim=0)                              # (H,)
            chunk_embs.append(doc_emb)
        return torch.stack(chunk_embs, dim=0)  # (B,H)

    def forward(self, input_ids: torch.Tensor, attention_mask: torch.Tensor):
        """Forward pass — returns pooled embeddings of shape (B, H)."""
        return self._bert_embed_long(input_ids, attention_mask)
