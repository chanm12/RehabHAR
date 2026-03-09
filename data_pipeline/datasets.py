"""
RehabHAR – PyTorch Dataset Classes
Based on: LanHAR models/dataset.py

Provides:
  - AllPairsDataset: dataset for stage-2 training / evaluation (4-field tuples)
  - AllPairsDatasetContrastive: dataset for stage-1 contrastive training (8-field tuples)
"""

import torch
from torch.utils.data import Dataset


class AllPairsDataset(Dataset):
    def __init__(self, data, tokenizer):
        self.data = data
        self.tokenizer = tokenizer

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        time_series, s1, s2, label = self.data[idx]
        e1 = self.tokenizer(s1, add_special_tokens=True, truncation=False)
        e2 = self.tokenizer(s2, add_special_tokens=True, truncation=False)

        input_ids1 = torch.tensor(e1["input_ids"], dtype=torch.long)
        attn_mask1 = torch.tensor(e1["attention_mask"], dtype=torch.long)
        input_ids2 = torch.tensor(e2["input_ids"], dtype=torch.long)
        attn_mask2 = torch.tensor(e2["attention_mask"], dtype=torch.long)

        return (
            torch.as_tensor(time_series).float(),
            input_ids1, attn_mask1,
            input_ids2, attn_mask2,
            torch.tensor(label, dtype=torch.long)
        )


class AllPairsDatasetContrastive(Dataset):
    def __init__(self, data, tokenizer):
        self.data = data
        self.tokenizer = tokenizer

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        time_series, s1, s2, label, s3, s4, s5, s6 = self.data[idx]

        def enc(text):
            e = self.tokenizer(text, add_special_tokens=True, truncation=False)
            return (torch.tensor(e["input_ids"], dtype=torch.long),
                    torch.tensor(e["attention_mask"], dtype=torch.long))

        input_ids1, attn_mask1 = enc(s1)
        input_ids2, attn_mask2 = enc(s2)
        input_ids3, attn_mask3 = enc(s3)
        input_ids4, attn_mask4 = enc(s4)
        input_ids5, attn_mask5 = enc(s5)
        input_ids6, attn_mask6 = enc(s6)

        return (
            torch.as_tensor(time_series).float(),
            input_ids1, attn_mask1,
            input_ids2, attn_mask2,
            torch.tensor(label, dtype=torch.long),
            input_ids3, attn_mask3,
            input_ids4, attn_mask4,
            input_ids5, attn_mask5,
            input_ids6, attn_mask6
        )
