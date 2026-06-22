import torch
from torch.utils.data import Dataset
from transformers import PreTrainedTokenizer

def collate_fn(batch, tokenizer, max_length):
    texts = [item["text"] for item in batch]
    return tokenizer(
        texts,
        padding=True,
        truncation=True,
        max_length=max_length,
        return_tensors="pt"
    )