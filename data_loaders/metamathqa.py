from datasets import load_dataset
from torch.utils.data import Dataset

class MetaMathQADataset(Dataset):
    def __init__(self, split="train", cache_dir=None):
        # MetaMathQA dataset: contains 'query' and 'response'
        self.data = load_dataset(
            "meta-math/MetaMathQA",
            split=split,
            cache_dir=cache_dir,
            trust_remote_code=True
        )
        self.data = self.data.map(
            lambda ex: {"text": ex["query"] + "\n" + ex["response"]}
        )
        self.data = self.data.remove_columns(
            [c for c in self.data.column_names if c != "text"]
        )

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        return {"text": self.data[idx]["text"]}