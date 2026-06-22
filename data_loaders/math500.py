from datasets import load_dataset
from torch.utils.data import Dataset

class Math500Dataset(Dataset):
    def __init__(self, split="train", cache_dir=None):
        # Using 'rasbt/math_full_minus_math500' – it has 'train' split.
        # The dataset contains 'problem' and 'solution' fields.
        self.data = load_dataset(
            "rasbt/math_full_minus_math500",
            split=split,
            cache_dir=cache_dir,
            trust_remote_code=True
        )
        # Format: we want a single 'text' field with prompt + solution.
        # For distillation we need both teacher and student to process the same input;
        # we can use the problem as input and the solution as target for task loss.
        # For alignment, we only need the input text.
        self.data = self.data.map(
            lambda ex: {"text": ex["problem"] + "\n" + ex["solution"]}
        )
        # Remove other columns
        self.data = self.data.remove_columns(
            [c for c in self.data.column_names if c != "text"]
        )

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        return {"text": self.data[idx]["text"]}