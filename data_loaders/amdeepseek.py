"""
AM-DeepSeek-R1-Distilled-1.4M dataset loader.

This dataset contains 1.4 million high-quality reasoning traces with:
- 0.5M entries from open-source datasets
- 0.9M entries distilled from DeepSeek-R1-671B

The dataset includes mathematics, code, scientific Q&A, and general chat tasks.
https://huggingface.co/datasets/a-m-team/AM-DeepSeek-R1-Distilled-1.4M
"""

from datasets import load_dataset
from torch.utils.data import Dataset


class AMDeepSeekDataset(Dataset):
    """
    Dataset loader for AM-DeepSeek-R1-Distilled-1.4M.

    The dataset uses a 'messages' field containing a list of conversation turns,
    each with 'role' (user/assistant) and 'content' fields.

    For distillation, we concatenate the user instruction and assistant response
    into a single 'text' field, matching the format expected by the training pipeline.
    """

    def __init__(self, split="train", cache_dir=None, max_samples=None):
        """
        Args:
            split (str): Dataset split to load (default: "train").
            cache_dir (str): Cache directory for the dataset.
            max_samples (int, optional): Maximum number of samples to load.
                Useful for quick testing or subset selection.
        """
        self.data = load_dataset(
            "a-m-team/AM-DeepSeek-R1-Distilled-1.4M",
            "am_0.9M",
            split=split,
            cache_dir=cache_dir,
        )

        # Optionally limit the number of samples
        if max_samples is not None and max_samples < len(self.data):
            self.data = self.data.select(range(max_samples))

        # Format: extract user instruction + assistant response into a single text field
        # The dataset stores messages as a list of {role, content} dicts
        self.data = self.data.map(
            self._format_example,
            remove_columns=self.data.column_names
        )

    def _format_example(self, example):
        """
        Extract and format a single example from the dataset.

        The dataset structure (from README):
        {
            "messages": [
                {"role": "user", "content": "...", "info": {...}},
                {"role": "assistant", "content": "...", "info": {...}}
            ]
        }

        We combine the user instruction and assistant response into a single text.
        """
        messages = example.get("messages", [])

        user_content = ""
        assistant_content = ""

        for msg in messages:
            role = msg.get("role", "")
            content = msg.get("content", "")
            if role == "user":
                user_content = content
            elif role == "assistant":
                assistant_content = content

        # Format: "instruction\nresponse" (matching Math500 and MetaMathQA format)
        text = user_content + "\n" + assistant_content

        return {"text": text}

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        return {"text": self.data[idx]["text"]}
