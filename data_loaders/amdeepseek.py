"""
AM-DeepSeek-R1-Distilled-1.4M dataset loader.

This dataset contains 1.4 million high-quality reasoning traces with:
- 0.5M entries from open-source datasets
- 0.9M entries distilled from DeepSeek-R1-671B

The dataset includes mathematics, code, scientific Q&A, and general chat tasks.
https://huggingface.co/datasets/a-m-team/AM-DeepSeek-R1-Distilled-1.4M
"""

from datasets import Features, Value, load_dataset
from torch.utils.data import Dataset


class AMDeepSeekDataset(Dataset):
    """
    Dataset loader for AM-DeepSeek-R1-Distilled-1.4M.

    The dataset uses a 'messages' field containing a list of conversation turns,
    each with 'role' (user/assistant) and 'content' fields.

    For distillation, we concatenate the user instruction and assistant response
    into a single 'text' field, matching the format expected by the training pipeline.
    """

    DATASET_ID = "a-m-team/AM-DeepSeek-R1-Distilled-1.4M"
    CONFIG_NAME = "am_0.9M"

    def __init__(self, split="train", cache_dir=None, max_samples=None, skip_samples=0):
        """
        Args:
            split (str): Dataset split to load (default: "train").
            cache_dir (str): Cache directory for the dataset.
            max_samples (int, optional): Maximum number of samples to load.
                Useful for quick testing or subset selection.
            skip_samples (int): Number of samples to skip from the start of
                the (unshuffled) dataset before taking `max_samples`. Use
                this when resuming training on a fresh slice of data — e.g.
                after training on the first 40_000 samples, set
                skip_samples=40_000 to train on the *next* max_samples
                examples instead of re-selecting samples[0:max_samples],
                which would silently repeat data the model already saw.
        """
        self.data = load_dataset(
            self.DATASET_ID,
            self.CONFIG_NAME,
            split=split,
            cache_dir=cache_dir,
            features=self._features(),
        )

        # Optionally skip already-seen samples and/or limit how many to load
        if skip_samples:
            end = len(self.data) if max_samples is None else skip_samples + max_samples
            end = min(end, len(self.data))
            self.data = self.data.select(range(skip_samples, end))
        elif max_samples is not None and max_samples < len(self.data):
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

    @staticmethod
    def _features():
        # The dataset card documents test_case as a string. Some rows contain
        # non-null test_case values, so declaring it as null makes Arrow fail
        # during JSON import around row 87188.
        return Features({
            "messages": [{
                "role": Value("string"),
                "content": Value("string"),
                "info": {
                    "source": Value("string"),
                    "reference_answer": Value("string"),
                    "test_case": Value("string"),
                    "think_content": Value("string"),
                    "answer_content": Value("string"),
                },
            }],
        })

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        return {"text": self.data[idx]["text"]}