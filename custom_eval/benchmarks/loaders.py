from datasets import load_dataset


def try_load_dataset(candidates, cache_dir=None, split="test"):
    errors = []
    for candidate in candidates:
        path = candidate["path"]
        name = candidate.get("name")
        splits = candidate.get("splits", [split, "test", "validation", "train"])
        for split_name in splits:
            try:
                ds = load_dataset(path, name, split=split_name, cache_dir=cache_dir)
                return ds, {"path": path, "name": name, "split": split_name}
            except Exception as exc:
                errors.append(f"{path}/{name or '-'}:{split_name}: {exc}")
    return None, {"errors": errors}


def limit_examples(examples, max_samples):
    if max_samples is None:
        return examples
    return examples[: int(max_samples)]

