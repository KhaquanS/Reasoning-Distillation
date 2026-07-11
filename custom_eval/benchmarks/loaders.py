"""Dataset loading utilities."""

from typing import Any, Dict, List, Optional, Tuple

from datasets import load_dataset


def try_load_dataset(
    candidates: List[Dict[str, Any]],
    cache_dir: Optional[str] = None,
    split: str = "test",
) -> Tuple[Optional[Any], Dict[str, Any]]:
    """
    Try loading a dataset from multiple candidates.
    
    Args:
        candidates: List of dataset candidates with 'path', 'name', 'splits'
        cache_dir: Cache directory
        split: Preferred split name
    
    Returns:
        Tuple of (dataset, source_info)
    """
    errors = []
    
    for candidate in candidates:
        path = candidate["path"]
        name = candidate.get("name")
        splits = candidate.get("splits", [split, "test", "validation", "train"])
        
        for split_name in splits:
            try:
                ds = load_dataset(
                    path,
                    name,
                    split=split_name,
                    cache_dir=cache_dir,
                    trust_remote_code=True,
                )
                return ds, {"path": path, "name": name, "split": split_name}
            except Exception as exc:
                errors.append(f"{path}/{name or '-'}:{split_name}: {exc}")
    
    return None, {"errors": errors}


def limit_examples(examples: List, max_samples: Optional[int]) -> List:
    """Limit examples to max_samples."""
    if max_samples is None:
        return examples
    return examples[:int(max_samples)]