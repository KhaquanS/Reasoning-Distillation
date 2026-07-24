from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig
from pathlib import Path
from huggingface_hub import snapshot_download


def _resolve_student_checkpoint_path(path):
    """
    Resolve a student checkpoint reference to a local directory path.
    
    Supports:
      - Local paths: used as-is.
      - "hf://" references to directories on the Hugging Face Hub.
    
    Unlike _resolve_checkpoint_path (which handles single files), this uses
    snapshot_download to fetch entire directories.
    """
    path = str(path)
    if not path.startswith("hf://"):
        return Path(path)
    
    ref = path[len("hf://"):]
    parts = ref.split("/")
    if len(parts) < 3:
        raise ValueError(
            f"Invalid hf:// reference '{path}'. Expected "
            "'hf://<namespace>/<repo_name>/<path/to/directory>'."
        )
    
    repo_id = "/".join(parts[:2])
    subpath = "/".join(parts[2:])
    revision = None
    if "@" in repo_id:
        repo_id, revision = repo_id.split("@", 1)
    
    # Download the entire repository
    local_dir = snapshot_download(
        repo_id=repo_id,
        revision=revision,
        repo_type="model",
    )
    return Path(local_dir) / subpath


def load_student_checkpoint(checkpoint_path, device, dtype, cache_dir=None):
    """
    Load student model and aligner weights from a checkpoint.
    
    Supports:
      - Local paths: used as-is.
      - HF Hub directories: downloads using snapshot_download.
    
    Returns:
        (model, aligner_state_dict): The loaded student model and aligner state dict
    """
    print(f"📦 Resolving checkpoint: {checkpoint_path}")
    resolved = _resolve_student_checkpoint_path(checkpoint_path)
    print(f"   → Resolved to local path: {resolved}")
    
    if not resolved.exists():
        raise FileNotFoundError(f"Checkpoint not found: {resolved}")
    
    # Verify it's a valid model directory
    if not (resolved / "config.json").exists():
        raise ValueError(f"Invalid model directory: {resolved} (missing config.json)")
    
    # Load the student model
    print(f"Loading student weights from {resolved} ...")
    model = AutoModelForCausalLM.from_pretrained(
        str(resolved),
        torch_dtype=dtype,
        device_map={"": 0},
        cache_dir=cache_dir
    )
    model.train()
    print("✅ Student weights loaded.")
    
    # Load aligner if it exists
    aligner_state = None
    aligner_path = resolved / "aligner.pt"
    if aligner_path.exists():
        aligner_state = torch.load(aligner_path, map_location=device)
        print("✅ Aligner weights loaded.")
    else:
        print("ℹ️  No aligner.pt found; will use fresh aligner.")
    
    return model, aligner_state


def load_teacher(model_id, device, dtype, quantize_8bit=False, cache_dir=None):
    # Default now matches scripts/reason_score.py's --teacher_quantize_8bit
    # default (False) so the two can't silently drift apart again.
    if quantize_8bit:
        bnb_config = BitsAndBytesConfig(load_in_8bit=True, llm_int8_threshold=6.0)
        model = AutoModelForCausalLM.from_pretrained(
            model_id,
            quantization_config=bnb_config,
            device_map={"": 0},
            torch_dtype=dtype,
            cache_dir=cache_dir
        )
    else:
        # quantization_config=None explicitly overrides any
        # quantization_config baked into the checkpoint's config.json (some
        # hub repos, especially large MoE models, ship with weights
        # pre-quantized). Without this, from_pretrained can silently load
        # bitsandbytes layers even though you never asked for 8-bit here --
        # which also means downstream activations may not stay bf16.
        model = AutoModelForCausalLM.from_pretrained(
            model_id,
            torch_dtype=dtype,
            quantization_config=None,
            device_map={"": 0},
            cache_dir=cache_dir
        )
    model.eval()
    for p in model.parameters():
        p.requires_grad_(False)
    return model


def load_student(model_id, device, dtype, cache_dir=None):
    model = AutoModelForCausalLM.from_pretrained(
        model_id,
        torch_dtype=dtype,
        device_map={"": 0},
        cache_dir=cache_dir
    )
    model.train()
    return model


def load_tokenizer(model_id, cache_dir=None):
    tokenizer = AutoTokenizer.from_pretrained(
        model_id,
        trust_remote_code=True,
        cache_dir=cache_dir
    )
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    return tokenizer