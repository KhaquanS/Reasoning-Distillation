import torch
from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig
from pathlib import Path
from huggingface_hub import hf_hub_download
import shutil
import os


def _resolve_student_checkpoint_path(path, cache_dir=None):
    """
    Resolve a student checkpoint reference to a local directory path.
    
    Supports:
      - Local paths: used as-is.
      - "hf://" references to directories on the Hugging Face Hub.
    
    Downloads ONLY the necessary files from the HF repo subdirectory.
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
    
    # Create a local directory for this checkpoint
    if cache_dir:
        local_dir = Path(cache_dir) / "student_checkpoints" / subpath.replace("/", "_")
    else:
        local_dir = Path("/tmp") / "student_checkpoints" / subpath.replace("/", "_")
    local_dir.mkdir(parents=True, exist_ok=True)
    
    # Files we need to download
    needed_files = [
        "config.json",
        "generation_config.json",
        "tokenizer.json",
        "tokenizer_config.json",
        "model.safetensors",
        "aligner.pt",
        "trainer_state.pt",
        "chat_template.jinja",
    ]
    
    # Download each file
    for filename in needed_files:
        full_path = f"{subpath}/{filename}"
        target_file = local_dir / filename
        if target_file.exists():
            # Check if file is complete (not 0 bytes)
            if target_file.stat().st_size > 0:
                print(f"   ✅ Already cached: {filename}")
                continue
            else:
                print(f"   ⚠️  Found incomplete file: {filename}, re-downloading")
                target_file.unlink()
            
        try:
            # Download with retry
            downloaded = hf_hub_download(
                repo_id=repo_id,
                filename=full_path,
                revision=revision,
                repo_type="model",
                cache_dir=cache_dir,
            )
            # Copy to our local directory
            shutil.copy2(downloaded, target_file)
            print(f"   ✅ Downloaded: {filename}")
        except Exception as e:
            print(f"   ⚠️  Skipped: {filename} (not found)")
            continue
    
    # Verify required files exist
    for req in ["config.json", "model.safetensors"]:
        if not (local_dir / req).exists():
            raise FileNotFoundError(f"Required file {req} not found in {local_dir}")
    
    return local_dir


def load_student_checkpoint(checkpoint_path, device, dtype, cache_dir=None):
    """
    Load student model and aligner weights from a checkpoint.
    
    Supports:
      - Local paths: used as-is.
      - HF Hub directories: downloads ONLY the necessary files.
    
    Returns:
        (model, aligner_state_dict): The loaded student model and aligner state dict
    """
    print(f"📦 Resolving checkpoint: {checkpoint_path}")
    resolved = _resolve_student_checkpoint_path(checkpoint_path, cache_dir=cache_dir)
    print(f"   → Resolved to local path: {resolved}")
    
    if not resolved.exists():
        raise FileNotFoundError(f"Checkpoint not found: {resolved}")
    
    # Verify it's a valid model directory
    if not (resolved / "config.json").exists():
        raise ValueError(f"Invalid model directory: {resolved} (missing config.json)")
    
    # Check model.safetensors exists
    safetensors_path = resolved / "model.safetensors"
    if not safetensors_path.exists():
        raise FileNotFoundError(f"model.safetensors not found in {resolved}")
    
    # Check file size - if 0 bytes, something went wrong
    if safetensors_path.stat().st_size == 0:
        raise ValueError(f"model.safetensors is empty (0 bytes) in {resolved}")
    
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