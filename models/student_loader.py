from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig

def load_teacher(model_id, device, dtype, quantize_8bit=True, cache_dir=None):
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