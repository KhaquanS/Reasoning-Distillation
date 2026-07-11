"""
Standard prompt templates for each benchmark, following established practices.
"""

# ============================================================================
# Math benchmarks (MATH-500, AIME25, GSM8K)
# ============================================================================
MATH_BASE_TEMPLATE = """{question}

Please reason step by step, and put your final answer within \\boxed{{}}.

Your final answer should be in the format: \\boxed{{answer}}"""

# ============================================================================
# Multiple Choice benchmarks (ARC-C, MMLU, GPQA, HellaSwag)
# ============================================================================
CHOICE_BASE_TEMPLATE = """{question}

Please show your choice in the "answer" field with only the choice letter.

Respond with a JSON object like: {{"answer": "C"}}"""

# ============================================================================
# Individual benchmark templates
# ============================================================================

ARC_C_TEMPLATE = CHOICE_BASE_TEMPLATE

AIME25_TEMPLATE = MATH_BASE_TEMPLATE

GPQA_TEMPLATE = CHOICE_BASE_TEMPLATE

GSM8K_TEMPLATE = MATH_BASE_TEMPLATE

HELLASWAG_TEMPLATE = CHOICE_BASE_TEMPLATE

MATH500_TEMPLATE = MATH_BASE_TEMPLATE

MMLU_TEMPLATE = CHOICE_BASE_TEMPLATE


# ============================================================================
# Prompt builder function
# ============================================================================

def build_prompt(question: str, benchmark_name: str) -> str:
    """
    Build a prompt for a given benchmark using the appropriate template.
    
    Args:
        question: The question to ask
        benchmark_name: Name of the benchmark (e.g., "arc-c", "math500")
    
    Returns:
        Formatted prompt string
    """
    templates = {
        "arc-c": ARC_C_TEMPLATE,
        "math500": MATH500_TEMPLATE,
        "aime25": AIME25_TEMPLATE,
        "gsm8k": GSM8K_TEMPLATE,
        "hellaswag": HELLASWAG_TEMPLATE,
        "mmlu": MMLU_TEMPLATE,
        "gpqa": GPQA_TEMPLATE,
    }
    
    template = templates.get(benchmark_name.lower(), "{question}")
    return template.format(question=question)