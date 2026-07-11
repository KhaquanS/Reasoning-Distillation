"""
Qwen-specific chat formatting utilities following Qwen best practices.
"""

from typing import Any, Dict, List, Optional


class QwenChatFormatter:
    """
    Formats prompts for Qwen models using the official chat template.
    
    Qwen models expect messages in a specific format:
    - Each message has a 'role' (system, user, assistant)
    - The chat template handles thinking/non-thinking modes
    - Historical model output should only include final output, not thinking content
    """
    
    @staticmethod
    def format_messages(
        prompt: str,
        system_prompt: Optional[str] = None,
        enable_thinking: Optional[bool] = None,
    ) -> List[Dict[str, str]]:
        """
        Format a prompt into Qwen-compatible messages.
        
        Args:
            prompt: The user prompt
            system_prompt: Optional system message
            enable_thinking: Whether to enable thinking mode (None = use model default)
        
        Returns:
            List of message dicts with 'role' and 'content' keys
        """
        messages = []
        
        if system_prompt:
            messages.append({"role": "system", "content": system_prompt})
        
        # Add the user message with the prompt
        messages.append({"role": "user", "content": prompt})
        
        return messages
    
    @staticmethod
    def create_generation_kwargs(
        enable_thinking: bool,
        temperature: float = 1.0,
        top_p: float = 0.95,
        top_k: int = 20,
        max_new_tokens: int = 32768,
        presence_penalty: float = 1.5,
        repetition_penalty: float = 1.0,
    ) -> Dict[str, Any]:
        """
        Create generation kwargs following Qwen best practices.
        
        Args:
            enable_thinking: Whether to enable thinking mode
            temperature: Sampling temperature
            top_p: Top-p sampling parameter
            top_k: Top-k sampling parameter
            max_new_tokens: Maximum tokens to generate
            presence_penalty: Presence penalty
            repetition_penalty: Repetition penalty
        
        Returns:
            Dictionary of generation parameters
        """
        kwargs = {
            "temperature": temperature,
            "top_p": top_p,
            "top_k": top_k,
            "max_new_tokens": max_new_tokens,
            "repetition_penalty": repetition_penalty,
            "presence_penalty": presence_penalty,
            "do_sample": temperature > 0.0,
        }
        
        # For Qwen, thinking mode is controlled via chat template kwargs
        # The model will use its default if not specified
        if enable_thinking is not None:
            # This is framework-specific; for HF transformers, we use the chat template
            kwargs["chat_template_kwargs"] = {"enable_thinking": enable_thinking}
        
        return kwargs
    
    @staticmethod
    def extract_response(raw_output: str) -> str:
        """
        Extract the final response from Qwen output.
        
        Qwen outputs may include thinking content in <think> tags when in thinking mode.
        This extracts just the final response part.
        
        Args:
            raw_output: Raw model output text
        
        Returns:
            Extracted final response
        """
        import re
        
        # Look for thinking tags and extract content after them
        think_pattern = re.compile(r"<think\s*>\s*(.*?)\s*</think\s*>\s*(.*)", re.IGNORECASE | re.DOTALL)
        match = think_pattern.search(raw_output)
        
        if match:
            # Return the content after the think tag
            return match.group(2).strip()
        
        # If no think tag, return the whole output
        return raw_output.strip()