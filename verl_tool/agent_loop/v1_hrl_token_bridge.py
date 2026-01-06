from typing import Tuple


class TokenBridge:
    """Maintains text transcript and re-encodes per-model."""

    def __init__(self, prompt_text: str) -> None:
        self.prompt_text = prompt_text or ""
        self.response_text = ""

    @property
    def transcript_text(self) -> str:
        return self.prompt_text + self.response_text

    def encode(self, tokenizer) -> list[int]:
        """Tokenize current transcript with provided tokenizer."""
        return tokenizer.encode(self.transcript_text, add_special_tokens=False)

    def encode_response(self, tokenizer) -> list[int]:
        """Tokenize only the generated response with provided tokenizer."""
        return tokenizer.encode(self.response_text, add_special_tokens=False)

    def response_token_length(self, tokenizer) -> int:
        """Count response tokens for a tokenizer."""
        return len(self.encode_response(tokenizer))

    def remaining_budget(self, tokenizer, max_total_tokens: int) -> int:
        used = len(self.encode_response(tokenizer))
        return max(0, max_total_tokens - used)

    def append_from_tokens(self, tokenizer, token_ids: list[int]) -> str:
        delta_text = tokenizer.decode(token_ids, skip_special_tokens=False)
        self.response_text += delta_text
        return delta_text

    def finalize_tokens(
        self, tokenizer, max_total_tokens: int
    ) -> Tuple[list[int], list[int], list[int]]:
        prompt_ids = tokenizer.encode(self.prompt_text, add_special_tokens=False)
        response_ids = tokenizer.encode(self.response_text, add_special_tokens=False)
        response_ids = response_ids[:max_total_tokens]
        response_mask = [1] * len(response_ids)
        return prompt_ids, response_ids, response_mask

