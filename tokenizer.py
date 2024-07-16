from pathlib import Path
import tiktoken
from tiktoken.load import load_tiktoken_bpe

class Tokenizer:
    def __init__(self, model_path: str):
        self.special_tokens = [
            "<|begin_of_text|>",
            "<|end_of_text|>",
            "<|reserved_special_token_0|>",
            "<|reserved_special_token_1|>",
            "<|reserved_special_token_2|>",
            "<|reserved_special_token_3|>",
            "<|start_header_id|>",
            "<|end_header_id|>",
            "<|reserved_special_token_4|>",
            "<|eot_id|>",  # end of turn
        ] + [f"<|reserved_special_token_{i}|>" for i in range(5, 256 - 5)]
        self.mergeable_ranks = load_tiktoken_bpe(model_path)
        self.tokenizer = tiktoken.Encoding(
        name=Path(model_path).name,
        pat_str=r"(?i:'s|'t|'re|'ve|'m|'ll|'d)|[^\r\n\p{L}\p{N}]?\p{L}+|\p{N}{1,3}| ?[^\s\p{L}\p{N}]+[\r\n]*|\s*[\r\n]+|\s+(?!\S)|\s+",
        mergeable_ranks=self.mergeable_ranks,
        special_tokens={token: len(self.mergeable_ranks) + i for i, token in enumerate(self.special_tokens)},
        )
    
    def encode(self, prompt: str) -> list[int]:
        return self.tokenizer.encode(prompt)
    
    def decode(self, ids):
        return self.tokenizer.decode(ids)
