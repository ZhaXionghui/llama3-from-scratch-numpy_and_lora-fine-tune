from typing import Optional
from dataclasses import dataclass


@dataclass
class ModelArgs:
    # @formatter:off
    # Model params for llama3
    dim: int                    = 4096      # D
    n_layers: int               = 32        # 
    n_heads: int                = 32        # QHN, HN,  ------>  HD = D // QHN  = 4096 // 32 = 128
    n_kv_heads: Optional[int]   = 8         # KVHN = 6
    vocab_size: int             = 128256    # VS
    multiple_of: int            = 1024      #
    ffn_dim_multiplier: float   = 1.3       # 
    norm_eps: float             = 1e-6      # 
    rope_theta: float           = 500000.0  # 
    max_batch_size: int         = 1         # B
    max_seq_len: int            = 128       # M
    max_new_tokens: int         = 50
    alpha: int                  = 16
    rank: int                   = 8