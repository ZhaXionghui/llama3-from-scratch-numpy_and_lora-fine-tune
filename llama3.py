from tokenizer import Tokenizer
from config import ModelArgs
import numpy as np
import time
import sys

# 定义softmax函数
def softmax(X):
    exp_x = np.exp(X - np.max(X, axis=-1, keepdims=True))
    return exp_x/np.sum(exp_x, axis=-1, keepdims=True)

# 使用Numpy实现的silu函数（即Swish激活函数）
def silu(x):
    # 计算 sigmod(x)
    sigmod_x = 1 / (1 + np.exp(-x))
    # 计算 SiLU(x) = x* sigmoid(x)
    return x * sigmod_x

def precompute_freqs_cis(head_dim: int, max_seq_len: int, rope_theta: int = 10000):
    """
    cis(x)=cos(x)+i·sin(x)
    Args:
        head_dim: head_dim = dim // n_heads --> 4096  / 32 = 128 ------>64组
        max_seq_len: int
        rope_theta: 对应parameter `rope_theta: 500000.0`，默认值为10000
    可以看出旋转位置编码使用复数表示，比SinCos位置编码，实部代表偶数，虚部代表奇数。 cos和sin都在一个数中，因此旋转位置编码比绝对的SinCos位置编码要少一半。进一步减少内存占用
    """
    # [HD//2]
    freqs = 1.0 / (rope_theta ** (np.arange(0, head_dim, 2)[: (head_dim // 2)] / head_dim))
    # 对应freqs_for_each_token部分，这里取了max_seq_len 
    # [M, HD//2]
    freqs_for_each_token = np.outer(np.arange(max_seq_len), freqs)
    freqs_cis = np.ones_like(freqs_for_each_token) * np.exp(1j * freqs_for_each_token)
    return freqs_cis


# 将向量转为复数表示
def view_as_complex(real_np):
    shape = real_np.shape
    # 确保最后一个维度是2，表示实部和虚部
    if shape[-1]!=2:
        raise ValueError("Last dimension size must be 2 to represent real and imaginary parts.")
    # 将最后一个维度合并为复数表示
    complex_np =real_np[...,0] + 1j * real_np[..., 1]
    return complex_np

# 将向量转为实数表示
def view_as_real(complex_np):
    # 获取复数张量的形状
    shape = complex_np.shape
    # 创建一个形状为 (...,2) 的新数组，用于存储实部和虚部
    # real_np = np.zeros(shape + (2,), dtype=complex_np.dtype)
    real_np = np.zeros(shape + (2,), dtype=float)
    # 将复数数组的实部和虚部分别存储到新数组的最后一个维度
    real_np[..., 0] = np.real(complex_np)
    real_np[..., 1] = np.imag(complex_np)
    return real_np


def apply_rotary_emb(xq, xk, freqs_cis):
    # # ["B, L or 1, QHN, HD"] -> ["B, L or 1, QHN,  HD//2, 2"]
    # xqri = xq.reshape(xq.shape[:-1] + (-1, 2))
    # xkri = xk.reshape(xk.shape[:-1] + (-1, 2))

    # # Reshape `xq` and `xk` to match the complex representation.
    # xq_r, xq_i = np.split(xqri, 2, axis=-1)
    # xq_r = xq_r.squeeze(-1)
    # xq_i = xq_i.squeeze(-1)

    # xk_r, xk_i = np.split(xkri, 2, axis=-1)
    # xk_r = xk_r.squeeze(-1)
    # xk_i = xk_i.squeeze(-1)

    # # Reshape `freqs_cos` and `freqs_sin` for broadcasting.
    # freqs_cos = np.expand_dims(freqs_cos, axis=(0, 2))
    # freqs_sin = np.expand_dims(freqs_sin, axis=(0, 2))

    # # Apply rotation using real numbers.
    # xq_out_r = xq_r * freqs_cos - xq_i * freqs_sin
    # xq_out_i = xq_r * freqs_sin + xq_i * freqs_cos
    # xk_out_r = xk_r * freqs_cos - xk_i * freqs_sin
    # xk_out_i = xk_r * freqs_sin + xk_i * freqs_cos

    # # Flatten last two dimensions.合并最后两个维度
    # xq_out = np.stack([xq_out_r, xq_out_i], axis=-1)
    # xk_out = np.stack([xk_out_r, xk_out_i], axis=-1)
    # xq_out = xq_out.reshape(xq_out.shape[:-2] + (-1,))
    # xk_out = xk_out.reshape(xk_out.shape[:-2] + (-1,))

    # return xq_out, xk_out
    # 将query和key复数化，具体就是将dim维度分成两半，每一半是dim/2维，分别用作实数和虚数部分
    # [bathsize, seq_len, heads, head_dim] -> [bathsize, seq_len, heads, head_dim//2, 2]
    # ["B, L or 1, QHN,  HD"] -> ["B, L or 1, QHN,   HD//2, 2"] -> ["B, L or 1, QHN,   HD//2"]
    # ["B, L or 1, KVHN, HD"] -> ["B, L or 1, KVHN,  HD//2, 2"] -> ["B, L or 1, QHN,   HD//2"]
    xq_ = view_as_complex(xq.reshape(*xq.shape[:-1], -1, 2))
    xk_ = view_as_complex(xk.reshape(*xk.shape[:-1], -1, 2))
    # 将复数表示的query和key与频率进行点积，得到旋转后的query和key
    # 1.将频率进行广播，使其形状与query和key匹配
    # ["M, HD//2"] -> ["1, 1, M, HD//2"]
    freqs_cis = np.expand_dims(freqs_cis, axis=(0,2))
    # 2.将query和key与频率进行点积
    # ["B, L or 1, QHN,  HD//2"] * ["1, 1, M, HD//2"] -> ["B, L or 1, QHN,  M"]
    xq_out_split = xq_ * freqs_cis
    xk_out_split = xk_ * freqs_cis
    # 将旋转后的query和key转换回实数表示
    xq_out_split = view_as_real(xq_out_split)
    xk_out_split = view_as_real(xk_out_split)
    # 将旋转后的query和key合并回原来的形状
    xq_out = xq_out_split.reshape(*xq.shape[:-1], -1)
    xk_out = xk_out_split.reshape(*xk.shape[:-1], -1)
    return xq_out, xk_out

def repeat_kv(x, n_rep: int):
    if n_rep == 1:
        return x
    xs = np.repeat(x, n_rep, axis=2)
    return xs

class RMSNorm:
    def __init__(self, weight, eps: float):
        self.eps = eps
        self.weight = weight
    def forward(self, x):
        # 计算向量的平方平均值  [B, L or 1, D] -> [B, L or 1, 1]
        squared_mean = np.mean(x**2, axis=-1, keepdims=True)
        # 计算向量的均方根      [B, L or 1, 1]
        rms = np.sqrt(squared_mean + self.eps)
        # 计算归一化权重        [B, L or 1, D]
        normalized_weight = x * self.weight / rms
        return normalized_weight


class FeedForward:
    def __init__(self, up_weight, gate_weight, down_weight):
        self.up_weight = up_weight.T     # w3
        self.gate_weight = gate_weight.T # w1
        self.down_weight = down_weight.T # w2

    def forward(self, x):
        # FD = 14336
        # [B, L or 1, D] @ [D, 14336] -> [B, L or 1, 14336]
        swish = silu(x @ self.gate_weight)
        # [B, L or 1, D] @ [D, FD] -> [B, L or 1, FD]
        x_V = x @ self.up_weight
        # [B, L or 1, FD] @ [B, L or 1, FD] -> [B, L or 1, FD]
        x = swish * x_V
        # [B, L or 1, FD] @ [FD, D] -> [B, L or 1, D]
        x = x @ self.down_weight
        return x


class Attention:
    def __init__(self, wq, wk, wv, wo, args):
        # KVHN
        self.n_kv_heads = args.n_kv_heads
        # QHN, HN
        self.n_heads = args.n_heads
        # 每个KV头共享的Q头的个数 SHD = 4
        self.n_rep = self.n_heads // self.n_kv_heads
        # D // HN = 4096 // 32 = 128
        self.head_dim = args.dim // self.n_heads
        # wq: [D, D], wk: [D // 4, D], wv: [D  // 4, D], wo: [D, D]
        self.wq = wq.T
        self.wk = wk.T
        self.wv = wv.T
        self.wo = wo.T
        
        self.cache_k = np.zeros((args.max_batch_size, args.max_seq_len, self.n_kv_heads, self.head_dim))
        self.cache_v = np.zeros((args.max_batch_size, args.max_seq_len, self.n_kv_heads, self.head_dim))

    def forward(self, x, start_pos: int, mask, freqs_cis):
        # x: [B, L or 1, D]
        B, L, _ = x.shape

        # QKV
        # xq: [B, L or 1, D] @ [D, D] -> [B, L or 1, D]
        # xk: [B, L or 1, D] @ [D, D // 4] -> [B, L or 1, D // 4]
        # xv: [B, L or 1, D] @ [D, D // 4] -> [B, L or 1, D // 4]
        xq = x @ self.wq
        xk = x @ self.wk
        xv = x @ self.wv
        # 维度转换，将注意力头分离
        # xq: [B, L or 1, D]      -> [B, L or 1, HN, HD]    [1, 1, 32, 128]
        # xk: [B, L or 1, D // 4] -> [B, L or 1, KVHN, HD]  [1, 1, 8,  128]
        # xv: [B, L or 1, D // 4] -> [B, L or 1, KVHN, HD]  [1, 1, 8,  128]
        xq = xq.reshape(B, L, self.n_heads, self.head_dim)
        xk = xk.reshape(B, L, self.n_kv_heads, self.head_dim)
        xv = xv.reshape(B, L, self.n_kv_heads, self.head_dim)

        # RoPE
        xq, xk = apply_rotary_emb(xq, xk, freqs_cis)

        # KV Cache
        self.cache_k[:B, start_pos:start_pos + L] = xk
        self.cache_v[:B, start_pos:start_pos + L] = xv
        # ks: [B, L, KVHN, HD], vs: [B, L, KVHN, HD]
        ks = self.cache_k[:B, start_pos:start_pos + L]
        vs = self.cache_v[:B, start_pos:start_pos + L]
        
        # GQA
        # xk: [B, L, HN, HD], xv: [B, L, HN, HD]
        xk = repeat_kv(ks, self.n_rep)
        xv = repeat_kv(vs, self.n_rep)

        # [B, L, HN, HD] -> [B, HN, L, HD]
        xq = xq.transpose(0, 2, 1, 3)
        xk = xk.transpose(0, 2, 1, 3)
        xv = xv.transpose(0, 2, 1, 3)

        # Scaled Dot-Product Attention 乘和缩放
        # [B, HN, L or 1, HD] @ [B, HN, HD, L] -> [B, HN, L or 1, L]
        attention = xq @ xk.transpose(0, 1, 3, 2) / np.sqrt(self.head_dim)
        # `mask` is used only once at the beginning.
        if mask is not None:
            attention = attention + mask[None, None, :, :]
        attention = softmax(attention)
        # [B, HN, L or 1, L] @ [B, HN, L, HD] -> [B, HN, L or 1, HD]
        output = attention @ xv

        # [B, HN, L or 1, HD] -> [B, L or 1, HN, HD] -> [B, L or 1, D]
        output = output.transpose(0, 2, 1, 3).reshape(B, L, -1)
        # [B, L or 1, D] @ [D, D] -> [B, L or 1, D]
        output = output @ self.wo

        return output


class TransformerBlock:
    def __init__(self, layer_weights: dict, layer_id: int, args: ModelArgs):
        
        self.before_attention_rms_norm = RMSNorm(
            layer_weights[f"layers.{layer_id}.attention_norm.weight"],
            eps=args.norm_eps
        )
        self.attention = Attention(
            layer_weights[f"layers.{layer_id}.attention.wq.weight"],
            layer_weights[f"layers.{layer_id}.attention.wk.weight"],
            layer_weights[f"layers.{layer_id}.attention.wv.weight"],
            layer_weights[f"layers.{layer_id}.attention.wo.weight"],
            args
        )
        
        self.before_ffn_rms_norm = RMSNorm(
            layer_weights[f"layers.{layer_id}.ffn_norm.weight"],
            eps=args.norm_eps
        )
        self.feed_forward = FeedForward(
            layer_weights[f"layers.{layer_id}.feed_forward.w3.weight"],
            layer_weights[f"layers.{layer_id}.feed_forward.w1.weight"],
            layer_weights[f"layers.{layer_id}.feed_forward.w2.weight"]            
        )
    def forward(self, x, start_pos: int, mask, freqs_cis):
        # Attention---------------------------------------------------------
        # RMSNorm
        # [B, L or 1, D]
        norm_x = self.before_attention_rms_norm.forward(x)
        # Masked Multi-Head Attention
        # [B, L or 1, D]
        h1 = self.attention.forward(norm_x, start_pos, mask, freqs_cis)
        z = x + h1

        # Feed Forward----------------------------------------------------------
        # RMSNorm
        norm_z = self.before_ffn_rms_norm.forward(z)
        # Feed Forward + SwiGLU
        # [B, L or 1, D]
        h2 = self.feed_forward.forward(norm_z)
        out = z + h2
        return out


class Llama:
    def __init__(self, Model_Home, args):
        self.args = args
        
        self.home = Model_Home
        # tok_embeding: [VS, D]
        self.tok_embeding = np.load(self.home + "shuke/llama3.8b.shuke.tok_embeddings.weight.npz")['tok_embeddings.weight']

        # RoPE
        self.freqs_cis = precompute_freqs_cis(args.dim // args.n_heads, args.max_seq_len)

        self.layers = []
        for layer_id in range(args.n_layers):
            layer_weights = np.load(self.home + f"shuke/llama3.8b.shuke.layer.{layer_id}.npz")
            self.layers.append(TransformerBlock(layer_weights, layer_id, args))

        self.norm = RMSNorm(
            np.load(Model_Home + f"shuke/llama3.8b.shuke.norm.weight.npz")["norm.weight"],
            eps=args.norm_eps
        )
        # [D,VS]
        self.wo = np.load(self.home + "shuke/llama3.8b.shuke.output.weight.npz")['output.weight'].T

    def forward(self, input_ids, start_pos: int):
        _, L = input_ids.shape
        # [B, L or 1, D]
        h = self.tok_embeding[input_ids]
        # 构建掩码mask: [L, L]，只在开始第一次时才构建mask
        mask = None
        if L > 1:
            mask = np.full((L, L), float('-inf'))
            mask = np.triu(mask, k=1)
            mask = np.concatenate([np.zeros((L, start_pos)), mask], axis=1)
        
        # Transformer Layers
        freqs_cis = self.freqs_cis[start_pos:start_pos+L]
        for i, layer in enumerate(self.layers):
            # [B, L or 1, D]
            h = layer.forward(h, start_pos, mask, freqs_cis)
        
        # RMSNorm
        # [B, L or 1, D]
        h = self.norm.forward(h)
        # Only forward the output from the last position.
        # [B, 1, VS] = [B, 1(L), D] @ [D, VS]
        logit = h[:,[-1],:] @ self.wo
        return logit
    
    def generate(self, input_ids, max_new_tokens: int):
        L = len(input_ids)
        for i, curr_pos in enumerate(range(L, max_new_tokens)):
            if i == 0: # Prefill Phase 第一次解码
                inputs = input_ids
                pos = 0
            else: # Decode Phase 逐个解码
                inputs = next_id
                pos = curr_pos
            # [B, 1, VS]
            logits = self.forward(inputs, pos)
            next_id = logits[:,-1,:].argmax(-1,keepdims=True)
            yield next_id
            
if __name__ == "__main__":
    args = ModelArgs()
    Model_Home = "../Meta-Llama-3/Meta-Llama-3-8B/original/"

    tokenizer = Tokenizer(Model_Home + "tokenizer.model")
    model = Llama(Model_Home, args)
    
    prompt = "the answer to the ultimate question of life, the universe, and everything is " # 42
    # prompt = "山东大学（威海）， "
    print(f"{prompt}", end="")
    input_ids = [128000] + tokenizer.encode(prompt)
    input_ids = np.expand_dims(input_ids,0)
    start = time.time()
    _, L = input_ids.shape
    
    for id in model.generate(input_ids, args.max_new_tokens):
        L += 1
        output_id = id[0].tolist()
        if output_id[-1] in [1, 2]:
            break
        print(tokenizer.decode(output_id), end="")
        elapsed = time.time() - start
        print(f"\n用时: {elapsed:.2f}s", end="")
        start = time.time()
        sys.stdout.flush()
    elapsed = time.time() - start
    print(f"\n\nToken count: {L}, elapsed: {elapsed:.2f}s, {round(L/elapsed, )} tokens/s")
    # print("Prompt: ", prompt)
    # print("Tokens: ", tokenizer.encode(prompt))
    # print("Decoded: ", tokenizer.decode(tokenizer.encode(prompt)))