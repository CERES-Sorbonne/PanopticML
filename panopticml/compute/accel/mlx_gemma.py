"""EmbeddingGemma 2 vision tower on MLX (Apple GPU): ~2x PyTorch MPS on the 2304-patch
sequences, thanks to MLX's fused attention. Weights are copied from the loaded torch model."""
import mlx.core as mx
import mlx.nn as nn
import numpy as np


def _mx(t, dtype):
    return mx.array(t.detach().float().cpu().numpy()).astype(dtype)


def _rotate_half(x):
    h = x.shape[-1] // 2
    return mx.concatenate([-x[..., h:], x[..., :h]], axis=-1)


class MLXGemmaVision:
    """Gemma 4 vision tower + 3x3 pooling + embed_vision for one fixed patch grid:
    (B, P, 3*p*p) pixels in [0, 1] -> (B, T, D) soft tokens."""

    def __init__(self, model, tables: dict, dtype=mx.bfloat16):
        vt, embed = model.vision_tower, model.embed_vision
        cfg = vt.config
        self.H, self.hd, self.eps, self.dtype = cfg.num_attention_heads, cfg.head_dim, cfg.rms_norm_eps, dtype
        self.P = tables['pos'].shape[0]
        self.in_w = _mx(vt.patch_embedder.input_proj.weight, dtype).T
        self.pos = _mx(tables['pos'], dtype)
        self.cos = _mx(tables['cos'], dtype)[None, :, None, :]           # (1, P, 1, hd)
        self.sin = _mx(tables['sin'], dtype)[None, :, None, :]
        self.pool = _mx(tables['pool'], dtype)                            # (T, P)
        # HF scales by sqrt(C) before the scale-free RMSNorm: same as dividing its eps by C
        self.out_eps = embed.eps / cfg.hidden_size
        self.proj = _mx(embed.embedding_projection.weight, dtype).T
        self.ones_hd = mx.ones((self.hd,), dtype=dtype)
        self.ones_c = mx.ones((cfg.hidden_size,), dtype=dtype)
        self.layers = []
        for l in vt.encoder.layers:
            a, m = l.self_attn, l.mlp
            self.layers.append({
                'ln_in': _mx(l.input_layernorm.weight, dtype),
                'ln_post_attn': _mx(l.post_attention_layernorm.weight, dtype),
                'ln_pre_ff': _mx(l.pre_feedforward_layernorm.weight, dtype),
                'ln_post_ff': _mx(l.post_feedforward_layernorm.weight, dtype),
                'q': _mx(a.q_proj.linear.weight, dtype).T, 'k': _mx(a.k_proj.linear.weight, dtype).T,
                'v': _mx(a.v_proj.linear.weight, dtype).T, 'o': _mx(a.o_proj.linear.weight, dtype).T,
                'qn': _mx(a.q_norm.weight, dtype), 'kn': _mx(a.k_norm.weight, dtype),
                'gate': _mx(m.gate_proj.linear.weight, dtype).T, 'up': _mx(m.up_proj.linear.weight, dtype).T,
                'down': _mx(m.down_proj.linear.weight, dtype).T,
            })
        mx.eval([v for v in self.__dict__.values() if isinstance(v, mx.array)], self.layers)
        self._fn = mx.compile(self._forward)

    def _rope(self, x):
        h = self.hd // 2          # each spatial axis rotates its own half of head_dim
        c, s = self.cos, self.sin
        a = x[..., :h] * c[..., :h] + _rotate_half(x[..., :h]) * s[..., :h]
        b = x[..., h:] * c[..., h:] + _rotate_half(x[..., h:]) * s[..., h:]
        return mx.concatenate([a, b], axis=-1)

    def _forward(self, pixels):
        H, hd, eps, P = self.H, self.hd, self.eps, self.P
        B = pixels.shape[0]
        x = (2 * (pixels - 0.5)) @ self.in_w + self.pos
        for L in self.layers:
            h = mx.fast.rms_norm(x, L['ln_in'], eps)
            q = self._rope(mx.fast.rms_norm((h @ L['q']).reshape(B, P, H, hd), L['qn'], eps))
            k = self._rope(mx.fast.rms_norm((h @ L['k']).reshape(B, P, H, hd), L['kn'], eps))
            v = mx.fast.rms_norm((h @ L['v']).reshape(B, P, H, hd), self.ones_hd, eps)
            o = mx.fast.scaled_dot_product_attention(
                q.transpose(0, 2, 1, 3), k.transpose(0, 2, 1, 3), v.transpose(0, 2, 1, 3), scale=1.0)
            o = o.transpose(0, 2, 1, 3).reshape(B, P, H * hd) @ L['o']
            x = x + mx.fast.rms_norm(o, L['ln_post_attn'], eps)
            h = mx.fast.rms_norm(x, L['ln_pre_ff'], eps)
            h = (nn.gelu_approx(h @ L['gate']) * (h @ L['up'])) @ L['down']
            x = x + mx.fast.rms_norm(h, L['ln_post_ff'], eps)
        x = mx.fast.rms_norm(self.pool @ x, self.ones_c, self.out_eps)
        return x @ self.proj

    def __call__(self, pixels: np.ndarray) -> np.ndarray:
        out = self._fn(mx.array(pixels).astype(self.dtype))
        return np.array(out.astype(mx.float32))
