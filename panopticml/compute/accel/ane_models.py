"""Static-shape image towers for the Apple Neural Engine, converted to Core ML.

Self-contained on purpose (torch + transformers only, no relative imports): the Core ML
worker process loads this file by path, and must not import the plugin package, which
Panoptic imports under the plugin's registered name.

Layout follows Apple's ml-ane-transformers recipe: (B, C, 1, S) tensors, 1x1 convs for
linear layers, norms over the channel axis, attention split per head. Shapes are fixed at
trace time and nothing reads a tensor shape: coremltools can't convert traced shape
arithmetic. Two Neural Engine pitfalls are avoided: fractional `pow` gives wrong results
(rsqrt is used), and the Neural Engine computes in fp16 (inputs of squares are prescaled).
"""
import torch
import torch.nn.functional as F

# Bump when a model below changes: cached Core ML models of older versions are rebuilt.
ANE_MODELS_VERSION = 1

CLIP_KIND = 'clip'
GEMMA_KIND = 'embedding_gemma2'
SIGLIP_KIND = 'siglip2'

# Gemma RMSNorm inputs are divided by this before squaring: its vision activations reach
# ~1100, whose squares overflow fp16 (65504) on the Neural Engine.
RMS_PRESCALE = 32.0


def conv(linear: torch.nn.Linear) -> torch.nn.Conv2d:
    """A Linear as the equivalent 1x1 Conv2d on (B, C, 1, S) tensors."""
    c = torch.nn.Conv2d(linear.in_features, linear.out_features, 1, bias=linear.bias is not None)
    c.weight.data = linear.weight.data.float()[:, :, None, None].clone()
    if linear.bias is not None:
        c.bias.data = linear.bias.data.float().clone()
    return c


def col(t: torch.Tensor) -> torch.Tensor:
    """A per-channel vector as a (1, C, 1, 1) buffer."""
    return t.detach().float().clone().view(1, -1, 1, 1)


def attention(q, k, v, head_dim: int, scale: float):
    """q (B, C, 1, Sq), k / v (B, C, 1, Sk) -> (B, C, 1, Sq), one einsum pair per head."""
    out = []
    for qi, ki, vi in zip(q.split(head_dim, dim=1), k.transpose(1, 3).split(head_dim, dim=3),
                          v.split(head_dim, dim=1)):
        w = (torch.einsum('bchq,bkhc->bkhq', qi, ki) * scale).softmax(dim=1)    # (B, Sk, 1, Sq)
        out.append(torch.einsum('bkhq,bchk->bchq', w, vi))                      # (B, hd, 1, Sq)
    return torch.cat(out, dim=1)


# ---------------------------------------------------------------------------
# CLIP vision tower + projection
# ---------------------------------------------------------------------------

class ClipVisionANE(torch.nn.Module):
    """CLIP ViT vision tower + visual projection, as `get_image_features`.
    Input: (B, 3, H, W) RGB pixels in 0..255 (the plugin's square resize). Output: (B, D)."""

    def __init__(self, model, mean, std, size: int):
        super().__init__()
        vm = model.vision_model
        cfg = vm.config
        if cfg.hidden_act != 'quick_gelu':
            raise ValueError(f"unsupported CLIP activation {cfg.hidden_act!r}")
        C, heads = cfg.hidden_size, cfg.num_attention_heads
        self.head_dim = C // heads
        self.scale = self.head_dim ** -0.5
        emb = vm.embeddings
        pos = emb.position_embedding.weight.data.float()                                   # (1 + S, C)
        if pos.shape[0] != 1 + (size // cfg.patch_size) ** 2:
            raise ValueError(f"{size}px is not the model's input size")
        # pixels are normalized first: folded into the patch conv, the 0..255 inputs give
        # partial sums whose fp16 rounding is visible in the vectors
        self.register_buffer('mean', (255 * torch.tensor(mean, dtype=torch.float32)).view(1, 3, 1, 1))
        self.register_buffer('inv_std', (1 / (255 * torch.tensor(std, dtype=torch.float32))).view(1, 3, 1, 1))
        self.patch = torch.nn.Conv2d(3, C, emb.patch_embedding.kernel_size, emb.patch_embedding.stride, bias=False)
        self.patch.weight.data = emb.patch_embedding.weight.data.float().clone()
        self.register_buffer('cls', col(emb.class_embedding + pos[0]))                      # (1, C, 1, 1)
        self.register_buffer('pos', pos[1:].T[None, :, None, :].contiguous())               # (1, C, 1, S)
        self.pre_ln = LayerNormANE(vm.pre_layrnorm)
        self.layers = torch.nn.ModuleList()
        for l in vm.encoder.layers:
            a, mod = l.self_attn, torch.nn.Module()
            mod.ln1, mod.ln2 = LayerNormANE(l.layer_norm1), LayerNormANE(l.layer_norm2)
            mod.q, mod.k, mod.v, mod.o = conv(a.q_proj), conv(a.k_proj), conv(a.v_proj), conv(a.out_proj)
            mod.fc1, mod.fc2 = conv(l.mlp.fc1), conv(l.mlp.fc2)
            self.layers.append(mod)
        self.post_ln = LayerNormANE(vm.post_layernorm)
        self.proj = conv(model.visual_projection)

    def forward(self, pixels):                       # (B, 3, H, W)
        x = self.patch((pixels - self.mean) * self.inv_std).flatten(2).unsqueeze(2) + self.pos   # (B, C, 1, S)
        cls = x[:, :, :, :1] * 0 + self.cls          # the class token, broadcast over the batch
        x = self.pre_ln(torch.cat([cls, x], dim=3))
        for L in self.layers:
            h = L.ln1(x)
            x = x + L.o(attention(L.q(h), L.k(h), L.v(h), self.head_dim, self.scale))
            h = L.fc1(L.ln2(x))
            x = x + L.fc2(h * torch.sigmoid(1.702 * h))                                      # quick_gelu
        return self.proj(self.post_ln(x[:, :, :, :1]))[:, :, 0, 0]


# ---------------------------------------------------------------------------
# SigLIP 2 (NaFlex) vision tower
# ---------------------------------------------------------------------------

class LayerNormANE(torch.nn.Module):
    """LayerNorm over the channel axis of a (B, C, 1, S) tensor."""
    def __init__(self, ln: torch.nn.LayerNorm):
        super().__init__()
        self.register_buffer('weight', col(ln.weight))
        self.register_buffer('bias', col(ln.bias))
        self.eps = ln.eps

    def forward(self, x):
        x = x - x.mean(dim=1, keepdim=True)
        x = x * torch.rsqrt((x * x).mean(dim=1, keepdim=True) + self.eps)
        return x * self.weight + self.bias


class _SiglipMLP(torch.nn.Module):
    def __init__(self, mlp):
        super().__init__()
        self.fc1, self.fc2 = conv(mlp.fc1), conv(mlp.fc2)

    def forward(self, x):
        return self.fc2(F.gelu(self.fc1(x), approximate='tanh'))


class _SiglipLayer(torch.nn.Module):
    def __init__(self, layer, head_dim: int):
        super().__init__()
        a = layer.self_attn
        self.head_dim, self.scale = head_dim, a.scale
        self.ln1, self.ln2 = LayerNormANE(layer.layer_norm1), LayerNormANE(layer.layer_norm2)
        self.q, self.k, self.v, self.o = conv(a.q_proj), conv(a.k_proj), conv(a.v_proj), conv(a.out_proj)
        self.mlp = _SiglipMLP(layer.mlp)

    def forward(self, x):
        h = self.ln1(x)
        x = x + self.o(attention(self.q(h), self.k(h), self.v(h), self.head_dim, self.scale))
        return x + self.mlp(self.ln2(x))


class SiglipVisionANE(torch.nn.Module):
    """SigLIP 2 NaFlex vision tower for one fixed patch grid (no padding, no mask).
    Input: the processor's patches as (B, 3*p*p, 1, S). Output: the pooled (B, C) embedding,
    as `get_image_features`."""

    def __init__(self, vision, grid: tuple[int, int]):
        super().__init__()
        cfg = vision.config
        if cfg.hidden_act != 'gelu_pytorch_tanh':
            raise ValueError(f"unsupported SigLIP activation {cfg.hidden_act!r}")
        C, self.heads = cfg.hidden_size, cfg.num_attention_heads
        self.head_dim = C // self.heads
        S = grid[0] * grid[1]
        patch_dim = vision.embeddings.patch_embedding.in_features
        self.patch = conv(vision.embeddings.patch_embedding)
        with torch.no_grad():
            emb = vision.embeddings.float()
            pos = emb(torch.zeros(1, S, patch_dim), torch.tensor([list(grid)]))
            pos = pos - emb.patch_embedding.bias
        self.register_buffer('pos', pos.transpose(1, 2)[:, :, None, :].contiguous())     # (1, C, 1, S)
        self.layers = torch.nn.ModuleList(_SiglipLayer(l, self.head_dim) for l in vision.encoder.layers)
        self.post_ln = LayerNormANE(vision.post_layernorm)
        head = vision.head
        wq, wk, wv = head.attention.in_proj_weight.data.float().chunk(3)
        bq, bk, bv = head.attention.in_proj_bias.data.float().chunk(3)

        def lin(w, b):
            l = torch.nn.Linear(C, C)
            l.weight.data, l.bias.data = w.clone(), b.clone()
            return conv(l)
        self.hq, self.hk, self.hv = lin(wq, bq), lin(wk, bk), lin(wv, bv)
        self.ho = conv(head.attention.out_proj)
        self.register_buffer('probe', head.probe.data.float().transpose(1, 2)[:, :, None, :].clone())
        self.head_ln = LayerNormANE(head.layernorm)
        self.head_mlp = _SiglipMLP(head.mlp)

    def forward(self, patches):                       # (B, 3*p*p, 1, S)
        x = self.patch(patches) + self.pos
        for layer in self.layers:
            x = layer(x)
        x = self.post_ln(x)
        # attention pooling: one probe query, broadcast over the batch by the einsum
        h = self.ho(attention(self.hq(self.probe), self.hk(x), self.hv(x), self.head_dim, self.head_dim ** -0.5))
        h = h + self.head_mlp(self.head_ln(h))
        return h[:, :, 0, 0]


def siglip_grid(processor, size: int) -> tuple[int, int]:
    """(height, width) patch grid the NaFlex processor gives a size x size image."""
    import numpy as np
    out = processor(images=[np.zeros((size, size, 3), dtype=np.uint8)], return_tensors="pt")
    h, w = (int(v) for v in out['spatial_shapes'][0])
    if not bool(out['pixel_attention_mask'].all()):
        raise ValueError("square images are padded by the processor: fixed grid unsupported")
    return h, w


# ---------------------------------------------------------------------------
# EmbeddingGemma 2 (Gemma 4) vision tower + pooling + embed_vision
# ---------------------------------------------------------------------------

def gemma_tables(vision_tower, position_ids: torch.Tensor) -> dict:
    """Everything position-dependent in the Gemma 4 vision tower, for one fixed patch grid.
    position_ids: (P, 2) patch (x, y) from the processor, padding removed."""
    pe = vision_tower.patch_embedder
    k = vision_tower.config.pooling_kernel_size
    with torch.no_grad():
        table = pe.position_embedding_table.float().cpu()
        pos = table[0][position_ids[:, 0]] + table[1][position_ids[:, 1]]                   # (P, C)
        rope = vision_tower.encoder.rotary_emb
        freqs = position_ids[..., None].float() * rope.inv_freq.float().cpu()               # (P, 2, hd/4)
        cos, sin = freqs.cos(), freqs.sin()
        cos = torch.cat([cos[:, 0], cos[:, 0], cos[:, 1], cos[:, 1]], dim=-1)               # (P, hd)
        sin = torch.cat([sin[:, 0], sin[:, 0], sin[:, 1], sin[:, 1]], dim=-1)
        width = int(position_ids[:, 0].max()) + 1
        kidx = (position_ids[:, 0] // k) + (width // k) * (position_ids[:, 1] // k)
        n_out = position_ids.shape[0] // (k * k)
        pool = F.one_hot(kidx, n_out).float().T / (k * k)                                     # (T, P)
    return {'pos': pos, 'cos': cos, 'sin': sin, 'pool': pool}


def _rms_c(x, eps: float, weight=None):
    """RMSNorm over the channel axis of (B, C, 1, S), prescaled to stay in fp16 range.
    RMSNorm(x, eps) == RMSNorm(x / a, eps / a^2)."""
    x = x * (1.0 / RMS_PRESCALE)
    x = x * torch.rsqrt((x * x).mean(dim=1, keepdim=True) + eps / RMS_PRESCALE ** 2)
    return x if weight is None else x * weight


class GemmaVisionANE(torch.nn.Module):
    """Gemma 4 vision tower + 3x3 pooling + embed_vision for one fixed patch grid.
    Input: (B, 3*p*p, 1, P) pixels in [0, 1]. Output: (B, T, text_hidden) soft tokens."""

    def __init__(self, model, tables: dict):
        super().__init__()
        vt, embed = model.vision_tower, model.embed_vision
        cfg = vt.config
        if cfg.standardize or cfg.use_clipped_linears or cfg.hidden_activation != 'gelu_pytorch_tanh':
            raise ValueError("unsupported Gemma vision config")
        self.heads, self.hd, self.eps = cfg.num_attention_heads, cfg.head_dim, cfg.rms_norm_eps
        self.inp = conv(vt.patch_embedder.input_proj)
        self.register_buffer('pos', tables['pos'].T[None, :, None, :].contiguous())         # (1, C, 1, P)
        self.register_buffer('cos', tables['cos'].T[None, :, None, :].contiguous())         # (1, hd, 1, P)
        self.register_buffer('sin', tables['sin'].T[None, :, None, :].contiguous())
        self.register_buffer('poolT', tables['pool'].T.contiguous())                          # (P, T)
        # HF scales the pooled tokens by sqrt(C) before embed_vision's scale-free RMSNorm,
        # which can overflow fp16: fold the scale into that norm's eps instead.
        self.out_eps = embed.eps / cfg.hidden_size
        self.proj = conv(embed.embedding_projection)
        self.layers = torch.nn.ModuleList()
        for l in vt.encoder.layers:
            a, m = l.self_attn, l.mlp
            mod = torch.nn.Module()
            mod.q, mod.k, mod.v, mod.o = (conv(p.linear) for p in (a.q_proj, a.k_proj, a.v_proj, a.o_proj))
            mod.gate, mod.up, mod.down = (conv(p.linear) for p in (m.gate_proj, m.up_proj, m.down_proj))
            for name, t in (('ln_in', l.input_layernorm.weight), ('ln_post_attn', l.post_attention_layernorm.weight),
                            ('ln_pre_ff', l.pre_feedforward_layernorm.weight),
                            ('ln_post_ff', l.post_feedforward_layernorm.weight),
                            ('qn', a.q_norm.weight), ('kn', a.k_norm.weight)):
                mod.register_buffer(name, col(t))
            self.layers.append(mod)

    def _rope(self, x):                       # (B, hd, 1, P), one head
        h, q = self.hd // 2, self.hd // 4    # each spatial axis rotates its own half of hd

        def rot(t):
            return torch.cat((-t[:, q:], t[:, :q]), dim=1)
        c, s = self.cos, self.sin
        a = x[:, :h] * c[:, :h] + rot(x[:, :h]) * s[:, :h]
        b = x[:, h:] * c[:, h:] + rot(x[:, h:]) * s[:, h:]
        return torch.cat((a, b), dim=1)

    def _attention(self, L, h):
        out = []
        for qi, ki, vi in zip(L.q(h).split(self.hd, dim=1), L.k(h).split(self.hd, dim=1),
                              L.v(h).split(self.hd, dim=1)):
            qi = self._rope(_rms_c(qi, self.eps, L.qn))
            ki = self._rope(_rms_c(ki, self.eps, L.kn)).transpose(1, 3)                   # (B, P, 1, hd)
            vi = _rms_c(vi, self.eps)
            w = torch.einsum('bchq,bkhc->bkhq', qi, ki).softmax(dim=1)                     # scaling is 1.0
            out.append(torch.einsum('bkhq,bchk->bchq', w, vi))
        return L.o(torch.cat(out, dim=1))

    def forward(self, pixels):                # (B, 3*p*p, 1, P)
        eps = self.eps
        x = self.inp(2 * (pixels - 0.5)) + self.pos
        for L in self.layers:
            x = x + _rms_c(self._attention(L, _rms_c(x, eps, L.ln_in)), eps, L.ln_post_attn)
            h = _rms_c(x, eps, L.ln_pre_ff)
            h = L.down(F.gelu(L.gate(h), approximate='tanh') * L.up(h))
            x = x + _rms_c(h, eps, L.ln_post_ff)
        x = (x[:, :, 0, :] @ self.poolT)[:, :, None, :]                                        # (B, C, 1, T)
        x = _rms_c(x, self.out_eps)
        return self.proj(x)[:, :, 0, :].transpose(1, 2)                                        # (B, T, D)


def gemma_patches(processor, arrays) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Processor output for same-size images with the padding patches removed:
    (pixel_values (B, P, 3*p*p), position_ids (P, 2), input_ids (B, L)).
    The processor pads every image to its patch budget; padding sits at the tail."""
    out = processor(images=[[a] for a in arrays], return_tensors="pt")
    pos = out['image_position_ids']
    valid = (pos != -1).all(dim=-1)
    n = int(valid[0].sum())
    if not (bool((valid.sum(dim=1) == n).all()) and bool(valid[:, :n].all())
            and bool((pos[:, :n] == pos[:1, :n]).all())):
        raise ValueError("images don't share one patch grid")
    return out['pixel_values'][:, :n].contiguous(), pos[0, :n].contiguous(), out['input_ids']


# ---------------------------------------------------------------------------
# Build for conversion (fp32 on CPU)
# ---------------------------------------------------------------------------

def build(kind: str, model_id: str, size: int):
    """(module, example_input, check) for a size x size image. check(module) compares the
    rewrite with the HuggingFace model in fp32 and raises if they differ."""
    import numpy as np
    from transformers import AutoConfig, AutoModel, AutoProcessor

    def load(loader=AutoModel, **kw):
        try:
            return loader.from_pretrained(model_id, local_files_only=True, dtype=torch.float32, **kw)
        except Exception:
            return loader.from_pretrained(model_id, dtype=torch.float32, **kw)
    try:
        processor = AutoProcessor.from_pretrained(model_id, local_files_only=True)
    except Exception:
        processor = AutoProcessor.from_pretrained(model_id)
    rng = np.random.default_rng(0)
    arrays = [rng.integers(0, 255, (size, size, 3), dtype=np.uint8) for _ in range(2)]

    if kind == CLIP_KIND:
        from transformers import CLIPModel
        hf = load(CLIPModel, attn_implementation='eager').eval()
        ip = processor.image_processor
        module = ClipVisionANE(hf, ip.image_mean, ip.image_std, size).eval()
        example = torch.from_numpy(np.stack(arrays)).permute(0, 3, 1, 2).float().contiguous()
        mean = torch.tensor(ip.image_mean).view(1, 3, 1, 1)
        std = torch.tensor(ip.image_std).view(1, 3, 1, 1)
        with torch.no_grad():
            ref = hf.get_image_features(pixel_values=(example / 255 - mean) / std)
            ref = getattr(ref, 'pooler_output', ref)
    elif kind == SIGLIP_KIND:
        from transformers import Siglip2VisionModel
        vision = load(Siglip2VisionModel, attn_implementation='eager').eval()   # without the text tower
        module = SiglipVisionANE(vision, siglip_grid(processor, size)).eval()
        inputs = processor(images=arrays, return_tensors="pt")
        example = inputs['pixel_values'].transpose(1, 2)[:, :, None, :].contiguous()
        with torch.no_grad():
            ref = vision(**inputs).pooler_output
    elif kind == GEMMA_KIND:
        config = AutoConfig.from_pretrained(model_id, local_files_only=True, audio_config=None)
        hf = load(config=config).eval()
        pixels, position_ids, _ = gemma_patches(processor, arrays)
        module = GemmaVisionANE(hf, gemma_tables(hf.vision_tower, position_ids)).eval()
        example = pixels.transpose(1, 2)[:, :, None, :].contiguous()
        with torch.no_grad():
            pos = position_ids[None].expand(len(arrays), -1, -1)
            ref = torch.stack(hf.get_image_features(pixels, pos, return_dict=True).pooler_output)
    else:
        raise ValueError(f"no Neural Engine model for {kind!r}")

    with torch.no_grad():
        mine = module(example)
    cos = F.cosine_similarity(mine.flatten(1).float(), ref.flatten(1).float(), dim=-1).min().item()
    if cos < 0.9999:
        raise ValueError(f"Neural Engine rewrite differs from the HuggingFace model (cos {cos:.6f})")
    return module, example
