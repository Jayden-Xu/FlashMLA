
import torch
import math
from flash_mla.ops.interface_fused import flash_mla_prefill_fused, flash_mla_decode_fused


def apply_rotary_interleaved(x: torch.Tensor, cos: torch.Tensor, sin: torch.Tensor):

    # reshape to pairs: [..., D//2, 2]
    shape = x.shape
    x_reshaped = x.view(shape[:-1] + (-1, 2))
    x0, x1 = x_reshaped.unbind(-1)
    
    # rotate: [-x1, x0]
    x_rotated = torch.stack((-x1, x0), dim=-1).view(shape)
    
    # apply cos/sin
    ndim = x.dim()
    if ndim == 4:
        cos = cos.view(1, cos.shape[0], 1, cos.shape[1])
        sin = sin.view(1, sin.shape[0], 1, sin.shape[1])
    elif ndim == 3:
        cos = cos.view(1, cos.shape[0], cos.shape[1])
        sin = sin.view(1, sin.shape[0], sin.shape[1])
        
    return x * cos + x_rotated * sin

def manual_mla_prefill_reference(
    q_content, kv_content, 
    q_rope_raw, k_rope_rot, 
    cos, sin,
    sm_scale
):
    # Q: [B, N, H, D] -> [B, H, N, D]
    qc = q_content.transpose(1, 2)
    # K: [B, N, D] -> [B, 1, N, D] -> broadcast to [B, H, N, D]
    B, N, H, _ = q_content.shape
    kc = kv_content.unsqueeze(1).expand(-1, H, -1, -1)
    
    score_content = torch.matmul(qc, kc.transpose(-2, -1))
    
    # RoPE: rotate Q
    q_rope_rot = apply_rotary_interleaved(q_rope_raw, cos, sin)
    qr = q_rope_rot.transpose(1, 2)
    
    # K is already rotated
    kr = k_rope_rot.unsqueeze(1).expand(-1, H, -1, -1)
    
    score_rope = torch.matmul(qr, kr.transpose(-2, -1))
    
    scores = (score_content + score_rope) * sm_scale
    
    # causal mask
    mask = torch.tril(torch.ones(N, N, device=scores.device))
    scores = scores.masked_fill(mask == 0, float('-inf'))
    
    probs = torch.softmax(scores.float(), dim=-1).to(q_content.dtype)

    out = torch.matmul(probs, kc)
    return out.transpose(1, 2)

def manual_mla_decode_reference(
    q_content,      # [B, H, D_L]
    kv_cache,       # [B, N, D_L]
    q_rope_raw,     # [B, H, D_R]
    k_rope_rot,     # [B, N, D_R] (pre-rotated)
    cos, sin,       # [Max_Seq, D_R]
    cur_pos,        # int
    sm_scale
):

    qc = q_content.unsqueeze(2)
    
    B, H, _ = q_content.shape
    N = kv_cache.shape[1]
    kc = kv_cache.unsqueeze(1).expand(-1, H, -1, -1)
    
    # [B, H, 1, D] @ [B, H, D, N] -> [B, H, 1, N]
    score_content = torch.matmul(qc, kc.transpose(-2, -1))
    
    cos_q = cos[cur_pos].unsqueeze(0) 
    sin_q = sin[cur_pos].unsqueeze(0)

    q_rope_4d = q_rope_raw.unsqueeze(1)
    q_rope_rot = apply_rotary_interleaved(q_rope_4d, cos_q, sin_q)
    qr = q_rope_rot.transpose(1, 2) # [B, H, 1, D_R]
    
    # K is already rotated: [B, N, D_R] -> [B, H, N, D_R]
    kr = k_rope_rot.unsqueeze(1).expand(-1, H, -1, -1)
    
    score_rope = torch.matmul(qr, kr.transpose(-2, -1))
    
    scores = (score_content + score_rope) * sm_scale
    probs = torch.softmax(scores.float(), dim=-1).to(q_content.dtype)

    out = torch.matmul(probs, kc)
    return out.squeeze(2)


def test_flash_mla_prefill_fused():
    print("\n=== Testing FlashMLA Prefill Fused ===")
    torch.manual_seed(42)
    
    B, N_CTX, H = 2, 128, 4
    D_LATENT = 512
    D_ROPE = 64
    D_HEAD_LOGIC = 128
    
    dtype = torch.float16
    device = "cuda"
    sm_scale = 1.0 / math.sqrt(D_HEAD_LOGIC + D_ROPE)

    # inputs
    q_content = torch.randn((B, N_CTX, H, D_LATENT), dtype=dtype, device=device)
    kv_content = torch.randn((B, N_CTX, D_LATENT), dtype=dtype, device=device)
    
    q_rope_raw = torch.randn((B, N_CTX, H, D_ROPE), dtype=dtype, device=device)
    k_rope_raw = torch.randn((B, N_CTX, D_ROPE), dtype=dtype, device=device)
    
    inv_freq = 1.0 / (10000.0 ** (torch.arange(0, D_ROPE, 2).float().to(device) / D_ROPE))
    t = torch.arange(N_CTX, device=device, dtype=inv_freq.dtype)
    freqs = torch.outer(t, inv_freq)
    freqs_full = torch.repeat_interleave(freqs, 2, dim=-1) 
    cos = freqs_full.cos().to(dtype)
    sin = freqs_full.sin().to(dtype)
    
    k_rope_rotated = apply_rotary_interleaved(k_rope_raw, cos, sin)

    tri_out = flash_mla_prefill_fused(
        q_content, kv_content,
        q_rope_raw, k_rope_rotated, cos, sin,
        sm_scale
    )

    ref_out = manual_mla_prefill_reference(
        q_content, kv_content,
        q_rope_raw, k_rope_rotated, cos, sin,
        sm_scale
    )

    diff = (tri_out - ref_out).abs().max().item()
    assert diff < 1e-2, f"Prefill diff too high: {diff}"
    print("Prefill Test Passed!")


def test_flash_mla_decode_fused():
    print("\n=== Testing FlashMLA Decode Fused ===")
    torch.manual_seed(42)
    
    B, N_CTX, H = 4, 1024, 8
    D_LATENT = 512
    D_ROPE = 64
    D_HEAD_LOGIC = 128
    
    dtype = torch.float16
    device = "cuda"
    sm_scale = 1.0 / math.sqrt(D_HEAD_LOGIC + D_ROPE)
    
    cur_pos = N_CTX - 1

    q_content = torch.randn((B, H, D_LATENT), dtype=dtype, device=device)
    kv_cache = torch.randn((B, N_CTX, D_LATENT), dtype=dtype, device=device)
    
    q_rope_raw = torch.randn((B, H, D_ROPE), dtype=dtype, device=device)
    k_rope_raw = torch.randn((B, N_CTX, D_ROPE), dtype=dtype, device=device)
    
    inv_freq = 1.0 / (10000.0 ** (torch.arange(0, D_ROPE, 2).float().to(device) / D_ROPE))
    t = torch.arange(N_CTX, device=device, dtype=inv_freq.dtype)
    freqs = torch.outer(t, inv_freq)
    freqs_full = torch.repeat_interleave(freqs, 2, dim=-1)
    cos = freqs_full.cos().to(dtype)
    sin = freqs_full.sin().to(dtype)
    
    k_rope_rotated = apply_rotary_interleaved(k_rope_raw, cos, sin)

    tri_out = flash_mla_decode_fused(
        q_content, kv_cache,
        q_rope_raw, k_rope_rotated, cos, sin,
        cur_pos,
        sm_scale
    )

    ref_out = manual_mla_decode_reference(
        q_content, kv_cache,
        q_rope_raw, k_rope_rotated,
        cos, sin,
        cur_pos,
        sm_scale
    )

    diff = (tri_out - ref_out).abs().max().item()
    assert diff < 1e-2, f"Decode diff too high: {diff}"
    print("Decode Test Passed!")

if __name__ == "__main__":
    if torch.cuda.is_available():
        test_flash_mla_prefill_fused()
        test_flash_mla_decode_fused()
    else:
        print("CUDA not available")