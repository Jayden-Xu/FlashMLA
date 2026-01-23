
import torch
import math
from flash_mla.ops.interface_fused import flash_mla_prefill_fused


def apply_rotary_interleaved(x: torch.Tensor, cos: torch.Tensor, sin: torch.Tensor):

    # reshape to pairs: [..., D//2, 2]
    shape = x.shape
    x_reshaped = x.view(shape[:-1] + (-1, 2))
    x0, x1 = x_reshaped.unbind(-1)
    
    # rotate: [-x1, x0]
    x_rotated = torch.stack((-x1, x0), dim=-1).view(shape)
    
    # apply cos/sin
    ndim = x.dim()
    if ndim == 4: # Q: [B, N, H, D]
        cos = cos.view(1, cos.shape[0], 1, cos.shape[1])
        sin = sin.view(1, sin.shape[0], 1, sin.shape[1])
    elif ndim == 3: # K: [B, N, D]
        cos = cos.view(1, cos.shape[0], cos.shape[1])
        sin = sin.view(1, sin.shape[0], sin.shape[1])
        
    return x * cos + x_rotated * sin

def manual_mla_reference(
    q_content, kv_content, 
    q_rope_raw, k_rope_rot, # K is already rotated
    cos, sin,
    sm_scale
):
    # content score
    # Q: [B, N, H, D] -> [B, H, N, D]
    qc = q_content.transpose(1, 2)
    # K: [B, N, D] -> [B, 1, N, D] -> broadcast to [B, H, N, D]
    B, N, H, _ = q_content.shape
    kc = kv_content.unsqueeze(1).expand(-1, H, -1, -1)
    
    score_content = torch.matmul(qc, kc.transpose(-2, -1))
    
    # rope score
    # rotate Q using interleaved logic
    q_rope_rot = apply_rotary_interleaved(q_rope_raw, cos, sin)
    qr = q_rope_rot.transpose(1, 2)
    
    # K is already rotated, just expand
    kr = k_rope_rot.unsqueeze(1).expand(-1, H, -1, -1)
    
    score_rope = torch.matmul(qr, kr.transpose(-2, -1))
    
    scores = (score_content + score_rope) * sm_scale
    
    # causal mask
    mask = torch.tril(torch.ones(N, N, device=scores.device))
    scores = scores.masked_fill(mask == 0, float('-inf'))
    
    probs = torch.softmax(scores.float(), dim=-1).to(q_content.dtype)

    # V same as K in MLA
    out = torch.matmul(probs, kc)
    
    return out.transpose(1, 2)


def test_flash_mla_prefill_fused():
    print("\n=== Testing FlashMLA Prefill Fused ===")
    torch.manual_seed(42)
    
    # config
    B, N_CTX, H = 2, 128, 4
    D_LATENT = 512   # content dim
    D_ROPE = 64      # RoPE dim
    D_HEAD_LOGIC = 128 # logical Head Dim for scaling
    
    dtype = torch.float16
    device = "cuda"
    
    sm_scale = 1.0 / math.sqrt(D_HEAD_LOGIC + D_ROPE)

    # absorbed Q and compressed KV
    q_content = torch.randn((B, N_CTX, H, D_LATENT), dtype=dtype, device=device)
    kv_content = torch.randn((B, N_CTX, D_LATENT), dtype=dtype, device=device)
    
    # RoPE (raw Q and raw K)
    q_rope_raw = torch.randn((B, N_CTX, H, D_ROPE), dtype=dtype, device=device)
    k_rope_raw = torch.randn((B, N_CTX, D_ROPE), dtype=dtype, device=device)
    
    # cos/sin Tables (interleaved)
    inv_freq = 1.0 / (10000.0 ** (torch.arange(0, D_ROPE, 2).float().to(device) / D_ROPE))
    t = torch.arange(N_CTX, device=device, dtype=inv_freq.dtype)
    freqs = torch.outer(t, inv_freq) # [N, D/2]
    # interleave frequencies [f0, f0, f1, f1...] to match interleaved rotation
    freqs_full = torch.repeat_interleave(freqs, 2, dim=-1) 
    cos = freqs_full.cos().to(dtype)
    sin = freqs_full.sin().to(dtype)
    
    # pre-rotate K_rope
    k_rope_rotated = apply_rotary_interleaved(k_rope_raw, cos, sin)

    tri_out = flash_mla_prefill_fused(
        q_content, kv_content,
        q_rope_raw, k_rope_rotated, cos, sin,
        sm_scale
    )

    ref_out = manual_mla_reference(
        q_content, kv_content,
        q_rope_raw, k_rope_rotated, cos, sin,
        sm_scale
    )

    diff = (tri_out - ref_out).abs().max().item()
    
    assert diff < 1e-2
    print("Fused Prefill Test Passed! Output Max Diff:", diff)

if __name__ == "__main__":
    if torch.cuda.is_available():
        test_flash_mla_prefill_fused()
    else:
        print("CUDA not available")