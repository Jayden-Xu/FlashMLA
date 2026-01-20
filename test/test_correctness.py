# test FlashMLA against PyTorch reference implementation

import torch
from ..flash_mla.ops.interface import flash_mla_prefill, flash_mla_decode


def manual_mla_reference(q_abs, kv_latent, sm_scale, is_causal=True):
    """
    Inputs:
      q_abs: [B, N, H, D] for Prefill OR [B, H, D] for Decode
      kv_latent: [B, N_CTX, D]
    """

    # Q: [B, H, N, D]
    if q_abs.dim() == 3: # decode [B, H, D] -> [B, H, 1, D]
        q = q_abs.unsqueeze(2) 
    else: # prefill [B, N, H, D] -> [B, H, N, D]
        q = q_abs.transpose(1, 2)
        
    B, H, N_Q, D = q.shape
    N_KV = kv_latent.shape[1]
    
    # K, V: [B, N_CTX, D] -> [B, 1, N_CTX, D] -> broadcast to [B, H, N_CTX, D]
    k = kv_latent.unsqueeze(1).expand(-1, H, -1, -1)
    v = k # MLA shares K and V
    
    # [B, H, N_Q, D] @ [B, H, D, N_KV] -> [B, H, N_Q, N_KV]
    attn_score = torch.matmul(q, k.transpose(-2, -1)) * sm_scale
    
    if is_causal and N_Q > 1:
        # causal mask: query i can attend to key j <= i
        mask = torch.tril(torch.ones(N_Q, N_KV, device=q.device))
        attn_score = attn_score.masked_fill(mask == 0, float('-inf'))
        
    # float32 for softmax stability
    attn_probs = torch.softmax(attn_score.float(), dim=-1).to(q.dtype)
    
    # [B, H, N_Q, N_KV] @ [B, H, N_KV, D] -> [B, H, N_Q, D]
    out = torch.matmul(attn_probs, v)
    
    if q_abs.dim() == 3: # decode output: [B, H, D]
        return out.squeeze(2)
    else: # prefill output: [B, N, H, D]
        return out.transpose(1, 2)


def test_flash_mla_prefill():
    print("\n=== Testing FlashMLA Prefill ===")
    torch.manual_seed(42)
    
    # config
    B, N_CTX, H, D_LATENT = 2, 256, 2, 128
    dtype = torch.float16
    device = "cuda"
    sm_scale = 1.0 / (D_LATENT ** 0.5)

    # inputs
    q_abs = torch.randn((B, N_CTX, H, D_LATENT), dtype=dtype, device=device)
    kv_latent = torch.randn((B, N_CTX, D_LATENT), dtype=dtype, device=device)

    tri_out = flash_mla_prefill(q_abs, kv_latent, sm_scale)
    ref_out = manual_mla_reference(q_abs, kv_latent, sm_scale, is_causal=True)

    assert torch.allclose(tri_out, ref_out, atol=1e-2, rtol=1e-2)
    print("Prefill Test Passed! Output Max Diff:", (tri_out - ref_out).abs().max().item())


def test_flash_mla_decode():
    print("\n=== Testing FlashMLA Decode ===")
    torch.manual_seed(42)

    # config
    B, N_CTX, H, D_LATENT = 4, 4096, 16, 512
    dtype = torch.float16
    device = "cuda"
    sm_scale = 1.0 / (D_LATENT ** 0.5)

    # inputs
    # Q_abs is [B, H, D] for decoding
    q_abs = torch.randn((B, H, D_LATENT), dtype=dtype, device=device)
    kv_cache = torch.randn((B, N_CTX, D_LATENT), dtype=dtype, device=device)

    tri_out = flash_mla_decode(q_abs, kv_cache, sm_scale)
    ref_out = manual_mla_reference(q_abs, kv_cache, sm_scale, is_causal=False)

    assert torch.allclose(tri_out, ref_out, atol=1e-2, rtol=1e-2)
    print("Decode Test Passed! Output Max Diff:", (tri_out - ref_out).abs().max().item())

if __name__ == "__main__":
    if torch.cuda.is_available():
        test_flash_mla_prefill()
        test_flash_mla_decode()
    else:
        print("CUDA not available.")
