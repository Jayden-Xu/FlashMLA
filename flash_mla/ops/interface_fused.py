
import torch
from flash_mla.kernels.prefill_fused import flash_mla_prefill_kernel


def cdiv(x, y): 
    return (x + y - 1) // y


def flash_mla_prefill_fused(
    q_abs,      # [Batch, N_CTX, Num_Heads, D_LATENT]
    kv_latent,  # [Batch, N_CTX, D_LATENT]
    q_rope,     # [Batch, N_CTX, Num_Heads, D_ROPE] unrotated
    k_rope,     # [Batch, N_CTX, D_ROPE] assume alreadyrotated
    cos, sin,   # [N_CTX, D_ROPE]
    sm_scale
):

    B, N_CTX, H, D_LATENT = q_abs.shape
    _, _, _, D_ROPE = q_rope.shape
    
    output = torch.empty_like(q_abs)

    grid = lambda META: (cdiv(N_CTX, META['BLOCK_M']), B, H)
    
    flash_mla_prefill_kernel[grid](
        q_abs, kv_latent,
        q_rope, k_rope, cos, sin,
        output,
        *q_abs.stride(),
        *kv_latent.stride(),
        *q_rope.stride(),
        *k_rope.stride(),
        *cos.stride(),
        *output.stride(),
        N_CTX=N_CTX,
        D_LATENT=D_LATENT,
        D_ROPE=D_ROPE,
        sm_scale=sm_scale
    )
    
    return output