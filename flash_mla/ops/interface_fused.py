
import torch
from flash_mla.kernels.prefill_fused import flash_mla_prefill_kernel
from flash_mla.kernels.decode_fused import flash_mla_decode_stage_1_kernel, flash_mla_decode_stage_2_kernel


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
        N_CTX,
        D_LATENT=D_LATENT,
        D_ROPE=D_ROPE,
        sm_scale=sm_scale
    )
    
    return output


def flash_mla_decode_fused(
    q_abs,      # [Batch, Heads, D_LATENT]
    kv_cache,   # [Batch, N_CTX, D_LATENT]
    q_rope,     # [Batch, Heads, D_ROPE] # raw rope Q
    k_rope,     # [Batch, N_CTX, D_ROPE] # pre-rotated rope K
    cos, sin,   # [Max_Seq, D_ROPE]
    cur_pos,    # current position of the token
    sm_scale
):

    B, H, D_LATENT = q_abs.shape
    _, N_CTX, _ = kv_cache.shape
    _, _, D_ROPE = q_rope.shape
    
    SPLIT_N_SIZE = 2048
    NUM_SPLITS = cdiv(N_CTX, SPLIT_N_SIZE)
    NUM_SPLITS = min(NUM_SPLITS, 128) 
    SPLIT_N_SIZE = cdiv(N_CTX, NUM_SPLITS)

    mid_o = torch.empty((B, H, NUM_SPLITS, D_LATENT), device=q_abs.device, dtype=torch.float32)
    mid_lse = torch.empty((B, H, NUM_SPLITS), device=q_abs.device, dtype=torch.float32)
    
    output = torch.empty_like(q_abs)

    grid_1 = (B, H, NUM_SPLITS)
    
    flash_mla_decode_stage_1_kernel[grid_1](
        q_abs, kv_cache,
        q_rope, k_rope, cos, sin,
        mid_o, mid_lse,
        *q_abs.stride(),
        *kv_cache.stride(),
        *q_rope.stride(),
        *k_rope.stride(),
        *cos.stride(),
        *mid_o.stride(),
        *mid_lse.stride(),
        cur_pos,
        N_CTX,
        SPLIT_N_SIZE,
        D_LATENT=D_LATENT,
        D_ROPE=D_ROPE,
        sm_scale=sm_scale
    )

    grid_2 = (B, H)
    flash_mla_decode_stage_2_kernel[grid_2](
        mid_o, mid_lse, output,
        *mid_o.stride(),
        *mid_lse.stride(),
        *output.stride(),
        D_LATENT=D_LATENT, 
        NUM_SPLITS=NUM_SPLITS
    )
    
    return output