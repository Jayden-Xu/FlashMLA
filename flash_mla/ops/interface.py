
import torch
from flash_mla.kernels.prefill import flash_mla_prefill_kernel
from flash_mla.kernels.decode import flash_mla_decode_stage_1_kernel, flash_mla_decode_stage_2_kernel


def cdiv(x, y): 
    return (x + y - 1) // y


def flash_mla_prefill(q_abs, kv_latent, sm_scale):
    """
    Args:
        q_abs: [Batch, N_CTX, Num_Heads, D_LATENT] (Absorbed Q)
        kv_latent: [Batch, N_CTX, D_LATENT] (Shared Latent KV)
    """

    B, N_CTX, H, D_LATENT = q_abs.shape
    
    output = torch.empty_like(q_abs)
    
    BLOCK_M = 128
    BLOCK_N = 64
    
    grid = (cdiv(N_CTX, BLOCK_M), B, H)
    
    flash_mla_prefill_kernel[grid](
        q_abs, kv_latent, output,
        *q_abs.stride(),
        *kv_latent.stride(), # [stride_b, stride_n, stride_d]
        *output.stride(),
        N_CTX=N_CTX, D_LATENT=D_LATENT,
        BLOCK_M=BLOCK_M, BLOCK_N=BLOCK_N,
        sm_scale=sm_scale,
        num_warps=4, num_stages=2
    )
    
    return output


def flash_mla_decode(q_abs, kv_cache, sm_scale):
    """
    Args:
        q_abs: [Batch, Heads, D_LATENT] # Absorbed Query
        kv_cache: [Batch, N_CTX, D_LATENT] # Latent Cache
    Returns:
        output: [Batch, Heads, D_LATENT]
    """

    B, H, D_LATENT = q_abs.shape
    _, N_CTX, _ = kv_cache.shape
    
    BLOCK_N = 64
    SPLIT_N_SIZE = 2048
    NUM_SPLITS = cdiv(N_CTX, SPLIT_N_SIZE)
    # maximum NUM_SPLITS
    NUM_SPLITS = min(NUM_SPLITS, 128) 
    SPLIT_N_SIZE = cdiv(N_CTX, NUM_SPLITS)

    # allocation
    mid_o = torch.empty((B, H, NUM_SPLITS, D_LATENT), device=q_abs.device, dtype=torch.float32)
    mid_lse = torch.empty((B, H, NUM_SPLITS), device=q_abs.device, dtype=torch.float32)
    output = torch.empty_like(q_abs)

    # stage 1
    grid_1 = (B, H, NUM_SPLITS)
    flash_mla_decode_stage_1_kernel[grid_1](
        q_abs, kv_cache, mid_o, mid_lse,
        *q_abs.stride(),
        *kv_cache.stride(),
        *mid_o.stride(),
        *mid_lse.stride(),
        N_CTX=N_CTX, D_LATENT=D_LATENT, 
        BLOCK_N=BLOCK_N, SPLIT_N_SIZE=SPLIT_N_SIZE, sm_scale=sm_scale
    )

    # stage 2
    grid_2 = (B, H)
    flash_mla_decode_stage_2_kernel[grid_2](
        mid_o, mid_lse, output,
        *mid_o.stride(),
        *mid_lse.stride(),
        *output.stride(),
        D_LATENT=D_LATENT, NUM_SPLITS=NUM_SPLITS
    )
    
    return output