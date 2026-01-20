# FlashAttention + MLA
import torch
import triton
import triton.language as tl


@triton.jit
def flash_mla_prefill_kernel(
    Q_ptr,         # [Batch, N_CTX, Num_Heads, D_LATENT]
    KV_ptr,        # [Batch, N_CTX, D_LATENT]
    Output_ptr,    # [Batch, N_CTX, Num_Heads, D_LATENT]

    # strides
    stride_q_b, stride_q_n, stride_q_h, stride_q_d,
    stride_kv_b, stride_kv_n, stride_kv_d, # no head stride for KV
    stride_o_b, stride_o_n, stride_o_h, stride_o_d,

    # dimensions
    N_CTX: tl.constexpr,
    D_LATENT: tl.constexpr,
    BLOCK_M: tl.constexpr,
    BLOCK_N: tl.constexpr,
    sm_scale: tl.constexpr
):

    pid_m = tl.program_id(0) # block ID
    pid_b = tl.program_id(1) # batch ID
    pid_h = tl.program_id(2) # head ID

    # load Q block
    offs_m = pid_m * BLOCK_M + tl.arange(0, BLOCK_M)
    offs_d = tl.arange(0, D_LATENT)

    q_ptr_base = Q_ptr + pid_b * stride_q_b + pid_h * stride_q_h
    q_ptrs = q_ptr_base + offs_m[:, None] * stride_q_n + offs_d[None, :] * stride_q_d
    
    q = tl.load(q_ptrs, mask=offs_m[:, None] < N_CTX, other=0.0)

    m_i = tl.zeros([BLOCK_M], dtype=tl.float32) - float('inf')
    l_i = tl.zeros([BLOCK_M], dtype=tl.float32)
    acc = tl.zeros([BLOCK_M, D_LATENT], dtype=tl.float32)

    # loop over KV blocks
    # causal Attention
    end_n = (pid_m + 1) * BLOCK_M 
    end_n = min(end_n, N_CTX)

    kv_ptr_base = KV_ptr + pid_b * stride_kv_b
    
    for start_n in range(0, end_n, BLOCK_N):
        offs_n = start_n + tl.arange(0, BLOCK_N)
        
        # load KV block (shared across heads, no head stride)
        kv_ptrs = kv_ptr_base + \
                  offs_n[:, None] * stride_kv_n + \
                  offs_d[None, :] * stride_kv_d
        
        kv_block = tl.load(kv_ptrs, mask=offs_n[:, None] < N_CTX, other=0.0)

        # Q [M, D] @ K.T [D, N] -> [M, N]
        score_ij = tl.dot(q, tl.trans(kv_block)) * sm_scale

        mask_padding = offs_n[None, :] < N_CTX
        # causal mask
        mask_causal = offs_n[None, :] <= offs_m[:, None]
        
        mask = mask_padding & mask_causal
        score_ij = tl.where(mask, score_ij, float('-inf'))

        # online softmax
        m_ij = tl.max(score_ij, 1)
        m_new = tl.maximum(m_i, m_ij)
        
        alpha = tl.exp(m_i - m_new)
        # beta = tl.exp(m_ij - m_new)
        
        p_ij = tl.exp(score_ij - m_new[:, None])
        
        # precision
        acc = alpha[:, None] * acc + tl.dot(p_ij.to(tl.float16), kv_block)
        
        l_block = tl.sum(p_ij, 1)
        l_i = alpha * l_i + l_block
        m_i = m_new

    acc = acc / l_i[:, None]

    # output in latent dim
    o_ptr_base = Output_ptr + pid_b * stride_o_b + pid_h * stride_o_h
    o_ptrs = o_ptr_base + offs_m[:, None] * stride_o_n + offs_d[None, :] * stride_o_d
    
    tl.store(o_ptrs, acc.to(Output_ptr.dtype.element_ty), mask=offs_m[:, None] < N_CTX)


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