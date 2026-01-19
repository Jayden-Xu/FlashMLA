# FlashAttention + MLA
import torch
import triton
import triton.language as tl


@triton.jit
def flash_mla_prefill_kernel(
    Q_ptr,         # [Batch, NUM_HEADS, N_CTX, HEAD_DIM]
    KV_latent_ptr, # [Batch, N_CTX, D_LATENT]
    W_UK_ptr,      # [Num_Heads, D_LATENT, HEAD_DIM]
    W_UV_ptr,      # [Num_Heads, D_LATENT, HEAD_DIM]
    Output_ptr,    # [Batch, NUM_HEADS, N_CTX, HEAD_DIM]

    # strides
    stride_q_b, stride_q_h, stride_q_m, stride_q_d,
    stride_kv_b, stride_kv_n, stride_kv_l,
    stride_w_h, stride_w_l, stride_w_d,
    stride_o_b, stride_o_h, stride_o_m, stride_o_d,

    # dimensions
    N_CTX: tl.constexpr,
    HEAD_DIM: tl.constexpr,
    D_LATENT: tl.constexpr,
    BLOCK_M: tl.constexpr,
    BLOCK_N: tl.constexpr,
    sm_scale
):
    
    pid = tl.program_id(0) # sequence len
    off_b = tl.program_id(1) # batch id
    off_h = tl.program_id(2) # head id

    # offsets for q
    offs_m = pid * BLOCK_M + tl.arange(0, BLOCK_M) # row offset (sequence len)
    offs_d = tl.arange(0, HEAD_DIM) # col offset (head dim)

    q_ptrs = Q_ptr + off_b * stride_q_b + off_h * stride_q_h + \
            offs_m[:, None] * stride_q_m + offs_d[None, :] * stride_q_d
    
    q = tl.load(q_ptrs, mask = offs_m[:, None] < N_CTX, other = 0.0)

    # accumulators
    m_i = tl.zeros([BLOCK_M], dtype = tl.float32) - float('inf')
    l_i = tl.zeros([BLOCK_M], dtype = tl.float32)
    acc = tl.zeros([BLOCK_M, HEAD_DIM], dtype = tl.float32)

    # MLA load projection weight

    offs_lat = tl.arange(0, D_LATENT) # latent offset
    # offsets for head id
    w_uk_base = W_UK_ptr + off_h * stride_w_h
    w_uv_base = W_UV_ptr + off_h * stride_w_h
    # offsets for latent len and head dim
    w_ptrs_offset = offs_lat[:, None] * stride_w_l + \
                    offs_d[None, :] * stride_w_d

    w_uk_ptrs = w_uk_base + w_ptrs_offset
    w_uv_ptrs = w_uv_base + w_ptrs_offset

    w_uk = tl.load(w_uk_ptrs)
    w_uv = tl.load(w_uv_ptrs)

    # loop over KV (flash v2)
    for start_n in range(0, N_CTX, BLOCK_N):

        # offsets for kv latent
        offs_n = start_n + tl.arange(0, BLOCK_N)
        kv_latent_ptrs = KV_latent_ptr + off_b* stride_kv_b + \
                        offs_n[:, None] * stride_kv_n + \
                        offs_lat[None, :] * stride_kv_l
    
        kv_latent = tl.load(
            kv_latent_ptrs, 
            mask = offs_n[:, None] < N_CTX,
            other = 0.0)
        
        # up projection
        k = tl.dot(kv_latent, w_uk, allow_tf32 = True)
        v = tl.dot(kv_latent, w_uv, allow_tf32 = True)
    
        # atten score
        score_ij = tl.dot(q, tl.trans(k), allow_tf32 = True)
        score_ij *= sm_scale

        mask = offs_n[None, :] < N_CTX
        score_ij = tl.where(mask, score_ij, float('-inf'))

        # online softmax
        m_ij = tl.max(score_ij, axis = 1) # reduction on cols
        m_new = tl.maximum(m_i, m_ij)
        alpha = tl.exp(m_i - m_new)
        beta = tl.exp(m_ij - m_new)

        p_ij_hat = tl.exp(score_ij - m_ij[:, None])
        l_block = tl.sum(p_ij_hat, axis = 1)
        l_new = alpha * l_i + beta * l_block

        acc = alpha[:, None] * acc + beta[:, None] * tl.dot(p_ij_hat, v, allow_tf32 = True)

        m_i = m_new
        l_i = l_new

    # write output
    acc = acc / l_i[:, None]

    output_ptrs = Output_ptr + off_b * stride_o_b + \
                    off_h * stride_o_h + \
                    offs_m[:, None] * stride_o_m + \
                    offs_d[None, :] * stride_o_d
    tl.store(output_ptrs, acc, mask = offs_m[:, None] < N_CTX)


def cdiv(x, y):
    return (x + y - 1) // y


def mla_attention(q, kv_latent, w_uk, w_uv, sm_scale):

    BATCH, N_CTX, D_LATENT = kv_latent.shape
    NUM_HEADS, _, HEAD_DIM = w_uk.shape
    
    # Output buffer
    output = torch.empty((BATCH, NUM_HEADS, N_CTX, HEAD_DIM), device=q.device, dtype=q.dtype)
    
    # Grid definition
    BLOCK_M = 64
    BLOCK_N = 32
    grid = (cdiv(N_CTX, BLOCK_M), BATCH, NUM_HEADS)
    
    flash_mla_prefill_kernel[grid](
        q, kv_latent, w_uk, w_uv, output,
        q.stride(0), q.stride(1), q.stride(2), q.stride(3),
        kv_latent.stride(0), kv_latent.stride(1), kv_latent.stride(2),
        w_uk.stride(0), w_uk.stride(1), w_uk.stride(2),
        output.stride(0), output.stride(1), output.stride(2), output.stride(3),
        N_CTX=N_CTX,
        HEAD_DIM=HEAD_DIM,
        D_LATENT=D_LATENT,
        BLOCK_M=BLOCK_M,
        BLOCK_N=BLOCK_N,
        sm_scale=sm_scale
    )
    return output
