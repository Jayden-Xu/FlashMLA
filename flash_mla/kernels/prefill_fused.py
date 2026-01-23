# FlashAttention + MLA + Decoupled RoPE

import triton
import triton.language as tl


autotune_configs = [
    triton.Config({'BLOCK_M': 64, 'BLOCK_N': 32, 'num_stages': 2, 'num_warps': 4}, num_stages=2, num_warps=4),
    triton.Config({'BLOCK_M': 32, 'BLOCK_N': 32, 'num_stages': 2, 'num_warps': 4}, num_stages=2, num_warps=4),
]


@triton.autotune(
    configs=autotune_configs,
    key=['N_CTX'],
)
@triton.jit
def flash_mla_prefill_kernel(
    Q_ptr,         # [Batch, N_CTX, Num_Heads, D_LATENT]
    KV_ptr,        # [Batch, N_CTX, D_LATENT]

    # RoPE
    Q_rope_ptr,    # [Batch, N_CTX, Num_Heads, D_ROPE]
    K_rope_ptr,    # [Batch, N_CTX, D_ROPE]
    Cos_ptr,       # [N_CTX, D_ROPE] pre-computed cos table
    Sin_ptr,       # [N_CTX, D_ROPE] pre-computed sin table

    Output_ptr,    # [Batch, N_CTX, Num_Heads, D_LATENT]

    # strides for content
    stride_q_b, stride_q_n, stride_q_h, stride_q_d,
    stride_kv_b, stride_kv_n, stride_kv_d, # no head stride for KV

    # strides for rope
    stride_qr_b, stride_qr_n, stride_qr_h, stride_qr_d,
    stride_kr_b, stride_kr_n, stride_kr_d,
    stride_cos_n, stride_cos_d,

    # strides for output
    stride_o_b, stride_o_n, stride_o_h, stride_o_d,

    # dimensions
    N_CTX: tl.constexpr,
    D_LATENT: tl.constexpr, # latent dim
    D_ROPE: tl.constexpr,   # rope dim
    BLOCK_M: tl.constexpr,
    BLOCK_N: tl.constexpr,
    sm_scale: tl.constexpr  # should now be 1/sqrt(D_H + D_ROPE)
):

    pid_m = tl.program_id(0) # block ID
    pid_b = tl.program_id(1) # batch ID
    pid_h = tl.program_id(2) # head ID

    pid_m = pid_m.to(tl.int64)
    pid_b = pid_b.to(tl.int64)
    pid_h = pid_h.to(tl.int64)

    # load Q content block
    offs_m = pid_m * BLOCK_M + tl.arange(0, BLOCK_M)
    offs_d = tl.arange(0, D_LATENT)

    q_ptr_base = Q_ptr + pid_b * stride_q_b + pid_h * stride_q_h
    q_ptrs = q_ptr_base + offs_m[:, None] * stride_q_n + offs_d[None, :] * stride_q_d
    
    q_content = tl.load(q_ptrs, mask=offs_m[:, None] < N_CTX, other=0.0)

    # RoPE
    offs_d_rope = tl.arange(0, D_ROPE)
    qr_ptr_base = Q_rope_ptr + pid_b * stride_qr_b + pid_h * stride_qr_h
    qr_ptrs = qr_ptr_base + offs_m[:, None] * stride_qr_n + offs_d_rope[None, :] * stride_qr_d
    q_rope = tl.load(qr_ptrs, mask = offs_m[:, None] < N_CTX, other = 0.0)

    # load cos and sin tables
    cos_ptrs = Cos_ptr + offs_m[:, None] * stride_cos_n + offs_d_rope[None, :] * stride_cos_d
    sin_ptrs = Sin_ptr + offs_m[:, None] * stride_cos_n + offs_d_rope[None, :] * stride_cos_d

    cos = tl.load(cos_ptrs, mask = offs_m[:, None] < N_CTX, other = 1.0)
    sin = tl.load(sin_ptrs, mask = offs_m[:, None] < N_CTX, other = 0.0)

    is_even = (offs_d_rope % 2) == 0
    # [x0, x1, x2, x3, ...] -> [x1, x0, x3, x2, ...]
    offs_d_rope_swap = offs_d_rope + tl.where(is_even, 1, -1)
    qr_ptrs_swap = qr_ptr_base + offs_m[:, None] * stride_qr_n + offs_d_rope_swap[None, :] * stride_qr_d
    q_rope_swap = tl.load(qr_ptrs_swap, mask = offs_m[:, None] < N_CTX, other = 0.0)
    rope_sign = tl.where(is_even, -1.0, 1.0).to(q_rope.dtype)

    q_rope_rot = q_rope * cos + (q_rope_swap * rope_sign) * sin

    m_i = tl.zeros([BLOCK_M], dtype=tl.float32) - float('inf')
    l_i = tl.zeros([BLOCK_M], dtype=tl.float32)
    acc = tl.zeros([BLOCK_M, D_LATENT], dtype=tl.float32)

    # loop over KV blocks
    # causal Attention
    end_n = (pid_m + 1) * BLOCK_M 
    end_n = min(end_n, N_CTX)

    kv_ptr_base = KV_ptr + pid_b * stride_kv_b
    kr_ptr_base = K_rope_ptr + pid_b * stride_kr_b
    
    for start_n in range(0, end_n, BLOCK_N):
        offs_n = start_n + tl.arange(0, BLOCK_N)
        
        # load KV block (shared across heads, no head stride)
        kv_ptrs = kv_ptr_base + \
                  offs_n[:, None] * stride_kv_n + \
                  offs_d[None, :] * stride_kv_d
        
        k_content = tl.load(kv_ptrs, mask=offs_n[:, None] < N_CTX, other=0.0)

        # RoPE, suppose key is already rotated
        kr_ptrs = kr_ptr_base + \
                  offs_n[:, None] * stride_kr_n + \
                  offs_d_rope[None, :] * stride_kr_d
        k_rope_rot = tl.load(kr_ptrs, mask = offs_n[:, None] < N_CTX, other = 0.0)

        # content & rope score
        score_content = tl.dot(q_content, tl.trans(k_content))
        score_rope = tl.dot(q_rope_rot, tl.trans(k_rope_rot))

        score_ij = (score_content + score_rope) * sm_scale

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
        acc = alpha[:, None] * acc + tl.dot(p_ij.to(tl.float16), k_content)
        
        l_block = tl.sum(p_ij, 1)
        l_i = alpha * l_i + l_block
        m_i = m_new

    acc = acc / l_i[:, None]

    # output in latent dim
    o_ptr_base = Output_ptr + pid_b * stride_o_b + pid_h * stride_o_h
    o_ptrs = o_ptr_base + offs_m[:, None] * stride_o_n + offs_d[None, :] * stride_o_d
    
    tl.store(o_ptrs, acc, mask=offs_m[:, None] < N_CTX)
