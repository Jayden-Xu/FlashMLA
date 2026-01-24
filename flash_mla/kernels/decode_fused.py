# FlashDecoding + MLA + Decoupled RoPE

import triton
import triton.language as tl


autotune_configs = [
    triton.Config({'BLOCK_N': 64, 'num_warps': 4}, num_warps=4),
    triton.Config({'BLOCK_N': 32, 'num_warps': 4}, num_warps=4),
    triton.Config({'BLOCK_N': 16, 'num_warps': 4}, num_warps=4),
]


@triton.autotune(
    configs=autotune_configs,
    key=['N_CTX'],
)
@triton.jit
def flash_mla_decode_stage_1_kernel(
    Q_abs_ptr,     # [Batch, Heads, D_LATENT] absorbed Query
    KV_cache_ptr,  # [Batch, N_CTX, D_LATENT] latent Cache

    # RoPE
    Q_rope_ptr,    # [Batch, Heads, D_ROPE] raw rope Query
    K_rope_ptr,    # [Batch, N_CTX, D_ROPE] shared pre-rotated rope K
    Cos_ptr,       # [MAX_SEQ, D_ROPE] pre-computed cos table
    Sin_ptr,       # [MAX_SEQ, D_ROPE] pre-computed sin table

    # mid outputs
    Mid_O_ptr,     # [Batch, Heads, Num_Splits, D_LATENT]
    Mid_LSE_ptr,   # [Batch, Heads, Num_Splits]
    
    # strides
    stride_q_b, stride_q_h, stride_q_l,
    stride_kv_b, stride_kv_n, stride_kv_l,

    # RoPE strides
    stride_qr_b, stride_qr_h, stride_qr_d,
    stride_kr_b, stride_kr_n, stride_kr_d,
    stride_cos_n, stride_cos_d,

    stride_mid_o_b, stride_mid_o_h, stride_mid_o_s, stride_mid_o_l,
    stride_mid_lse_b, stride_mid_lse_h, stride_mid_lse_s,
    
    # dimensions
    CUR_POS, # current position of the token (usually N_CTX - 1)
    N_CTX, # total kv length
    SPLIT_N_SIZE,
    D_LATENT: tl.constexpr,
    D_ROPE: tl.constexpr,
    BLOCK_N: tl.constexpr,
    sm_scale: tl.constexpr
):
    
    pid_b = tl.program_id(0) # batch ID
    pid_h = tl.program_id(1) # head ID
    pid_s = tl.program_id(2) # split ID

    pid_b = pid_b.to(tl.int64)
    pid_h = pid_h.to(tl.int64)
    pid_s = pid_s.to(tl.int64)

    # load absorbed q_content
    offs_l = tl.arange(0, D_LATENT)
    q_ptr = Q_abs_ptr + pid_b * stride_q_b + pid_h * stride_q_h + offs_l * stride_q_l
    q_abs = tl.load(q_ptr)

    # load q_rope
    offs_d_rope = tl.arange(0, D_ROPE)
    qr_ptr = Q_rope_ptr + pid_b * stride_qr_b + pid_h * stride_qr_h + \
             offs_d_rope * stride_qr_d
    q_rope = tl.load(qr_ptr) # [D_ROPE]

    # load cos/sin tables
    cos_ptr = Cos_ptr + CUR_POS * stride_cos_n + offs_d_rope * stride_cos_d
    sin_ptr = Sin_ptr + CUR_POS * stride_cos_n + offs_d_rope * stride_cos_d
    cos = tl.load(cos_ptr)
    sin = tl.load(sin_ptr)

    # interleaved rotation
    is_even = (offs_d_rope % 2) == 0
    offs_d_rope_swap = offs_d_rope + tl.where(is_even, 1, -1)
    qr_ptr_swap = Q_rope_ptr + pid_b * stride_qr_b + pid_h * stride_qr_h + \
                  offs_d_rope_swap * stride_qr_d
    q_rope_swap = tl.load(qr_ptr_swap) # [D_ROPE]
    rope_sign = tl.where(is_even, -1.0, 1.0).to(q_rope.dtype)

    q_rope_rot = q_rope * cos + (q_rope_swap * sin) * rope_sign
    
    # KV cache range
    start_n_global = pid_s * SPLIT_N_SIZE
    end_n_global = min(start_n_global + SPLIT_N_SIZE, N_CTX)

    if start_n_global >= N_CTX:
        lse_ptr = Mid_LSE_ptr + pid_b * stride_mid_lse_b + \
                  pid_h * stride_mid_lse_h + pid_s * stride_mid_lse_s
        tl.store(lse_ptr, float('-inf'))
        return

    m_i = float('-inf')
    l_i = 0.0
    acc = tl.zeros([D_LATENT], dtype=tl.float32)

    # loop over KV
    for start_n in range(start_n_global, end_n_global, BLOCK_N):
        offs_n = start_n + tl.arange(0, BLOCK_N)
        offs_n = offs_n.to(tl.int64)
        mask_n = offs_n < N_CTX

        # load KV latent
        kv_ptr = KV_cache_ptr + pid_b * stride_kv_b + \
                 offs_n[:, None] * stride_kv_n + \
                 offs_l[None, :] * stride_kv_l
        
        k_content = tl.load(kv_ptr, mask=mask_n[:, None], other=0.0)
        score_content = tl.sum(q_abs[None, :] * k_content, 1)

        # RoPE
        kr_ptr = K_rope_ptr + pid_b * stride_kr_b + \
                 offs_n[:, None] * stride_kr_n + \
                 offs_d_rope[None, :] * stride_kr_d
        k_rope_rot = tl.load(kr_ptr, mask=mask_n[:, None], other=0.0) # pre-rotated K
        score_rope = tl.sum(q_rope_rot[None, :] * k_rope_rot, 1)

        score = (score_content + score_rope) * sm_scale
        score = tl.where(mask_n, score, float('-inf'))

        # online softmax
        m_block = tl.max(score, 0)
        m_new = tl.maximum(m_i, m_block)
        
        alpha = tl.exp(m_i - m_new)
        p_block = tl.exp(score - m_new) # [BLOCK_N]

        # KV as K and V
        weighted_v = tl.sum(k_content * p_block[:, None], 0)

        acc = alpha * acc + weighted_v
        l_i = alpha * l_i + tl.sum(p_block, 0)
        m_i = m_new

    if l_i > 0:
        mid_o = acc / l_i
        mid_lse = m_i + tl.log(l_i)
    else:
        mid_o = tl.zeros([D_LATENT], dtype=tl.float32)
        mid_lse = float('-inf')

    off_mid_o = pid_b * stride_mid_o_b + pid_h * stride_mid_o_h + \
                pid_s * stride_mid_o_s + offs_l * stride_mid_o_l
    tl.store(Mid_O_ptr + off_mid_o, mid_o)
    
    off_mid_lse = pid_b * stride_mid_lse_b + pid_h * stride_mid_lse_h + \
                  pid_s * stride_mid_lse_s
    tl.store(Mid_LSE_ptr + off_mid_lse, mid_lse)


@triton.jit
def flash_mla_decode_stage_2_kernel(
    Mid_O_ptr, Mid_LSE_ptr, Output_ptr,
    stride_mid_o_b, stride_mid_o_h, stride_mid_o_s, stride_mid_o_l,
    stride_mid_lse_b, stride_mid_lse_h, stride_mid_lse_s,
    stride_out_b, stride_out_h, stride_out_l,
    D_LATENT: tl.constexpr, 
    NUM_SPLITS: tl.constexpr
):
    pid_b = tl.program_id(0)
    pid_h = tl.program_id(1)

    pid_b = pid_b.to(tl.int64)
    pid_h = pid_h.to(tl.int64)

    offs_s = tl.arange(0, NUM_SPLITS)
    offs_l = tl.arange(0, D_LATENT)

    lse_ptr = Mid_LSE_ptr + pid_b * stride_mid_lse_b + \
              pid_h * stride_mid_lse_h + offs_s * stride_mid_lse_s
    lse_all = tl.load(lse_ptr, mask=offs_s < NUM_SPLITS, other=float('-inf'))

    # global softmax weights
    lse_max = tl.max(lse_all, 0)
    weights = tl.exp(lse_all - lse_max)
    sum_weights = tl.sum(weights, 0)

    mid_o_ptr_base = Mid_O_ptr + pid_b * stride_mid_o_b + pid_h * stride_mid_o_h
    mid_o_ptrs = mid_o_ptr_base + \
                 offs_s[:, None] * stride_mid_o_s + \
                 offs_l[None, :] * stride_mid_o_l
    
    mid_o_all = tl.load(mid_o_ptrs, mask=offs_s[:, None] < NUM_SPLITS, other=0.0)

    weighted_o = mid_o_all * weights[:, None]
    final_acc = tl.sum(weighted_o, 0)

    output = final_acc / sum_weights

    # store final latent output
    out_ptr = Output_ptr + pid_b * stride_out_b + \
              pid_h * stride_out_h + offs_l * stride_out_l
    tl.store(out_ptr, output)
