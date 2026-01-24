# FlashDecoding + MLA

import triton
import triton.language as tl


autotune_configs = [
    triton.Config({'BLOCK_N': 64, 'num_warps': 4}, num_warps=4),
    triton.Config({'BLOCK_N': 32, 'num_warps': 4}, num_warps=4),
    triton.Config({'BLOCK_N': 16, 'num_warps': 4}, num_warps=4),
]


@triton.autotune(
    configs=autotune_configs,
    key=['D_LATENT'],
)
@triton.jit
def flash_mla_decode_stage_1_kernel(
    Q_abs_ptr,     # [Batch, Heads, D_LATENT] absorbed Query
    KV_cache_ptr,  # [Batch, N_CTX, D_LATENT] latent Cache
    
    # mid outputs
    Mid_O_ptr,     # [Batch, Heads, Num_Splits, D_LATENT]
    Mid_LSE_ptr,   # [Batch, Heads, Num_Splits]
    
    # strides
    stride_q_b, stride_q_h, stride_q_l,
    stride_kv_b, stride_kv_n, stride_kv_l,
    stride_mid_o_b, stride_mid_o_h, stride_mid_o_s, stride_mid_o_l,
    stride_mid_lse_b, stride_mid_lse_h, stride_mid_lse_s,
    
    # dimensions
    N_CTX,
    SPLIT_N_SIZE,
    D_LATENT: tl.constexpr,
    BLOCK_N: tl.constexpr,
    sm_scale: tl.constexpr
):
    
    pid_b = tl.program_id(0) # batch ID
    pid_h = tl.program_id(1) # head ID
    pid_s = tl.program_id(2) # split ID

    pid_b = pid_b.to(tl.int64)
    pid_h = pid_h.to(tl.int64)
    pid_s = pid_s.to(tl.int64)

    # load absorbed query
    offs_l = tl.arange(0, D_LATENT)
    q_ptr = Q_abs_ptr + pid_b * stride_q_b + pid_h * stride_q_h + offs_l * stride_q_l
    q_abs = tl.load(q_ptr)

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
        
        kv_block = tl.load(kv_ptr, mask=mask_n[:, None], other=0.0)

        # Q_abs [D] @ KV.T [D, Block] -> [Block]
        score = tl.sum(q_abs[None, :] * kv_block, 1) * sm_scale
        score = tl.where(mask_n, score, float('-inf'))

        # online softmax
        m_block = tl.max(score, 0)
        m_new = tl.maximum(m_i, m_block)
        
        alpha = tl.exp(m_i - m_new)
        p_block = tl.exp(score - m_new) # [BLOCK_N]

        # KV as K and V
        # [D, Block] @ [Block, 1] -> [D, 1]
        weighted_v = tl.sum(kv_block * p_block[:, None], 0)

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
