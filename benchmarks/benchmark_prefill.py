import torch
import torch.nn.functional as F
import math
from flash_mla.ops.interface import flash_mla_prefill


def pytorch_sdpa_native(q, k, v, sm_scale):
    # PyTorch Native SDPA
    q = q.transpose(1, 2)
    k = k.transpose(1, 2)
    v = v.transpose(1, 2)
    
    out = F.scaled_dot_product_attention(q, k, v, scale=sm_scale, is_causal=True)
    
    return out.transpose(1, 2)


def manual_mla_reference(q_abs, kv_latent, sm_scale, is_causal=True):
    # Reference implementation for correctness check
    q = q_abs.transpose(1, 2)
        
    B, H, N_Q, D = q.shape
    N_KV = kv_latent.shape[1]

    k = kv_latent.unsqueeze(1).expand(-1, H, -1, -1)
    v = k 
    
    attn_score = torch.matmul(q, k.transpose(-2, -1)) * sm_scale
    
    if is_causal and N_Q > 1:
        mask = torch.tril(torch.ones(N_Q, N_KV, device=q.device))
        attn_score = attn_score.masked_fill(mask == 0, float('-inf'))
        
    attn_probs = torch.softmax(attn_score.float(), dim=-1).to(q.dtype)
    out = torch.matmul(attn_probs, v)
    
    return out.transpose(1, 2)


def benchmark_kernel(func, name, args, n_repeat=50):
    torch.cuda.synchronize()
    torch.cuda.reset_peak_memory_stats()
    
    # Warmup
    try:
        for _ in range(3): func(*args)
    except torch.cuda.OutOfMemoryError:
        print(f"[{name}] OOM during warmup")
        torch.cuda.empty_cache()
        return None, None

    torch.cuda.synchronize()
    start = torch.cuda.Event(enable_timing=True)
    end = torch.cuda.Event(enable_timing=True)
    
    start.record()
    try:
        for _ in range(n_repeat):
            func(*args)
    except torch.cuda.OutOfMemoryError:
        print(f"[{name}] OOM during execution")
        torch.cuda.empty_cache()
        return None, None
    end.record()
    
    torch.cuda.synchronize()
    avg_time = start.elapsed_time(end) / n_repeat
    
    # 这里的 peak mem 包含了 Input Tensors + Output Tensor + Kernel overhead
    # 这是运行该操作所需的"真实峰值显存"
    peak_mem = torch.cuda.max_memory_allocated() / 1024**2 # MB
    
    return avg_time, peak_mem

def measure_tensor_size(shape, dtype, device):
    """辅助函数：实际测量 Tensor 分配占用的显存"""
    torch.cuda.synchronize()
    mem_before = torch.cuda.memory_allocated()
    t = torch.randn(shape, device=device, dtype=dtype)
    mem_after = torch.cuda.memory_allocated()
    size_mb = (mem_after - mem_before) / 1024**2
    return t, size_mb

def run_suite(B, N, H_Q, D, label):
    print(f"\n{'='*60}")
    print(f"--- {label} [Batch={B}, Seq={N}, Heads={H_Q}, Dim={D}] ---")
    print(f"{'='*60}")
    
    dtype = torch.float16
    device = "cuda"
    sm_scale = 1.0 / math.sqrt(D)
    
    # 1. Allocate Query (Common)
    try:
        q, _ = measure_tensor_size((B, N, H_Q, D), dtype, device)
    except torch.cuda.OutOfMemoryError:
        print("Skipping suite due to OOM on Query allocation.")
        return

    results = {}

    # === 1. Flash MLA (DeepSeek) ===
    try:
        # 实测 KV Tensor 大小
        kv_mla, kv_mla_size = measure_tensor_size((B, N, D), dtype, device)
        
        t, m = benchmark_kernel(flash_mla_prefill, "Flash MLA", (q, kv_mla, sm_scale))
        if t: 
            print(f"Flash MLA | KV Size: {kv_mla_size:.1f} MB | Run Peak: {m:.1f} MB | Time: {t:.3f} ms")
            results['MLA'] = (t, m, kv_mla_size)
        del kv_mla
    except torch.cuda.OutOfMemoryError:
        print("Flash MLA OOM during input allocation.")


    # === 2. PyTorch GQA (Llama 3 Standard, Group=8) ===
    group_size = 8
    if H_Q % group_size == 0:
        H_KV = H_Q // group_size
        try:
            # 实测 GQA KV Tensor 大小
            kv_gqa, kv_gqa_size = measure_tensor_size((B, N, H_KV, D), dtype, device)
            
            t, m = benchmark_kernel(pytorch_sdpa_native, f"PyTorch GQA (1/{group_size})", (q, kv_gqa, kv_gqa, sm_scale))
            if t:
                print(f"PyTorch GQA | KV Size: {kv_gqa_size:.1f} MB | Run Peak: {m:.1f} MB | Time: {t:.3f} ms")
                results['GQA'] = (t, m, kv_gqa_size)
            del kv_gqa
        except torch.cuda.OutOfMemoryError:
            print(f"PyTorch GQA (1/{group_size}) OOM during input allocation.")

    # === 3. PyTorch MHA (Standard) ===
    try:
        # 实测 MHA KV Tensor 大小
        kv_mha, kv_mha_size = measure_tensor_size((B, N, H_Q, D), dtype, device)
        
        t, m = benchmark_kernel(pytorch_sdpa_native, "PyTorch MHA", (q, kv_mha, kv_mha, sm_scale))
        if t:
            print(f"PyTorch MHA | KV Size: {kv_mha_size:.1f} MB | Run Peak: {m:.1f} MB | Time: {t:.3f} ms")
            results['MHA'] = (t, m, kv_mha_size)
        del kv_mha
    except torch.cuda.OutOfMemoryError:
        print("PyTorch MHA OOM during input allocation.")

    print("-" * 30)
    if 'MLA' in results and 'GQA' in results:
        t_mla, m_mla, kv_size_mla = results['MLA']
        t_gqa, m_gqa, kv_size_gqa = results['GQA']
        
        speedup = t_gqa / t_mla
        # 显存对比：不仅看 Peak（含Query/Output），更要看核心瓶颈 KV Cache 的实际体积
        kv_saving = kv_size_gqa / kv_size_mla
        
        print(f">>> [Speed] MLA is {speedup:.2f}x speed of GQA")
        print(f">>> [KV Cache Real Memory] GQA uses {kv_saving:.2f}x MORE memory than MLA")
    
    del q
    torch.cuda.empty_cache()

if __name__ == "__main__":
    print("\n=== Correctness Check ===")
    torch.manual_seed(42)
    B_check, N_check, H_check, D_check = 1, 256, 4, 128
    dtype = torch.float16
    device = "cuda"
    sm_scale = 1.0 / math.sqrt(D_check)

    try:
        q = torch.randn((B_check, N_check, H_check, D_check), device=device, dtype=dtype)
        kv = torch.randn((B_check, N_check, D_check), device=device, dtype=dtype)
        
        out_triton = flash_mla_prefill(q, kv, sm_scale)
        out_ref = manual_mla_reference(q, kv, sm_scale, is_causal=True)
        diff = (out_triton - out_ref).abs().max()
        print(f"Prefill Diff: {diff.item():.6f} {'[PASS]' if diff < 1e-2 else '[FAIL]'}")
        del q, kv, out_triton, out_ref
    except Exception as e:
        print(f"Prefill Test Error: {e}")

    print("\n=== Starting Industry Benchmark on A100 ===")

    run_suite(B=1, N=4096, H_Q=128, D=512, label="DeepSeek V2 Standard (4k)")
    run_suite(B=1, N=16384, H_Q=128, D=512, label="DeepSeek V2 Long Context (16k)")
    run_suite(B=1, N=32768, H_Q=128, D=512, label="DeepSeek V2 Extreme (32k)")
    run_suite(B=8, N=4096, H_Q=32, D=128, label="Llama-style 7B Params (Batch 8)")