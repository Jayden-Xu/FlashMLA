import torch
import torch.nn.functional as F
import math
import gc
import pandas as pd
from flash_mla.ops.interface import flash_mla_prefill

try:
    from flash_attn import flash_attn_func
    HAS_FLASH_ATTN = True
except ImportError:
    HAS_FLASH_ATTN = False

def benchmark_kernel(func, args, n_repeat=10):
    torch.cuda.empty_cache()
    gc.collect()
    torch.cuda.synchronize()
    torch.cuda.reset_peak_memory_stats()
    try:
        for _ in range(3): func(*args)
    except Exception: return None, None

    torch.cuda.synchronize()
    start, end = torch.cuda.Event(enable_timing=True), torch.cuda.Event(enable_timing=True)
    start.record()
    for _ in range(n_repeat): func(*args)
    end.record()
    torch.cuda.synchronize()
    avg_time = start.elapsed_time(end) / n_repeat
    peak_mem = torch.cuda.max_memory_allocated() / 1024**2
    return avg_time, peak_mem

def measure_kv_storage(shapes, dtype, device):
    torch.cuda.synchronize()
    torch.cuda.empty_cache() 
    mem_before = torch.cuda.memory_allocated()
    tensors = [torch.randn(s, device=device, dtype=dtype) for s in shapes]
    torch.cuda.synchronize()
    size_mb = (torch.cuda.memory_allocated() - mem_before) / 1024**2
    return tensors, size_mb

def run_suite(exp_name, B, N, H_Q, D_MLA, csv_data):
    print(f"\n{'='*90}\n EXPERIMENT: {exp_name} | Batch: {B} | SeqLen: {N}\n{'-'*90}")
    print(f"{'Method':<18} | {'Latency (ms)':<15} | {'KV Size (MB)':<15} | {'Peak Mem (MB)':<15}")
    print("-" * 90)

    dtype, device, D_H = torch.float16, "cuda", 128
    sm_scale = 1.0 / math.sqrt(D_H)
    q = torch.randn((B, N, H_Q, D_H), device=device, dtype=dtype)

    ret, kv_sz = measure_kv_storage([(B, N, D_MLA)], dtype, device)
    t, m = benchmark_kernel(flash_mla_prefill, (q, ret[0], sm_scale))
    if t:
        print(f"{'FlashMLA':<18} | {t:>12.2f} ms | {kv_sz:>12.1f} MB | {m:>12.1f} MB")
        csv_data.append({"Exp": exp_name, "B": B, "N": N, "Method": "FlashMLA", "Time_ms": t, "KV_MB": kv_sz, "Peak_MB": m})

    H_KV_GQA = H_Q // 8
    ret, kv_sz = measure_kv_storage([(B, N, H_KV_GQA, D_H), (B, N, H_KV_GQA, D_H)], dtype, device)
    t, m = benchmark_kernel(flash_attn_func, (q, ret[0], ret[1], 0.0, sm_scale, True))
    if t:
        print(f"{'Flash-GQA':<18} | {t:>12.2f} ms | {kv_sz:>12.1f} MB | {m:>12.1f} MB")
        csv_data.append({"Exp": exp_name, "B": B, "N": N, "Method": "Flash-GQA", "Time_ms": t, "KV_MB": kv_sz, "Peak_MB": m})

    ret, kv_sz = measure_kv_storage([(B, N, H_Q, D_H), (B, N, H_Q, D_H)], dtype, device)
    t, m = benchmark_kernel(flash_attn_func, (q, ret[0], ret[1], 0.0, sm_scale, True))
    if t:
        print(f"{'Flash-MHA':<18} | {t:>12.2f} ms | {kv_sz:>12.1f} MB | {m:>12.1f} MB")
        csv_data.append({"Exp": exp_name, "B": B, "N": N, "Method": "Flash-MHA", "Time_ms": t, "KV_MB": kv_sz, "Peak_MB": m})

    def pt_sdpa(q, k, v, s): return F.scaled_dot_product_attention(q.transpose(1, 2), k.transpose(1, 2), v.transpose(1, 2), scale=s, is_causal=True)
    t, m = benchmark_kernel(pt_sdpa, (q, ret[0], ret[1], sm_scale))
    if t:
        print(f"{'PyTorch Native':<18} | {t:>12.2f} ms | {kv_sz:>12.1f} MB | {m:>12.1f} MB")
        csv_data.append({"Exp": exp_name, "B": B, "N": N, "Method": "PyTorch", "Time_ms": t, "KV_MB": kv_sz, "Peak_MB": m})

if __name__ == "__main__":
    csv_results = []
    H_Q, D_MLA = 128, 512

    for n in [1024, 2048, 4096, 8192, 16384, 32768, 65536, 131072]:
        run_suite("Var_SeqLen", 4, n, H_Q, D_MLA, csv_results)

    for b in [1, 2, 4, 8, 16, 32, 64, 128]:
        run_suite("Var_Batch", b, 4096, H_Q, D_MLA, csv_results)

    pd.DataFrame(csv_results).to_csv("benchmark_prefill.csv", index=False)