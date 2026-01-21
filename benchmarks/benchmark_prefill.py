
import torch
import torch.nn.functional as F
import math
import csv
import os
import gc
import pandas as pd
from flash_mla.ops.interface import flash_mla_prefill


def pytorch_sdpa_native(q, k, v, sm_scale):

    B, N, H_Q, D = q.shape
    H_K = k.shape[2]
    
    q = q.transpose(1, 2)
    k = k.transpose(1, 2)
    v = v.transpose(1, 2)
    
    if H_Q != H_K:
        # GQA
        n_group = H_Q // H_K
        k = k.repeat_interleave(n_group, dim=1)
        v = v.repeat_interleave(n_group, dim=1)
    
    out = F.scaled_dot_product_attention(q, k, v, scale=sm_scale, is_causal=True)
    return out.transpose(1, 2)


def benchmark_kernel(func, name, args, n_repeat=10):
    torch.cuda.empty_cache()
    gc.collect()
    torch.cuda.synchronize()
    torch.cuda.reset_peak_memory_stats()
    
    # Warmup
    try:
        for _ in range(3): func(*args)
    except torch.cuda.OutOfMemoryError:
        return None, None
    except Exception as e:
        return None, None

    torch.cuda.synchronize()
    start = torch.cuda.Event(enable_timing=True)
    end = torch.cuda.Event(enable_timing=True)
    
    start.record()
    try:
        for _ in range(n_repeat):
            func(*args)
    except torch.cuda.OutOfMemoryError:
        return None, None
    except Exception as e:
        return None, None
    end.record()
    
    torch.cuda.synchronize()
    avg_time = start.elapsed_time(end) / n_repeat
    peak_mem = torch.cuda.max_memory_allocated() / 1024**2
    return avg_time, peak_mem

def measure_tensor_size(shape, dtype, device):

    torch.cuda.synchronize()
    torch.cuda.empty_cache() 
    mem_before = torch.cuda.memory_allocated()
    try:
        t = torch.randn(shape, device=device, dtype=dtype)
    except torch.cuda.OutOfMemoryError:
        return None, 0.0
    mem_after = torch.cuda.memory_allocated()
    size_mb = (mem_after - mem_before) / 1024**2
    return t, size_mb

def run_suite(exp_name, B, N, H_Q, D, csv_data):

    print(f"Running {exp_name}: B={B}, N={N}...", end="", flush=True)

    dtype = torch.float16
    device = "cuda"
    sm_scale = 1.0 / math.sqrt(D)
    
    # common query
    try:
        q_ret = measure_tensor_size((B, N, H_Q, D), dtype, device)
        if q_ret[0] is None:
             print(f"SKIP: OOM on Query")
             return
        q, _ = q_ret
    except Exception:
        return

    # Flash MLA
    print("")
    try:
        torch.cuda.empty_cache()
        ret = measure_tensor_size((B, N, D), dtype, device)
        if ret[0] is not None:
            kv_mla, kv_size = ret
            t, m = benchmark_kernel(flash_mla_prefill, "Flash MLA", (q, kv_mla, sm_scale), n_repeat=10)
            if t:
                print(f"[FlashMLA] Latency: {t:.2f}ms | KV Storage: {kv_size:.0f}MB | Peak Mem: {m:.0f}MB")
                csv_data.append({
                    "Experiment": exp_name, "Batch": B, "SeqLen": N, "Heads": H_Q, "Dim": D,
                    "Method": "FlashMLA", "Time_ms": t, "PeakMem_MB": m, "KV_Size_MB": kv_size
                })
            del kv_mla
        else:
            print(f"[FlashMLA] OOM (Alloc)")
            csv_data.append({"Experiment": exp_name, "Batch": B, "SeqLen": N, "Method": "FlashMLA", "Time_ms": None})
    except Exception: pass

    # PyTorch GQA (Group=8)
    group_size = 8
    H_KV = H_Q // group_size
    try:
        torch.cuda.empty_cache()
        ret = measure_tensor_size((B, N, H_KV, D), dtype, device)
        if ret[0] is not None:
            kv_gqa, kv_size = ret
            t, m = benchmark_kernel(pytorch_sdpa_native, "GQA", (q, kv_gqa, kv_gqa, sm_scale), n_repeat=10)
            if t:
                print(f"[GQA] Latency: {t:.2f}ms | KV Storage: {kv_size:.0f}MB | Peak Mem: {m:.0f}MB (Inflated)")
                csv_data.append({
                    "Experiment": exp_name, "Batch": B, "SeqLen": N, "Heads": H_Q, "Dim": D,
                    "Method": "PyTorch_GQA", "Time_ms": t, "PeakMem_MB": m, "KV_Size_MB": kv_size
                })
            del kv_gqa
        else:
            print(f"[GQA] OOM (Alloc)")
            csv_data.append({"Experiment": exp_name, "Batch": B, "SeqLen": N, "Method": "PyTorch_GQA", "Time_ms": None})
    except Exception: pass

    # PyTorch MHA
    try:
        torch.cuda.empty_cache()
        ret = measure_tensor_size((B, N, H_Q, D), dtype, device)
        if ret[0] is not None:
            kv_mha, kv_size = ret
            t, m = benchmark_kernel(pytorch_sdpa_native, "MHA", (q, kv_mha, kv_mha, sm_scale), n_repeat=10)
            if t:
                print(f"[MHA] Latency: {t:.2f}ms | KV Storage: {kv_size:.0f}MB | Peak Mem: {m:.0f}MB")
                csv_data.append({
                    "Experiment": exp_name, "Batch": B, "SeqLen": N, "Heads": H_Q, "Dim": D,
                    "Method": "PyTorch_MHA", "Time_ms": t, "PeakMem_MB": m, "KV_Size_MB": kv_size
                })
            del kv_mha
        else:
            print(f"[MHA] OOM (Alloc)")
            csv_data.append({"Experiment": exp_name, "Batch": B, "SeqLen": N, "Method": "PyTorch_MHA", "Time_ms": None})
    except Exception: pass
    
    del q
    torch.cuda.empty_cache()

if __name__ == "__main__":

    csv_data = []

    H = 128
    D = 512

    print(f"\n[Experiment 1] Variable SeqLen (Batch=4, Heads={H}, Dim={D})")
    seq_lens = [1024, 2048, 4096, 8192, 16384, 32768, 65536]
    for n in seq_lens:
        run_suite("Var_SeqLen", B=4, N=n, H_Q=H, D=D, csv_data=csv_data)

    print(f"\n[Experiment 2] Variable Batch (Seq=4096, Heads={H}, Dim={D})")
    batches = [1, 2, 4, 8, 16, 32, 64]
    for b in batches:
        run_suite("Var_Batch", B=b, N=4096, H_Q=H, D=D, csv_data=csv_data)

    csv_file = "benchmark_prefill.csv"
    if csv_data:
        keys = csv_data[0].keys()
        with open(csv_file, 'w', newline='') as f:
            writer = csv.DictWriter(f, fieldnames=keys)
            writer.writeheader()
            writer.writerows(csv_data)
        print(f"\n[Success] Results saved to {csv_file}")
    else:
        print("\n[Warning] No results recorded.")