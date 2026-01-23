import torch
import torch.nn.functional as F
import math
import gc
import pandas as pd
import torch.multiprocessing as mp
from flash_mla.ops.interface import flash_mla_decode_core

try:
    from flash_attn import flash_attn_with_kvcache
    HAS_FLASH_ATTN = True
except ImportError:
    HAS_FLASH_ATTN = False

def _worker_run_benchmark(method, B, N_CTX, H_Q, D_MLA, return_queue):
    try:
        torch.cuda.set_device(0)
        dtype, device, D_H = torch.float16, "cuda", 128
        sm_scale = 1.0 / math.sqrt(D_H)
        q = torch.randn((B, H_Q, D_H), device=device, dtype=dtype)
        n_repeat = 100

        def get_kv_size(shapes):
            torch.cuda.synchronize()
            torch.cuda.empty_cache()
            m1 = torch.cuda.memory_allocated()
            tensors = [torch.randn(s, device=device, dtype=dtype) for s in shapes]
            torch.cuda.synchronize()
            return tensors, (torch.cuda.memory_allocated() - m1) / 1024**2

        if method == "FlashMLA":
            ret, kv_sz = get_kv_size([(B, N_CTX, D_MLA)])
            fn = lambda: flash_mla_decode_core(q, ret[0], sm_scale)
        elif method == "Flash-GQA":
            ret, kv_sz = get_kv_size([(B, N_CTX, H_Q//8, D_H), (B, N_CTX, H_Q//8, D_H)])
            fn = lambda: flash_attn_with_kvcache(q.unsqueeze(1), ret[0], ret[1], softmax_scale=sm_scale, causal=False)
        elif method == "Flash-MHA":
            ret, kv_sz = get_kv_size([(B, N_CTX, H_Q, D_H), (B, N_CTX, H_Q, D_H)])
            fn = lambda: flash_attn_with_kvcache(q.unsqueeze(1), ret[0], ret[1], softmax_scale=sm_scale, causal=False)
        elif method == "PyTorch":
            ret, kv_sz = get_kv_size([(B, N_CTX, H_Q, D_H), (B, N_CTX, H_Q, D_H)])
            def fn():
                return F.scaled_dot_product_attention(q.unsqueeze(1).transpose(1, 2), ret[0].transpose(1, 2), ret[1].transpose(1, 2), scale=sm_scale)

        for _ in range(10): fn()
        torch.cuda.synchronize()
        start, end = torch.cuda.Event(enable_timing=True), torch.cuda.Event(enable_timing=True)
        start.record()
        for _ in range(n_repeat): fn()
        end.record()
        torch.cuda.synchronize()
        return_queue.put({"status": "success", "lat": (start.elapsed_time(end)/n_repeat)*1000, "kv": kv_sz})
    except Exception as e: return_queue.put({"status": "error", "note": str(e)})

def run_decode_suite(exp_name, B, N_CTX, H_Q, D_MLA, csv_data):
    print(f"\n{'='*75}\n EXPERIMENT: {exp_name} | Batch: {B} | Context: {N_CTX}\n{'-'*75}")
    print(f"{'Method':<18} | {'Latency (us)':<15} | {'KV Size (MB)':<15}")
    print("-" * 75)
    for m in ["FlashMLA", "Flash-GQA", "Flash-MHA", "PyTorch"]:
        q = mp.Queue()
        p = mp.Process(target=_worker_run_benchmark, args=(m, B, N_CTX, H_Q, D_MLA, q))
        p.start(); p.join()
        res = q.get() if not q.empty() else {"status": "crash"}
        if res["status"] == "success":
            print(f"{m:<18} | {res['lat']:>12.2f} us | {res['kv']:>12.1f} MB")
            csv_data.append({"Exp": exp_name, "B": B, "N": N_CTX, "Method": m, "Lat_us": res['lat'], "KV_MB": res['kv']})
        else: print(f"{m:<18} | {'FAIL/OOM':>12} | {'-':>12}")

if __name__ == "__main__":
    mp.set_start_method('spawn', force=True)
    csv_results = []
    H_Q, D_MLA = 128, 512

    for n in [1024, 2048, 4096, 8192, 16384, 32768, 65536, 131072]:
        run_decode_suite("ContextScaling", 4, n, H_Q, D_MLA, csv_results)
        
    for b in [1, 2, 4, 8, 16, 32, 64, 128]:
        run_decode_suite("BatchScaling", b, 4096, H_Q, D_MLA, csv_results)

    pd.DataFrame(csv_results).to_csv("benchmark_decode.csv", index=False)