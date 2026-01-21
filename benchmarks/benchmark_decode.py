
import torch
import torch.nn.functional as F
import math
import csv
import torch.multiprocessing as mp
from flash_mla.ops.interface import flash_mla_decode


def pytorch_sdpa_decode_step(q, k_cache, v_cache, sm_scale):
    q = q.transpose(1, 2)
    k = k_cache.transpose(1, 2)
    v = v_cache.transpose(1, 2)
    
    B, H_Q, _, D = q.shape
    B, H_K, N, _ = k.shape

    if H_Q != H_K:
        n_group = H_Q // H_K
        k = k.repeat_interleave(n_group, dim=1)
        v = v.repeat_interleave(n_group, dim=1)
    
    out = F.scaled_dot_product_attention(q, k, v, scale=sm_scale, is_causal=False)
    return out.transpose(1, 2)


def _worker_run_benchmark(method, B, N_CTX, H_Q, D, return_queue):

    try:
        torch.cuda.set_device(0)
        dtype = torch.float16
        device = "cuda"
        sm_scale = 1.0 / math.sqrt(D)

        q_mla = torch.randn((B, H_Q, D), device=device, dtype=dtype)
        q_torch = q_mla.unsqueeze(1)

        kv_size_mb = 0
        latency_us = 0
        
        n_warmup = 5
        n_repeat = 50 

        if method == "FlashMLA":
            kv = torch.randn((B, N_CTX, D), device=device, dtype=dtype)
            kv_size_mb = (kv.numel() * 2) / 1024**2
            
            for _ in range(n_warmup): flash_mla_decode(q_mla, kv, sm_scale)
            
            torch.cuda.synchronize()
            start = torch.cuda.Event(enable_timing=True)
            end = torch.cuda.Event(enable_timing=True)
            start.record()
            for _ in range(n_repeat):
                flash_mla_decode(q_mla, kv, sm_scale)
            end.record()
            torch.cuda.synchronize()
            latency_us = start.elapsed_time(end) / n_repeat * 1000

        elif method == "PyTorch_GQA":
            group_size = 8
            H_KV = H_Q // group_size
            kv = torch.randn((B, N_CTX, H_KV, D), device=device, dtype=dtype)
            kv_size_mb = (kv.numel() * 2) / 1024**2
            
            for _ in range(n_warmup): pytorch_sdpa_decode_step(q_torch, kv, kv, sm_scale)
            
            torch.cuda.synchronize()
            start = torch.cuda.Event(enable_timing=True)
            end = torch.cuda.Event(enable_timing=True)
            start.record()
            for _ in range(n_repeat):
                pytorch_sdpa_decode_step(q_torch, kv, kv, sm_scale)
            end.record()
            torch.cuda.synchronize()
            latency_us = start.elapsed_time(end) / n_repeat * 1000

        elif method == "PyTorch_MHA":
            kv = torch.randn((B, N_CTX, H_Q, D), device=device, dtype=dtype)
            kv_size_mb = (kv.numel() * 2) / 1024**2
            
            for _ in range(n_warmup): pytorch_sdpa_decode_step(q_torch, kv, kv, sm_scale)
            
            torch.cuda.synchronize()
            start = torch.cuda.Event(enable_timing=True)
            end = torch.cuda.Event(enable_timing=True)
            start.record()
            for _ in range(n_repeat):
                pytorch_sdpa_decode_step(q_torch, kv, kv, sm_scale)
            end.record()
            torch.cuda.synchronize()
            latency_us = start.elapsed_time(end) / n_repeat * 1000

        return_queue.put({"status": "success", "latency": latency_us, "kv_mb": kv_size_mb})
    
    except torch.cuda.OutOfMemoryError:
        return_queue.put({"status": "oom", "note": "Alloc OOM"})
    except Exception as e:
        return_queue.put({"status": "error", "note": str(e)})


def run_safe_benchmark(method, B, N_CTX, H_Q, D):
    """
    Spawns a new process for each benchmark run.
    """
    queue = mp.Queue()
    p = mp.Process(target=_worker_run_benchmark, args=(method, B, N_CTX, H_Q, D, queue))
    p.start()
    p.join()
    
    result = None
    if not queue.empty():
        result = queue.get()
    
    if p.exitcode != 0:
        return {"status": "crash", "note": "Crash/IllegalMem"}
    
    if result is None:
        return {"status": "crash", "note": "No Result"}
        
    return result

def run_decode_suite(exp_name, B, N_CTX, H_Q, D, csv_data):
    print(f"Running {exp_name}: B={B}, Context={N_CTX}...", end="", flush=True)
    
    methods = ["FlashMLA", "PyTorch_GQA", "PyTorch_MHA"]
    
    for method in methods:
        res = run_safe_benchmark(method, B, N_CTX, H_Q, D)
        
        status = res.get("status", "error")
        
        if status == "success":
            lat = res["latency"]
            kv = res["kv_mb"]
            print(f" [{method}: {lat:.1f}us | {kv:.0f}MB]", end="")
            csv_data.append({
                "Experiment": exp_name, "Batch": B, "Context": N_CTX, "Heads": H_Q, "Dim": D,
                "Method": method, "Latency_us": lat, "KV_MB": kv
            })
        else:
            note = res.get("note", "Fail")
            print(f" [{method}: {note}]", end="")
            # Record OOM as 0 latency for plotting
            csv_data.append({
                "Experiment": exp_name, "Batch": B, "Context": N_CTX, "Heads": H_Q, "Dim": D,
                "Method": method, "Latency_us": 0, "KV_MB": 0, "Note": note
            })
            
    print("")

if __name__ == "__main__":

    mp.set_start_method('spawn', force=True)
    
    print("=== FlashMLA Decode Benchmark (Multi-Process Safe) ===")
    csv_data = []
    H, D = 128, 512

    print("\n[Exp 1] Context Scaling (Batch=4, Heads=128, Dim=512)")
    ctx_lens = [1024, 2048, 4096, 8192, 16384, 32768, 65536] 
    for n in ctx_lens:
        run_decode_suite("ContextScaling", 4, n, H, D, csv_data)

    print("\n[Exp 2] Batch Scaling (Context=4k, Heads=128, Dim=512)")
    batches = [1, 2, 4, 8, 16, 32, 64]
    for b in batches:
        run_decode_suite("BatchScaling", b, 4096, H, D, csv_data)

    # Save
    csv_file = "benchmark_decode.csv"
    if csv_data:
        all_keys = set().union(*(d.keys() for d in csv_data))
        preferred_order = ["Experiment", "Batch", "Context", "Heads", "Dim", "Method", "Latency_us", "KV_MB", "Note"]
        keys = sorted(all_keys, key=lambda k: preferred_order.index(k) if k in preferred_order else 999)
        
        with open(csv_file, 'w', newline='') as f:
            writer = csv.DictWriter(f, fieldnames=keys)
            writer.writeheader()
            writer.writerows(csv_data)
        print(f"\n[Success] Results saved to {csv_file}")
    else:
        print("\n[Warning] No results.")