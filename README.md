# FlashMLA: High-Performance Multi-Head Latent Attention Kernels

> This serves as the kernel repository for [vLLM_FlashMLA](https://github.com/Jayden-Xu/vllm_FlashMLA).

FlashMLA delivers **fast and memory-efficient** Multi-Head Latent Attention (MLA) kernels 
in OpenAI Triton, enabling **8-64× smaller KV cache** compared to standard attention while 
maintaining competitive throughput, shines where memory 
is the bottleneck.

---

## Performance Benchmarks

Running on **NVIDIA A100-80GB** with **DeepSeek-V2-Lite-Chat**, with Cuda Graph enabled.

### System-Level Performance

![FlashMLA Decode Throughput](./assets/benchmark_throughput.png)

![FlashMLA Decode Latency](./assets/benchmark_latency.png)

### Core Kernels

> For core kernels, we focus on the Memory Footprint for KV Cache during the decode phase.

Key Architectural Parameters:
- FlashMLA: `d_c = 512` (Compressed KV Latent)
- MHA/GQA Baselines: `d_h = 128` (Standard Head Dimension)
- Attention Heads: `n_q_heads = 128`
- GQA Config: Group Size = 8 (`n_kv_heads = 16`)

![FlashMLA Decode Memory Footprint - Batch Scaling](./assets/kv_cache_batch.png)

![FlashMLA Decode Memory Footprint - Context Scaling](./assets/kv_cache_context.png)

---
