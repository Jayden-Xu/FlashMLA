# FlashMLA: High-Performance Multi-Head Latent Attention Kernels

FlashMLA delivers **fast and memory-efficient** Multi-Head Latent Attention (MLA) kernels 
in OpenAI Triton, enabling **8-64× smaller KV cache** compared to standard attention while 
maintaining competitive throughput. Purpose-built for long-context scenarios where memory 
is the bottleneck.

---

## Roadmap

FlashMLA is under active development. Our goal is to provide a production-ready Triton implementation for MLA-based models.

- [ ] Native RoPE support
- [ ] Paged Attention support
- [ ] Fused kernels & Core kernels tuning
- [ ] FP8 precision support

---

## Performance Benchmarks

Benchmarks are conducted on **NVIDIA A100 (80GB)**. We compare **FlashMLA** against industry-standard **FlashAttention-2 (FA2)** implementations and the **PyTorch SDPA** baseline.
> All benchmarks measure pure attention kernel performance, to enable fair comparison across different implementations.

Key Architectural Parameters:
- FlashMLA: `d_c = 512` (Compressed KV Latent)
- MHA/GQA Baselines: `d_h = 128` (Standard Head Dimension)
- Attention Heads: `n_q_heads = 128`
- GQA Config: Group Size = 8 (`n_kv_heads = 16`)

### Prefill Phase: TTFT & Footprint

In the prefill stage, FlashMLA focuses on maximizing KV cache capacity. While PyTorch SDPA and FA2 are highly optimized for dense computation, FlashMLA enables processing sequences that would otherwise exceed hardware memory limits.

#### Sequence Length Scaling (B = 4)

![](./benchmarks/results/prefill_seqlen.png)

#### Batch Size Scaling (N = 4096)

![](./benchmarks/results/prefill_batch.png)

### Decode Phase: TPOT & Footprint

Decoding is heavily memory-bandwidth bound. FlashMLA leverages its 1/64th (vs. MHA) and 1/8th (vs. GQA) I/O requirements to maintain high efficiency in long-context scenarios.

#### Context Length Scaling (B = 4)

![](./benchmarks/results/decode_context.png)

#### Batch Size Scaling (N = 4096)

![](./benchmarks/results/decode_batch.png)

---
