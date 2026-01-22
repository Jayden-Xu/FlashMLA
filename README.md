# FlashMLA: High-Performance Multi-Head Latent Attention Kernels

FlashMLA delivers **fast and memory-efficient** Multi-Head Latent Attention (MLA) kernels 
in OpenAI Triton, enabling **8-64× smaller KV cache** compared to standard attention while 
maintaining competitive throughput. Purpose-built for long-context scenarios where memory 
is the bottleneck.

---

## Key Features

**Core Prefill Kernel**: FlashAttention-2 + MLA
> Long-context and large-batch prompt processing

**Core Decode Kernel**: FlashDecoding + MLA
> Memory-efficient autoregressive generation

---

## Roadmap

FlashMLA is under active development. Our goal is to provide a production-ready Triton implementation for MLA-based models.

- [ ] Native RoPE support
- [ ] Paged Attention support
- [ ] Kernel tuning
- [ ] FP8 precision support (H100+ only)

---

## Performance Benchmarks

Benchmarks are conducted on **NVIDIA A100 (80GB)**. We compare **FlashMLA** against industry-standard **FlashAttention-2 (FA2)** implementations and the **PyTorch SDPA** baseline.

Key Architectural Parameters:
- FlashMLA: `d_c = 512` (Compressed KV Latent)
- MHA/GQA Baselines: `d_h = 128` (Standard Head Dimension)
- Attention Heads: `n_q_heads = 128`
- GQA Config: Group Size = 8 (`n_kv_heads = 16`)

### Prefill Phase: TTFT & Footprint

**Sequence Length Scaling (B = 4)**

![](./benchmarks/results/prefill_seqlen.png)

**Batch Size Scaling (N = 4096)**

![](./benchmarks/results/prefill_batch.png)

### Decode Phase: TPOT & Footprint

**Context Length Scaling (B = 4)**

![](./benchmarks/results/decode_context.png)

**Batch Size Scaling (N = 4096)**

![](./benchmarks/results/decode_batch.png)

---
