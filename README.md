# FlashMLA: High-Performance Multi-Head Latent Attention Kernels

FlashMLA is an fast and memory-efficient collection of MLA (Multi-Head Latent Attention) kernels implemented in OpenAI Triton. Specifically optimized for the MLA architecture, it bridges the gap between massive context windows and hardware efficiency.

---
## Performance Benchmarks

Benchmarks are conducted on NVIDIA A100 (80GB). We compare FlashMLA against the industry-standard FlashAttention-2 (FA2) implementations for Multi-Head Attention (MHA) and Grouped-Query Attention (GQA).

Key Architectural Parameters:
- FlashMLA: $d_c = 512$ (Compressed KV Latent)
- MHA/GQA Baselines: $d_h = 128$ (Standard Head Dimension)
- Attention Heads: $n_{q\_heads} = 128$
- GQA Config: Group Size = 8 ($n_{kv\_heads} = 16$)

### Prefill Phase

#### Sequence Length Scaling (Batch = 4)

In the prefill stage, FlashMLA focuses on maximizing KV cache capacity. While standard FA2 kernels are highly optimized for compute, FlashMLA enables processing sequences that would otherwise exceed hardware memory limits.

> Note: MLA is ~1.5x slower than GQA in prefill due to the computational complexity of Matrix Absorption. However, it reduces memory footprint by 8x compared to GQA and 64x compared to MHA.

![](./benchmarks/results/prefill_seqlen.png)

#### Batch Size Scaling (Seq = 4096)

![](./benchmarks/results/prefill_batch.png)

### Decode Phase

Decoding is heavily memory-bandwidth bound. FlashMLA leverages its 1/64th (vs. MHA) and 1/8th (vs. GQA) I/O requirements to maintain high efficiency in long-context scenarios.

#### Context Length Scaling (Batch = 4)

While the highly-tuned FA2-GQA kernel currently leads in raw latency for small batches, FlashMLA significantly outperforms standard MHA. Its true value lies in long-context scalability: at 64K context, MLA uses only 256MB, whereas GQA requires 2GB, effectively allowing for 8x larger batches or 8x longer sequences on the same hardware.

![](./benchmarks/results/decode_context.png)

#### Batch Size Scaling (Seq = 4096)

![](./benchmarks/results/decode_batch.png)

---
