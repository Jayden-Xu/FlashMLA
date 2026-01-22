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

In the prefill stage, FlashMLA focuses on maximizing KV cache capacity. While PyTorch SDPA kernel and FA2 kernels are highly optimized for compute, FlashMLA enables processing sequences that would otherwise exceed hardware memory limits.

![](./benchmarks/results/prefill_seqlen.png)

#### Batch Size Scaling (Seq = 4096)

![](./benchmarks/results/prefill_batch.png)

### Decode Phase

Decoding is heavily memory-bandwidth bound. FlashMLA leverages its 1/64th (vs. MHA) and 1/8th (vs. GQA) I/O requirements to maintain high efficiency in long-context scenarios.

#### Context Length Scaling (Batch = 4)

![](./benchmarks/results/decode_context.png)

#### Batch Size Scaling (Seq = 4096)

![](./benchmarks/results/decode_batch.png)

---
