<div align="center">
<h1>FlashMLA</h1>
<p>
<strong>High-Performance Multi-Head Latent Attention Kernels</strong>
</p>
</div>

FlashMLA is a high-performance kernel library specifically optimized for Multi-Head Latent Attention (MLA) architectures. Built on OpenAI Triton, it delivers state-of-the-art decoding performance for MLA-based models.

By leveraging MLA's compressed latent vectors, FlashMLA enables 8-64× smaller KV cache footprints compared to standard attention mechanisms while maintaining competitive throughput. It is designed for memory-bound scenarios where efficient cache management is critical.

> This repository hosts the standalone kernel primitives. For the end-to-end inference backend, please visit [vLLM_FlashMLA](https://github.com/Jayden-Xu/vllm_FlashMLA).

---
## Key Features

Extreme Speed & Memory Efficiency

Minimum Overhead with CUDA Graph

Production Ready with dynamic Split-K scheduling

---

## Performance Benchmarks

Benchmarks were conducted on **NVIDIA A100-80GB** using **DeepSeek-V2-Lite-Chat** with **CUDA Graph enabled**.

### Decoding Performance

![](./assets/flashmla_benchmark_seqlen_1024.png)
![](./assets/flashmla_benchmark_seqlen_2048.png)
![](./assets/flashmla_benchmark_seqlen_4096.png)
![](./assets/flashmla_benchmark_seqlen_8192.png)

---
