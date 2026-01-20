# FlashMLA: High-Performance Multi-Head Latent Attention Kernels

[![License: MIT](https://img.shields.io/badge/License-MIT-f39c12.svg)](https://opensource.org/licenses/MIT)
[![Python 3.9+](https://img.shields.io/badge/python-3.9+-blue.svg)](https://www.python.org/downloads/)
[![Triton](https://img.shields.io/badge/Implementation-OpenAI_Triton-000000.svg)](https://github.com/openai/triton)
[![CUDA 11.8+](https://img.shields.io/badge/CUDA-11.8+-76B900.svg)](https://developer.nvidia.com/cuda-toolkit)

**FlashMLA** is an fast and memory-efficient collection of MLA (Multi-Head Latent Attention) kernels implemented in OpenAI Triton. Specifically optimized for the MLA architecture used in **DeepSeek-V2** and **DeepSeek-V3**, it bridges the gap between massive context windows and hardware efficiency.

---
```
FlashMLA/
├── flash_mla/
│   ├── __init__.py
│   ├── ops/
│   │   ├── __init__.py
│   │   └── interface.py
│   └── kernels/
│       ├── __init__.py
│       ├── decode_kernel.py   # Decode Triton Kernel
│       └── prefill_kernel.py  # Prefill Triton Kernel
├── tests/
│   ├── __init__.py
│   └── test_correctness.py
├── benchmarks/
│   └── benchmark.py
├── setup.py
└── README.md
```



---

## Key Features

* **FlashAttention Prefill**: Optimized tile-based kernel for the prefill stage, handling MLA’s absorbed projections with minimal overhead.
* **FlashDecoding (Split-KV)**: Accelerates the decoding phase by parallelizing over the sequence length, significantly reducing Time-To-First-Token (TTFT).
* **Matrix Absorption**: Fuses projection matrices into the attention computation, transforming compute-heavy operations into streamlined, memory-bound tasks.
* **Shared-KV Native**: Full support for MLA’s compressed KV latent vectors, reducing KV cache memory footprint by up to 90% compared to standard MHA.

---

## Architecture Comparison

MLA redefines KV cache efficiency by decoupling the KV latent dimension from the number of attention heads.

| Architecture | KV Cache Shape | Memory Impact | Scaling Bottleneck |
| :--- | :--- | :--- | :--- |
| **Standard MHA** | `[B, N, H, D]` | **Massive** | Linear with Head Count |
| **MLA (FlashMLA)** | `[B, N, D_latent]`| **Tiny** | Independent of Heads |

---
