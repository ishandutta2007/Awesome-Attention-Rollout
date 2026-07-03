# The Hardware-Bus I/O Latency Penalty

Frequent memory read/write cycles to high-bandwidth memory (HBM) during sequence multiplication limit performance.

### Detailed Concept
Sequential matrix products require fetching weights from GPU memory to cache. To solve this, custom Triton or CUDA kernels are written to fuse the multiplication loop directly within GPU SRAM.

### Diagram
```mermaid
flowchart LR
    GlobalMemory[Global Memory / HBM] -->|Frequent Slow Transfers| Registers[GPU Registers / SRAM]
    Registers -->|Fused Triton Kernel Loop| Output[Rollout Map]
```
