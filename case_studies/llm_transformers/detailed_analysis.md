# Large Language Models: Space-Time Tradeoffs at Scale

## Overview
Modern LLMs are a masterclass in space-time tradeoffs. With models reaching trillions of parameters, every architectural decision trades memory for computation.

## 1. Attention Mechanisms

### Standard Attention (O(n²) Space)
```python
# Naive attention: Store full attention matrix
def standard_attention(Q, K, V):
    # Q, K, V: [batch, seq_len, d_model]
    scores = Q @ K.T / sqrt(d_model)  # [batch, seq_len, seq_len]
    attn = softmax(scores)            # Must store entire matrix!
    output = attn @ V
    return output

# Memory: O(seq_len²) - becomes prohibitive for long sequences
# For seq_len=32K: 4GB just for attention matrix!
```

### Flash Attention (O(n) Space)
```python
# Recompute attention in blocks during backward pass
def flash_attention(Q, K, V, block_size=256):
    # Process in blocks, never materializing full matrix
    output = []
    for q_block in chunks(Q, block_size):
        block_out = compute_block_attention(q_block, K, V)
        output.append(block_out)
    return concat(output)

# Memory: O(seq_len) - linear in sequence length!
# Time: ~2x slower but enables 10x longer sequences
```

### Real Impact
- GPT-3: Limited to 2K tokens due to quadratic memory
- GPT-4 with Flash: 32K tokens with same hardware
- Claude: 100K+ tokens using similar techniques

## 2. KV-Cache Optimization

### Standard KV-Cache
```python
# During generation, cache keys and values
class StandardKVCache:
    def __init__(self, max_seq_len, n_layers, n_heads, d_head):
        # Cache for all positions
        self.k_cache = zeros(n_layers, max_seq_len, n_heads, d_head)
        self.v_cache = zeros(n_layers, max_seq_len, n_heads, d_head)
    
    # Memory: O(max_seq_len × n_layers × hidden_dim)
    # For 70B model: ~140GB for 32K context!
```

### Multi-Query Attention (MQA)
```python
# Share keys/values across heads
class MQACache:
    def __init__(self, max_seq_len, n_layers, d_model):
        # Single K,V per layer instead of per head
        self.k_cache = zeros(n_layers, max_seq_len, d_model)
        self.v_cache = zeros(n_layers, max_seq_len, d_model)
    
    # Memory: O(max_seq_len × n_layers × d_model / n_heads)
    # 8-32x memory reduction!
```

### Grouped-Query Attention (GQA)
Balance between quality and memory:
- Groups of 4-8 heads share K,V
- 4-8x memory reduction
- <1% quality loss

## 3. Model Quantization

### Full Precision (32-bit)
```python
# Standard weights
weight = torch.randn(4096, 4096, dtype=torch.float32)
# Memory: 64MB per layer
# Computation: Fast matmul
```

### INT8 Quantization
```python
# 8-bit weights with scale factors
weight_int8 = (weight * scale).round().clamp(-128, 127).to(torch.int8)
# Memory: 16MB per layer (4x reduction)
# Computation: Slightly slower, dequantize on the fly
```

### 4-bit Quantization (QLoRA)
```python
# Extreme quantization with adapters
weight_4bit = quantize_nf4(weight)  # 4-bit normal float
lora_A = torch.randn(4096, 16)      # Low-rank adapter
lora_B = torch.randn(16, 4096)

def forward(x):
    # Dequantize and compute
    base = dequantize(weight_4bit) @ x
    adapter = lora_B @ (lora_A @ x)
    return base + adapter

# Memory: 8MB base + 0.5MB adapter (8x reduction)
# Time: 2-3x slower due to dequantization
```

## 4. Checkpoint Strategies

### Gradient Checkpointing
```python
# Standard: Store all activations
def transformer_layer(x):
    attn = self.attention(x)      # Store activation
    ff = self.feedforward(attn)   # Store activation
    return ff

# With checkpointing: Recompute during backward
@checkpoint
def transformer_layer(x):
    attn = self.attention(x)      # Don't store
    ff = self.feedforward(attn)   # Don't store
    return ff

# Memory: O(√n_layers) instead of O(n_layers)
# Time: 30% slower training
```

## 5. Sparse Models

### Dense Model
- Every token processed by all parameters
- Memory: O(n_params)
- Time: O(n_tokens × n_params)

### Mixture of Experts (MoE)
```python
# Route to subset of experts
def moe_layer(x):
    router_logits = self.router(x)
    expert_ids = top_k(router_logits, k=2)
    
    output = 0
    for expert_id in expert_ids:
        output += self.experts[expert_id](x)
    
    return output

# Memory: Full model size
# Active memory: O(n_params / n_experts)
# Enables 10x larger models with same compute
```

## 6. Real-World Examples

### GPT-3 vs GPT-4
| Aspect | GPT-3 | GPT-4 |
|--------|-------|-------|
| Parameters | 175B | ~1.8T (MoE) |
| Context | 2K | 32K-128K |
| Techniques | Dense | MoE + Flash + GQA |
| Memory/token | ~350MB | ~50MB (active) |

### Llama 2 Family
```
Llama-2-7B:  Full precision = 28GB
             INT8 = 7GB
             INT4 = 3.5GB
             
Llama-2-70B: Full precision = 280GB
             INT8 = 70GB
             INT4 + QLoRA = 35GB (fits on single GPU!)
```

## 7. Serving Optimizations

### Continuous Batching
Instead of fixed batches, dynamically batch requests:
- Memory: Reuse KV-cache across requests
- Time: Higher throughput via better GPU utilization

### PagedAttention (vLLM)
```python
# Treat KV-cache like virtual memory
class PagedKVCache:
    def __init__(self, block_size=16):
        self.blocks = {}  # Allocated on demand
        self.page_table = {}  # Maps positions to blocks
    
    def allocate(self, seq_id, position):
        # Only allocate blocks as needed
        if position // self.block_size not in self.page_table[seq_id]:
            self.page_table[seq_id].append(new_block())
```

Memory fragmentation: <5% vs 60% for naive allocation

## 8. Training vs Inference Tradeoffs

### Training (Memory Intensive)
- Gradients: 2x model size
- Optimizer states: 2-3x model size
- Activations: O(batch × seq_len × layers)
- Total: 15-20x model parameters

### Inference (Can Trade Memory for Time)
- Only model weights needed
- Quantize aggressively
- Recompute instead of cache
- Stream weights from disk if needed

## Key Insights

1. **Every major LLM innovation** is a space-time tradeoff:
   - Flash Attention: Recompute for linear memory
   - Quantization: Dequantize for smaller models
   - MoE: Route for sparse activation

2. **The √n pattern appears everywhere**:
   - Gradient checkpointing: √n_layers memory
   - Block-wise attention: √seq_len blocks
   - Optimal batch sizes: Often √total_examples

3. **Practical systems combine multiple techniques**:
   - GPT-4: MoE + Flash + INT8 + GQA
   - Llama: Quantization + RoPE + GQA
   - Claude: Flash + Constitutional training

4. **Memory is the binding constraint**:
   - Not compute or data
   - Drives all architectural decisions
   - Williams' result predicts these optimizations

## Connection to Theory

Williams showed TIME[t] ⊆ SPACE[√(t log t)]. In LLMs:
- Standard attention: O(n²) space, O(n²) time
- Flash attention: O(n) space, O(n² log n) time
- The log factor comes from block coordination

This validates that the theoretical √t space bound manifests in practice, driving the most important optimizations in modern AI systems.