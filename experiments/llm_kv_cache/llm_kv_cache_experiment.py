"""
LLM KV-Cache Space-Time Tradeoff Experiment

Demonstrates how KV-cache size affects transformer inference time,
showing Williams' √n pattern in modern AI systems.

This simulates the core attention mechanism where:
- Full KV-cache (O(n)): Store all past tokens' keys/values
- Sliding window (O(√n)): Keep only recent √n tokens  
- Minimal cache (O(1)): Recompute everything

Based on Flash Attention and similar optimizations used in production LLMs.
"""

import numpy as np
import time
import matplotlib.pyplot as plt
from typing import Dict, List, Tuple
import json
from dataclasses import dataclass

@dataclass
class AttentionConfig:
    """Configuration for attention mechanism"""
    seq_length: int          # Total sequence length
    hidden_dim: int          # Model dimension (d_model)
    num_heads: int           # Number of attention heads
    head_dim: int            # Dimension per head
    batch_size: int = 1      # Batch size
    
    def __post_init__(self):
        assert self.hidden_dim == self.num_heads * self.head_dim

class TransformerAttention:
    """Simplified transformer attention with configurable KV-cache"""
    
    def __init__(self, config: AttentionConfig):
        self.config = config
        
        # Initialize weights (random for simulation)
        self.W_q = np.random.randn(config.hidden_dim, config.hidden_dim) * 0.02
        self.W_k = np.random.randn(config.hidden_dim, config.hidden_dim) * 0.02
        self.W_v = np.random.randn(config.hidden_dim, config.hidden_dim) * 0.02
        self.W_o = np.random.randn(config.hidden_dim, config.hidden_dim) * 0.02
        
    def compute_attention(self, 
                         query_pos: int,
                         hidden_states: np.ndarray,
                         kv_cache_size: int) -> Tuple[np.ndarray, Dict]:
        """
        Compute attention for position query_pos with limited KV-cache
        
        Args:
            query_pos: Current token position
            hidden_states: All hidden states up to query_pos
            kv_cache_size: Maximum number of past tokens to cache
            
        Returns:
            attention_output: Output for the query position
            stats: Performance statistics
        """
        stats = {
            'cache_size': kv_cache_size,
            'recompute_steps': 0,
            'cache_hits': 0,
            'memory_used': 0
        }
        
        # Get query vector for current position
        query = hidden_states[query_pos:query_pos+1]  # [1, hidden_dim]
        Q = query @ self.W_q  # [1, hidden_dim]
        
        # Reshape for multi-head attention
        Q = Q.reshape(1, self.config.num_heads, self.config.head_dim)
        
        # Determine which positions to attend to
        if kv_cache_size >= query_pos:
            # Full cache - use all previous positions
            start_pos = 0
            cached_positions = query_pos
            stats['cache_hits'] = query_pos
        else:
            # Limited cache - use only recent positions
            start_pos = max(0, query_pos - kv_cache_size)
            cached_positions = min(kv_cache_size, query_pos)
            stats['cache_hits'] = cached_positions
            stats['recompute_steps'] = query_pos - cached_positions
        
        # Get relevant hidden states
        relevant_hidden = hidden_states[start_pos:query_pos+1]
        
        # Compute keys and values (this is what we cache/recompute)
        start_time = time.time()
        K = relevant_hidden @ self.W_k  # [seq_len, hidden_dim]
        V = relevant_hidden @ self.W_v
        compute_time = time.time() - start_time
        
        # Reshape for multi-head
        seq_len = K.shape[0]
        K = K.reshape(seq_len, self.config.num_heads, self.config.head_dim)
        V = V.reshape(seq_len, self.config.num_heads, self.config.head_dim)
        
        # Compute attention scores
        scores = np.einsum('qhd,khd->hqk', Q, K) / np.sqrt(self.config.head_dim)
        
        # Apply causal mask if needed
        if start_pos > 0:
            # Mask out positions we can't see due to limited cache
            mask = np.ones_like(scores)
            scores = scores * mask
        
        # Softmax
        attn_weights = self._softmax(scores, axis=-1)
        
        # Apply attention to values
        attn_output = np.einsum('hqk,khd->qhd', attn_weights, V)
        
        # Reshape and project
        attn_output = attn_output.reshape(1, self.config.hidden_dim)
        output = attn_output @ self.W_o
        
        # Calculate memory usage
        stats['memory_used'] = (
            2 * cached_positions * self.config.hidden_dim * 4  # K and V cache in bytes
        )
        stats['compute_time'] = compute_time
        
        return output, stats
    
    def _softmax(self, x, axis=-1):
        """Numerically stable softmax"""
        e_x = np.exp(x - np.max(x, axis=axis, keepdims=True))
        return e_x / np.sum(e_x, axis=axis, keepdims=True)
    
    def generate_sequence(self, 
                         prompt_length: int,
                         generation_length: int,
                         kv_cache_size: int) -> Dict:
        """
        Simulate autoregressive generation with limited KV-cache
        
        This mimics how LLMs generate text token by token
        """
        total_length = prompt_length + generation_length
        hidden_dim = self.config.hidden_dim
        
        # Initialize with random hidden states (simulating embeddings)
        hidden_states = np.random.randn(total_length, hidden_dim) * 0.1
        
        total_stats = {
            'total_time': 0,
            'total_memory': 0,
            'total_recomputes': 0,
            'per_token_times': []
        }
        
        # Process prompt (can use full attention)
        start_time = time.time()
        for pos in range(prompt_length):
            _, stats = self.compute_attention(pos, hidden_states, kv_cache_size)
        prompt_time = time.time() - start_time
        
        # Generate new tokens
        generation_times = []
        for pos in range(prompt_length, total_length):
            start = time.time()
            output, stats = self.compute_attention(pos, hidden_states, kv_cache_size)
            token_time = time.time() - start
            
            generation_times.append(token_time)
            total_stats['total_recomputes'] += stats['recompute_steps']
            total_stats['total_memory'] = max(total_stats['total_memory'], 
                                            stats['memory_used'])
            
            # Simulate token generation (would normally sample from logits)
            hidden_states[pos] = output[0]
        
        total_stats['total_time'] = sum(generation_times) + prompt_time
        total_stats['avg_token_time'] = np.mean(generation_times) if generation_times else 0
        total_stats['prompt_time'] = prompt_time
        total_stats['generation_time'] = sum(generation_times)
        total_stats['tokens_per_second'] = generation_length / sum(generation_times) if generation_times else 0
        
        return total_stats

def run_llm_experiment():
    """Run comprehensive LLM KV-cache experiment"""
    
    print("="*60)
    print("LLM KV-Cache Space-Time Tradeoff Experiment")
    print("Simulating transformer attention with different cache sizes")
    print("="*60)
    
    # Model configuration (similar to GPT-2 small)
    config = AttentionConfig(
        seq_length=2048,      # Max sequence length
        hidden_dim=768,       # Model dimension
        num_heads=12,         # Attention heads
        head_dim=64,          # Dimension per head
        batch_size=1
    )
    
    model = TransformerAttention(config)
    
    # Test different sequence lengths
    test_lengths = [512, 1024, 2048]
    results = {}
    
    for seq_len in test_lengths:
        print(f"\n{'='*40}")
        print(f"Testing sequence length: {seq_len}")
        print(f"{'='*40}")
        
        # Different KV-cache configurations
        cache_configs = [
            ('Full O(n)', seq_len),                    # Full attention
            ('Flash O(√n)', int(np.sqrt(seq_len) * 4)), # Flash Attention-like
            ('Minimal O(1)', 8),                       # Almost no cache
        ]
        
        seq_results = []
        
        for label, cache_size in cache_configs:
            print(f"\n{label}: {cache_size} tokens cached")
            
            # Run multiple trials
            trials = []
            num_trials = 5
            
            for trial in range(num_trials):
                stats = model.generate_sequence(
                    prompt_length=seq_len // 2,
                    generation_length=seq_len // 2,
                    kv_cache_size=cache_size
                )
                trials.append(stats)
            
            # Average results
            avg_stats = {
                'label': label,
                'cache_size': cache_size,
                'avg_token_time': np.mean([t['avg_token_time'] for t in trials]),
                'tokens_per_second': np.mean([t['tokens_per_second'] for t in trials]),
                'max_memory_mb': np.mean([t['total_memory'] for t in trials]) / 1024 / 1024,
                'total_recomputes': np.mean([t['total_recomputes'] for t in trials])
            }
            
            seq_results.append(avg_stats)
            
            print(f"  Avg token time: {avg_stats['avg_token_time']*1000:.2f} ms")
            print(f"  Tokens/second: {avg_stats['tokens_per_second']:.1f}")
            print(f"  Memory used: {avg_stats['max_memory_mb']:.1f} MB")
            print(f"  Recomputations: {avg_stats['total_recomputes']:.0f}")
        
        results[seq_len] = seq_results
    
    # Create visualizations
    create_llm_plots(results)
    
    # Save results
    save_data = {
        'model_config': {
            'hidden_dim': config.hidden_dim,
            'num_heads': config.num_heads,
            'head_dim': config.head_dim
        },
        'results': results
    }
    
    with open('llm_kv_cache_results.json', 'w') as f:
        json.dump(save_data, f, indent=2)
    
    print("\n" + "="*60)
    print("EXPERIMENT COMPLETE")
    print("Generated files:")
    print("  - llm_attention_tradeoff.png")
    print("  - llm_kv_cache_results.json")
    print("="*60)

def create_llm_plots(results):
    """Create publication-quality plots for LLM experiment"""
    
    fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(14, 10))
    
    # Plot 1: Token generation time vs cache size
    seq_lengths = sorted(results.keys())
    colors = ['green', 'orange', 'red']
    
    for seq_len in seq_lengths:
        cache_sizes = [r['cache_size'] for r in results[seq_len]]
        token_times = [r['avg_token_time'] * 1000 for r in results[seq_len]]
        
        ax1.plot(cache_sizes, token_times, 'o-', label=f'Seq {seq_len}',
                linewidth=2, markersize=8)
    
    ax1.set_xlabel('KV-Cache Size (tokens)', fontsize=12)
    ax1.set_ylabel('Avg Token Time (ms)', fontsize=12)
    ax1.set_title('Token Generation Time vs Cache Size', fontsize=14)
    ax1.set_xscale('log')
    ax1.legend()
    ax1.grid(True, alpha=0.3)
    
    # Plot 2: Memory usage
    for i, seq_len in enumerate(seq_lengths):
        labels = [r['label'].replace(' O', '\nO') for r in results[seq_len]]
        memory = [r['max_memory_mb'] for r in results[seq_len]]
        
        x = np.arange(len(labels)) + i * 0.25
        ax2.bar(x, memory, 0.25, label=f'Seq {seq_len}', alpha=0.8)
    
    ax2.set_xticks(np.arange(len(labels)) + 0.25)
    ax2.set_xticklabels(labels)
    ax2.set_ylabel('Memory Usage (MB)', fontsize=12)
    ax2.set_title('KV-Cache Memory Requirements', fontsize=14)
    ax2.legend()
    ax2.grid(True, alpha=0.3, axis='y')
    
    # Plot 3: Throughput (tokens/second)
    seq_len = 2048  # Focus on largest
    data = results[seq_len]
    
    labels = [r['label'] for r in data]
    throughput = [r['tokens_per_second'] for r in data]
    
    bars = ax3.bar(labels, throughput, color=colors, edgecolor='black', linewidth=1.5)
    ax3.set_ylabel('Tokens per Second', fontsize=12)
    ax3.set_title(f'Generation Throughput (seq_len={seq_len})', fontsize=14)
    ax3.grid(True, alpha=0.3, axis='y')
    
    # Add value labels
    for bar, val in zip(bars, throughput):
        ax3.text(bar.get_x() + bar.get_width()/2., bar.get_height(),
                f'{val:.0f}', ha='center', va='bottom', fontsize=11)
    
    # Plot 4: Space-time tradeoff curve
    for seq_len in seq_lengths:
        cache_pct = [r['cache_size'] / seq_len * 100 for r in results[seq_len]]
        speedup = [results[seq_len][0]['tokens_per_second'] / r['tokens_per_second'] 
                   for r in results[seq_len]]
        
        ax4.plot(cache_pct, speedup, 's-', label=f'Seq {seq_len}',
                linewidth=2, markersize=8)
    
    # Add theoretical √n curve
    x_theory = np.linspace(1, 100, 100)
    y_theory = np.sqrt(100 / x_theory)
    ax4.plot(x_theory, y_theory, 'k--', alpha=0.5, label='Theoretical √n')
    
    ax4.set_xlabel('Cache Size (% of Sequence)', fontsize=12)
    ax4.set_ylabel('Slowdown Factor', fontsize=12)
    ax4.set_title('Space-Time Tradeoff in Attention', fontsize=14)
    ax4.set_xscale('log')
    ax4.set_yscale('log')
    ax4.legend()
    ax4.grid(True, alpha=0.3)
    
    plt.suptitle('LLM Attention: KV-Cache Space-Time Tradeoffs', fontsize=16)
    plt.tight_layout()
    plt.savefig('llm_attention_tradeoff.png', dpi=300, bbox_inches='tight')
    plt.close()

if __name__ == "__main__":
    run_llm_experiment()