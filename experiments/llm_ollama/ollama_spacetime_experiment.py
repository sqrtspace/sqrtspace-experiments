#!/usr/bin/env python3
"""
LLM Space-Time Tradeoff Experiments using Ollama

Demonstrates real-world space-time tradeoffs in LLM inference:
1. Context window chunking (√n chunks)
2. Streaming vs full generation
3. Checkpointing for long generations
"""

import json
import time
import psutil
import requests
import numpy as np
from typing import List, Dict, Tuple
import argparse
import sys
import os

# Ollama API endpoint
OLLAMA_API = "http://localhost:11434/api"

def get_process_memory():
    """Get current process memory usage in MB"""
    return psutil.Process().memory_info().rss / 1024 / 1024

def generate_with_ollama(model: str, prompt: str, stream: bool = False) -> Tuple[str, float]:
    """Generate text using Ollama API"""
    url = f"{OLLAMA_API}/generate"
    data = {
        "model": model,
        "prompt": prompt,
        "stream": stream
    }
    
    start_time = time.time()
    response = requests.post(url, json=data, stream=stream)
    
    if stream:
        full_response = ""
        for line in response.iter_lines():
            if line:
                chunk = json.loads(line)
                if "response" in chunk:
                    full_response += chunk["response"]
        result = full_response
    else:
        result = response.json()["response"]
    
    elapsed = time.time() - start_time
    return result, elapsed

def chunked_context_processing(model: str, long_text: str, chunk_size: int) -> Dict:
    """Process long context in chunks vs all at once"""
    print(f"\n=== Chunked Context Processing ===")
    print(f"Total context length: {len(long_text)} chars")
    print(f"Chunk size: {chunk_size} chars")
    
    results = {}
    
    # Method 1: Process entire context at once
    print("\nMethod 1: Full context (O(n) memory)")
    prompt_full = f"Summarize the following text:\n\n{long_text}\n\nSummary:"
    
    mem_before = get_process_memory()
    summary_full, time_full = generate_with_ollama(model, prompt_full)
    mem_after = get_process_memory()
    
    results["full_context"] = {
        "time": time_full,
        "memory_delta": mem_after - mem_before,
        "summary_length": len(summary_full)
    }
    print(f"Time: {time_full:.2f}s, Memory delta: {mem_after - mem_before:.2f}MB")
    
    # Method 2: Process in √n chunks
    print(f"\nMethod 2: Chunked processing (O(√n) memory)")
    chunks = [long_text[i:i+chunk_size] for i in range(0, len(long_text), chunk_size)]
    chunk_summaries = []
    
    mem_before = get_process_memory()
    time_start = time.time()
    
    for i, chunk in enumerate(chunks):
        prompt_chunk = f"Summarize this text fragment:\n\n{chunk}\n\nSummary:"
        summary, _ = generate_with_ollama(model, prompt_chunk)
        chunk_summaries.append(summary)
        print(f"  Processed chunk {i+1}/{len(chunks)}")
    
    # Combine chunk summaries
    combined_prompt = f"Combine these summaries into one:\n\n" + "\n\n".join(chunk_summaries) + "\n\nCombined summary:"
    final_summary, _ = generate_with_ollama(model, combined_prompt)
    
    time_chunked = time.time() - time_start
    mem_after = get_process_memory()
    
    results["chunked_context"] = {
        "time": time_chunked,
        "memory_delta": mem_after - mem_before,
        "summary_length": len(final_summary),
        "num_chunks": len(chunks),
        "chunk_size": chunk_size
    }
    print(f"Time: {time_chunked:.2f}s, Memory delta: {mem_after - mem_before:.2f}MB")
    print(f"Slowdown: {time_chunked/time_full:.2f}x")
    
    return results

def streaming_vs_full_generation(model: str, prompt: str, num_tokens: int = 200) -> Dict:
    """Compare streaming vs full generation"""
    print(f"\n=== Streaming vs Full Generation ===")
    print(f"Generating ~{num_tokens} tokens")
    
    results = {}
    
    # Create a prompt that generates substantial output
    generation_prompt = prompt + "\n\nWrite a detailed explanation (at least 200 words):"
    
    # Method 1: Full generation (O(n) memory for response)
    print("\nMethod 1: Full generation")
    mem_before = get_process_memory()
    response_full, time_full = generate_with_ollama(model, generation_prompt, stream=False)
    mem_after = get_process_memory()
    
    results["full_generation"] = {
        "time": time_full,
        "memory_delta": mem_after - mem_before,
        "response_length": len(response_full),
        "estimated_tokens": len(response_full.split())
    }
    print(f"Time: {time_full:.2f}s, Memory delta: {mem_after - mem_before:.2f}MB")
    
    # Method 2: Streaming generation (O(1) memory)
    print("\nMethod 2: Streaming generation")
    mem_before = get_process_memory()
    response_stream, time_stream = generate_with_ollama(model, generation_prompt, stream=True)
    mem_after = get_process_memory()
    
    results["streaming_generation"] = {
        "time": time_stream,
        "memory_delta": mem_after - mem_before,
        "response_length": len(response_stream),
        "estimated_tokens": len(response_stream.split())
    }
    print(f"Time: {time_stream:.2f}s, Memory delta: {mem_after - mem_before:.2f}MB")
    
    return results

def checkpointed_generation(model: str, prompts: List[str], checkpoint_interval: int) -> Dict:
    """Simulate checkpointed generation for multiple prompts"""
    print(f"\n=== Checkpointed Generation ===")
    print(f"Processing {len(prompts)} prompts")
    print(f"Checkpoint interval: {checkpoint_interval}")
    
    results = {}
    
    # Method 1: Process all prompts without checkpointing
    print("\nMethod 1: No checkpointing")
    responses_full = []
    mem_before = get_process_memory()
    time_start = time.time()
    
    for i, prompt in enumerate(prompts):
        response, _ = generate_with_ollama(model, prompt)
        responses_full.append(response)
        print(f"  Processed prompt {i+1}/{len(prompts)}")
    
    time_full = time.time() - time_start
    mem_after = get_process_memory()
    
    results["no_checkpoint"] = {
        "time": time_full,
        "memory_delta": mem_after - mem_before,
        "total_responses": len(responses_full),
        "avg_response_length": np.mean([len(r) for r in responses_full])
    }
    
    # Method 2: Process with checkpointing (simulate by clearing responses)
    print(f"\nMethod 2: Checkpointing every {checkpoint_interval} prompts")
    responses_checkpoint = []
    checkpoint_data = []
    mem_before = get_process_memory()
    time_start = time.time()
    
    for i, prompt in enumerate(prompts):
        response, _ = generate_with_ollama(model, prompt)
        responses_checkpoint.append(response)
        
        # Simulate checkpoint: save and clear memory
        if (i + 1) % checkpoint_interval == 0:
            checkpoint_data.append({
                "index": i,
                "responses": responses_checkpoint.copy()
            })
            responses_checkpoint = []  # Clear to save memory
            print(f"  Checkpoint at prompt {i+1}")
        else:
            print(f"  Processed prompt {i+1}/{len(prompts)}")
    
    # Final checkpoint for remaining
    if responses_checkpoint:
        checkpoint_data.append({
            "index": len(prompts) - 1,
            "responses": responses_checkpoint
        })
    
    time_checkpoint = time.time() - time_start
    mem_after = get_process_memory()
    
    # Reconstruct all responses from checkpoints
    all_responses = []
    for checkpoint in checkpoint_data:
        all_responses.extend(checkpoint["responses"])
    
    results["with_checkpoint"] = {
        "time": time_checkpoint,
        "memory_delta": mem_after - mem_before,
        "total_responses": len(all_responses),
        "avg_response_length": np.mean([len(r) for r in all_responses]),
        "num_checkpoints": len(checkpoint_data),
        "checkpoint_interval": checkpoint_interval
    }
    
    print(f"\nTime comparison:")
    print(f"  No checkpoint: {time_full:.2f}s")
    print(f"  With checkpoint: {time_checkpoint:.2f}s")
    print(f"  Overhead: {(time_checkpoint/time_full - 1)*100:.1f}%")
    
    return results

def run_all_experiments(model: str = "llama3.2:latest"):
    """Run all space-time tradeoff experiments"""
    print(f"Using model: {model}")
    
    # Check if model is available
    try:
        test_response = requests.post(f"{OLLAMA_API}/generate", 
                                     json={"model": model, "prompt": "test", "stream": False})
        if test_response.status_code != 200:
            print(f"Error: Model {model} not available. Please pull it first with: ollama pull {model}")
            return
    except:
        print("Error: Cannot connect to Ollama. Make sure it's running with: ollama serve")
        return
    
    all_results = {
        "model": model,
        "timestamp": time.strftime("%Y-%m-%d %H:%M:%S"),
        "experiments": {}
    }
    
    # Experiment 1: Context chunking
    # Create a long text by repeating a passage
    base_text = """The quick brown fox jumps over the lazy dog. This pangram contains every letter of the alphabet.
    It has been used for decades to test typewriters and computer keyboards. The sentence is memorable and 
    helps identify any malfunctioning keys. Many variations exist in different languages."""
    
    long_text = (base_text + " ") * 50  # ~10KB of text
    chunk_size = int(np.sqrt(len(long_text)))  # √n chunk size
    
    context_results = chunked_context_processing(model, long_text, chunk_size)
    all_results["experiments"]["context_chunking"] = context_results
    
    # Experiment 2: Streaming vs full generation
    prompt = "Explain the concept of space-time tradeoffs in computer science."
    streaming_results = streaming_vs_full_generation(model, prompt)
    all_results["experiments"]["streaming"] = streaming_results
    
    # Experiment 3: Checkpointed generation
    prompts = [
        "What is machine learning?",
        "Explain neural networks.",
        "What is deep learning?",
        "Describe transformer models.",
        "What is attention mechanism?",
        "Explain BERT architecture.",
        "What is GPT?",
        "Describe fine-tuning.",
        "What is transfer learning?",
        "Explain few-shot learning."
    ]
    checkpoint_interval = int(np.sqrt(len(prompts)))  # √n checkpoint interval
    
    checkpoint_results = checkpointed_generation(model, prompts, checkpoint_interval)
    all_results["experiments"]["checkpointing"] = checkpoint_results
    
    # Save results
    with open("ollama_experiment_results.json", "w") as f:
        json.dump(all_results, f, indent=2)
    
    print("\n=== Summary ===")
    print(f"Results saved to ollama_experiment_results.json")
    
    # Print summary
    print("\n1. Context Chunking:")
    if "context_chunking" in all_results["experiments"]:
        full = all_results["experiments"]["context_chunking"]["full_context"]
        chunked = all_results["experiments"]["context_chunking"]["chunked_context"]
        print(f"   Full context: {full['time']:.2f}s, {full['memory_delta']:.2f}MB")
        print(f"   Chunked (√n): {chunked['time']:.2f}s, {chunked['memory_delta']:.2f}MB")
        print(f"   Slowdown: {chunked['time']/full['time']:.2f}x")
        print(f"   Memory reduction: {(1 - chunked['memory_delta']/max(full['memory_delta'], 0.1))*100:.1f}%")
    
    print("\n2. Streaming Generation:")
    if "streaming" in all_results["experiments"]:
        full = all_results["experiments"]["streaming"]["full_generation"]
        stream = all_results["experiments"]["streaming"]["streaming_generation"]
        print(f"   Full generation: {full['time']:.2f}s, {full['memory_delta']:.2f}MB")
        print(f"   Streaming: {stream['time']:.2f}s, {stream['memory_delta']:.2f}MB")
    
    print("\n3. Checkpointing:")
    if "checkpointing" in all_results["experiments"]:
        no_ckpt = all_results["experiments"]["checkpointing"]["no_checkpoint"]
        with_ckpt = all_results["experiments"]["checkpointing"]["with_checkpoint"]
        print(f"   No checkpoint: {no_ckpt['time']:.2f}s, {no_ckpt['memory_delta']:.2f}MB")
        print(f"   With checkpoint: {with_ckpt['time']:.2f}s, {with_ckpt['memory_delta']:.2f}MB")
        print(f"   Time overhead: {(with_ckpt['time']/no_ckpt['time'] - 1)*100:.1f}%")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="LLM Space-Time Tradeoff Experiments")
    parser.add_argument("--model", default="llama3.2:latest", help="Ollama model to use")
    parser.add_argument("--experiment", choices=["all", "context", "streaming", "checkpoint"], 
                       default="all", help="Which experiment to run")
    
    args = parser.parse_args()
    
    if args.experiment == "all":
        run_all_experiments(args.model)
    else:
        print(f"Running {args.experiment} experiment with {args.model}")
        # Run specific experiment
        if args.experiment == "context":
            base_text = "The quick brown fox jumps over the lazy dog. " * 100
            results = chunked_context_processing(args.model, base_text, int(np.sqrt(len(base_text))))
        elif args.experiment == "streaming":
            results = streaming_vs_full_generation(args.model, "Explain AI in detail.")
        elif args.experiment == "checkpoint":
            prompts = [f"Explain concept {i}" for i in range(10)]
            results = checkpointed_generation(args.model, prompts, 3)
        
        print(f"\nResults: {json.dumps(results, indent=2)}")