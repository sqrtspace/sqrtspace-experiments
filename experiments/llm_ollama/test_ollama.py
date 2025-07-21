#!/usr/bin/env python3
"""Quick test to verify Ollama is working"""

import requests
import json

def test_ollama():
    """Test Ollama connection"""
    try:
        # Test API endpoint
        response = requests.get("http://localhost:11434/api/tags")
        if response.status_code == 200:
            models = response.json()
            print("✓ Ollama is running")
            print(f"✓ Found {len(models['models'])} models:")
            for model in models['models'][:5]:  # Show first 5
                print(f"  - {model['name']} ({model['size']//1e9:.1f}GB)")
            return True
        else:
            print("✗ Ollama API not responding correctly")
            return False
    except requests.exceptions.ConnectionError:
        print("✗ Cannot connect to Ollama. Make sure it's running with: ollama serve")
        return False
    except Exception as e:
        print(f"✗ Error: {e}")
        return False

def test_generation():
    """Test model generation"""
    model = "llama3.2:latest"
    print(f"\nTesting generation with {model}...")
    
    try:
        response = requests.post(
            "http://localhost:11434/api/generate",
            json={
                "model": model,
                "prompt": "Say hello in 5 words or less",
                "stream": False
            }
        )
        
        if response.status_code == 200:
            result = response.json()
            print(f"✓ Generation successful: {result['response'].strip()}")
            return True
        else:
            print(f"✗ Generation failed: {response.status_code}")
            return False
    except Exception as e:
        print(f"✗ Generation error: {e}")
        return False

if __name__ == "__main__":
    print("Testing Ollama setup...")
    if test_ollama() and test_generation():
        print("\n✓ All tests passed! Ready to run experiments.")
        print("\nRun the main experiment with:")
        print("  python ollama_spacetime_experiment.py")
    else:
        print("\n✗ Please fix the issues above before running experiments.")