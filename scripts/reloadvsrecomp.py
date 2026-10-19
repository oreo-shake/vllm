"""
python vllm_load_test.py \
    --model "poolside/Laguna-XS-2.1" \
    --input-lens 1024 2048 4096 8192 \
    --output-len 1 \
    --endpoint "http://127.0.0.1:8000/v1/completions"

"""

import argparse
import json
import time
import random
import uuid
import requests
from transformers import AutoTokenizer

def generate_exact_length_token_ids(tokenizer, target_length):
    """
    Generates an array of token IDs with exactly `target_length` tokens.
    Uses deterministic seeding based on the target_length so that prompts
    of the same length are identical across runs, but different lengths 
    have completely distinct prefixes.
    """
    text_blocks = [
        "The quick brown fox jumps over the lazy dog. ",
        "Artificial intelligence is rapidly transforming global infrastructure. ",
        "Quantum computing leverages the principles of quantum mechanics. ",
        "Photosynthesis is a vital process used by plants and other organisms. ",
        "The Great Wall of China is a historical series of fortifications. ",
        "In computer science, an algorithm is a finite sequence of instructions. ",
        "The history of aviation spans over more than two thousand years. ",
        "Music is the expressive art of arranging sounds in time. ",
        "A black hole is a region of spacetime where gravity is absolute. ",
        "Deep within the ocean, extraordinary creatures adapt to extreme pressure. ",
        "Cybersecurity measures are essential for protecting modern digital assets. ",
        "The Renaissance was a fervent period of European cultural, artistic, and economic rebirth. "
    ]
    
    # Isolate the random number generator and seed it with the target_length
    rng = random.Random(target_length)
    
    tokens = []
    
    if tokenizer.bos_token_id is not None:
        tokens.append(tokenizer.bos_token_id)
        
    # Generate a deterministic UUID based on the length to ensure different lengths 
    # don't share prefixes, but the same length always gets the same prefix.
    length_namespace = uuid.NAMESPACE_OID
    deterministic_uuid = uuid.uuid5(length_namespace, f"benchmark_{target_length}")
    
    unique_prefix_str = f"CACHE_BUSTER_{deterministic_uuid.hex} | "
    unique_prefix_tokens = tokenizer.encode(unique_prefix_str, add_special_tokens=False)
    tokens.extend(unique_prefix_tokens)
        
    while len(tokens) < target_length:
        # Use the seeded generator to shuffle, ensuring repeatable patterns for this length
        rng.shuffle(text_blocks)
        chunk_text = "".join(text_blocks)
        chunk_tokens = tokenizer.encode(chunk_text, add_special_tokens=False)
        tokens.extend(chunk_tokens)
        
    exact_tokens = tokens[:target_length]
    return exact_tokens

def test_vllm_endpoint(endpoint, model, prompt_ids, max_tokens, token_length):
    headers = {"Content-Type": "application/json"}
    payload = {
        "model": model,
        "prompt": prompt_ids, 
        "max_tokens": max_tokens,
        "temperature": 0.0,  
        "ignore_eos": True,  
        "stream": True       
    }
    
    print(f"[*] Sending request to {endpoint}...")
    
    ttft = None
    start_time = time.perf_counter()
    chunks_received = 0
    
    try:
        response = requests.post(endpoint, json=payload, headers=headers, stream=True)
        response.raise_for_status()
        
        for line in response.iter_lines():
            if not line:
                continue
                
            line_str = line.decode("utf-8")
            if line_str.startswith("data: "):
                data_str = line_str[6:].strip()
                if data_str == "[DONE]":
                    break
                
                try:
                    chunk = json.loads(data_str)
                    choices = chunk.get("choices", [])
                    if choices:
                        text_delta = choices[0].get("text", "")
                        
                        if text_delta and ttft is None:
                            ttft = time.perf_counter() - start_time
                        
                        if text_delta:
                            chunks_received += 1
                except json.JSONDecodeError:
                    continue

        end_time = time.perf_counter()
        total_latency = end_time - start_time
        
        print(f"    [+] Request Completed.")
        if ttft is not None:
            print(f"    [+] Time To First Token (TTFT): {ttft * 1000:.2f} ms ({ttft:.4f} s)")
        else:
            print(f"    [-] Could not calculate TTFT (no token deltas received).")
            
        generation_time = total_latency - (ttft if ttft else 0)
        print(f"    [+] Total Request Latency:     {total_latency:.2f} s")
        if generation_time > 0 and chunks_received > 1:
            decode_throughput = (chunks_received - 1) / generation_time
            print(f"    [+] Decode Throughput:         {decode_throughput:.2f} tokens/sec")
            
    except requests.exceptions.RequestException as e:
        print(f"    [-] Request failed: {e}")
        if 'response' in locals() and response is not None:
            print(f"    [-] Server Response: {response.text}")

def main():
    parser = argparse.ArgumentParser(description="Deterministic vLLM benchmarking script.")
    parser.add_argument("--model", type=str, required=True, 
                        help="HuggingFace model ID")
    parser.add_argument("--input-lens", type=int, nargs='+', required=True, 
                        help="List of prompt lengths to test")
    parser.add_argument("--endpoint", type=str, default="http://localhost:8000/v1/completions",
                        help="vLLM server endpoint")
    parser.add_argument("--output-len", type=int, default=64,
                        help="Number of output tokens to generate")
    
    args = parser.parse_args()

    print(f"Loading tokenizer for model '{args.model}'...")
    tokenizer = AutoTokenizer.from_pretrained(args.model, trust_remote_code=True)
    
    print("\nStarting sequential benchmarks (Deterministic Inputs)...")
    print("=" * 70)
    
    for idx, length in enumerate(args.input_lens, 1):
        print(f"\n--- Test {idx}/{len(args.input_lens)}: Input Length {length} tokens ---")
        
        prompt_ids = generate_exact_length_token_ids(tokenizer, length)
        actual_length = len(prompt_ids)
        
        prefix_preview = tokenizer.decode(prompt_ids[:25]).replace('\n', ' ')
        print(f"[!] Target tokens: {length} | Actual array length: {actual_length}")
        print(f"[!] Deterministic Prefix: \"{prefix_preview}...\"")
        
        test_vllm_endpoint(args.endpoint, args.model, prompt_ids, args.output_len, actual_length)
        
        time.sleep(1)

if __name__ == "__main__":
    main()
