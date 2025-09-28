#!/usr/bin/env python3
"""
Creative-Sparse Example: Demonstrating the two-stage algorithm for creative generation

This example shows how to use the Creative-Sparse framework for generating
creative content with optimal sparsity control.
"""

import torch
import argparse
from typing import List
import sys
import os

# Add parent directory to path for imports
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from spas_sage_attn import (
    CreativeSparseWrapper,
    SparsityConfig,
    QualityWeights,
    find_optimal_sparsity,
    compute_quality_score,
    spas_sage2_attn_meansim_cuda
)


def demonstrate_sparse_attention():
    """Demonstrate basic sparse attention usage"""
    print("=== Sparse Attention Demonstration ===")
    
    # Create sample attention tensors
    batch_size, num_heads, seq_len, head_dim = 1, 8, 512, 64
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    
    q = torch.randn(batch_size, num_heads, seq_len, head_dim, dtype=torch.float16, device=device)
    k = torch.randn(batch_size, num_heads, seq_len, head_dim, dtype=torch.float16, device=device)
    v = torch.randn(batch_size, num_heads, seq_len, head_dim, dtype=torch.float16, device=device)
    
    print(f"Input shapes: Q={q.shape}, K={k.shape}, V={v.shape}")
    
    # Apply sparse attention with different sparsity levels
    sparsity_levels = [0.3, 0.5, 0.7, 0.9]
    
    for sparsity in sparsity_levels:
        try:
            # Adjust parameters based on sparsity
            simthreshd1 = 0.6 * (1 - sparsity * 0.3)
            cdfthreshd = 0.97 * (1 - sparsity * 0.2)
            pvthreshd = max(5, int(15 * (1 - sparsity * 0.4)))
            
            output = spas_sage2_attn_meansim_cuda(
                q, k, v,
                simthreshd1=simthreshd1,
                cdfthreshd=cdfthreshd,
                pvthreshd=pvthreshd,
                is_causal=False,
                return_sparsity=True
            )
            
            if isinstance(output, tuple):
                attn_output, actual_sparsity = output
                print(f"Sparsity {sparsity:.1f}: Actual sparsity = {actual_sparsity:.3f}")
            else:
                print(f"Sparsity {sparsity:.1f}: Sparse attention applied successfully")
                
        except Exception as e:
            print(f"Sparsity {sparsity:.1f}: Error - {e}")


def demonstrate_creative_generation():
    """Demonstrate Creative-Sparse generation"""
    print("\n=== Creative Generation Demonstration ===")
    
    # Initialize Creative-Sparse wrapper
    wrapper = CreativeSparseWrapper(
        model_name="gpt2-large",
        sparsity_schedule=(0.7, 0.3),  # (divergence, refinement)
        quality_weights=QualityWeights(alpha=0.4, beta=0.3, gamma=0.3)
    )
    
    # Creative prompts
    prompts = [
        "Write a creative story about a robot learning to paint",
        "Describe a world where colors have sounds",
        "Create a poem about digital dreams",
        "Imagine a conversation between a tree and a cloud"
    ]
    
    for i, prompt in enumerate(prompts):
        print(f"\n--- Prompt {i+1}: {prompt} ---")
        
        try:
            # Generate creative content
            output = wrapper.generate_creative(
                prompt=prompt,
                max_length=150,
                num_candidates=3,
                return_best=True
            )
            
            print(f"Generated: {output}")
            
            # Compute quality score
            quality_score = compute_quality_score([output])
            print(f"Quality Score: {quality_score[0]:.3f}")
            
        except Exception as e:
            print(f"Error generating content: {e}")


def demonstrate_sparsity_optimization():
    """Demonstrate sparsity optimization"""
    print("\n=== Sparsity Optimization Demonstration ===")
    
    # Different task types and their optimal sparsities
    tasks = [
        ("creative_writing", 0.8),
        ("image_generation", 0.6),
        ("video_generation", 0.7),
        ("code_generation", 0.5)
    ]
    
    for task_type, compute_budget in tasks:
        optimal_sparsity = find_optimal_sparsity(
            model_name="gpt2-large",
            task_type=task_type,
            compute_budget=compute_budget
        )
        
        print(f"Task: {task_type}, Compute Budget: {compute_budget}")
        print(f"Optimal Sparsity: {optimal_sparsity:.3f}")


def demonstrate_quality_assessment():
    """Demonstrate quality assessment"""
    print("\n=== Quality Assessment Demonstration ===")
    
    # Sample outputs with different quality levels
    outputs = [
        "The robot painted a beautiful sunset with vibrant colors.",
        "Color sounds like music, each hue a different note.",
        "Digital dreams dance in binary streams of light.",
        "The tree whispered secrets to the wandering cloud above."
    ]
    
    # Reference data for novelty computation
    reference_data = [
        "A simple story about a robot.",
        "Basic description of colors.",
        "Short poem about technology.",
        "Simple nature conversation."
    ]
    
    # Compute quality scores
    scores = compute_quality_score(
        outputs=outputs,
        reference_data=reference_data,
        novelty_weight=0.4,
        diversity_weight=0.3,
        coherence_weight=0.3
    )
    
    print("Quality Assessment Results:")
    for i, (output, score) in enumerate(zip(outputs, scores)):
        print(f"{i+1}. Score: {score:.3f} - {output[:50]}...")


def demonstrate_sparsity_controller():
    """Demonstrate sparsity controller functionality"""
    print("\n=== Sparsity Controller Demonstration ===")
    
    from spas_sage_attn.creative_sparse import SparsityController
    
    config = SparsityConfig()
    controller = SparsityController(config)
    
    sparsity_levels = [0.2, 0.4, 0.6, 0.8]
    
    print("Sparsity Level | Temperature | Nucleus | Experts | Attention Scale")
    print("-" * 65)
    
    for sparsity in sparsity_levels:
        params = controller.get_sparsity_params(sparsity)
        print(f"{sparsity:13.1f} | {params['temperature']:10.3f} | "
              f"{params['nucleus_threshold']:7.3f} | {params['expert_activation']:7d} | "
              f"{params['attention_scaling']:14.3f}")


def main():
    parser = argparse.ArgumentParser(description="Creative-Sparse Example")
    parser.add_argument("--demo", type=str, default="all", 
                       choices=["all", "attention", "generation", "optimization", "quality", "controller"],
                       help="Which demonstration to run")
    parser.add_argument("--device", type=str, default="auto",
                       choices=["auto", "cuda", "cpu"],
                       help="Device to use")
    
    args = parser.parse_args()
    
    # Set device
    if args.device == "auto":
        device = "cuda" if torch.cuda.is_available() else "cpu"
    else:
        device = args.device
    
    print(f"Using device: {device}")
    print("Creative-Sparse: A Two-Stage Algorithm for Controllable Creative AI")
    print("=" * 70)
    
    if args.demo == "all" or args.demo == "attention":
        demonstrate_sparse_attention()
    
    if args.demo == "all" or args.demo == "generation":
        demonstrate_creative_generation()
    
    if args.demo == "all" or args.demo == "optimization":
        demonstrate_sparsity_optimization()
    
    if args.demo == "all" or args.demo == "quality":
        demonstrate_quality_assessment()
    
    if args.demo == "all" or args.demo == "controller":
        demonstrate_sparsity_controller()
    
    print("\n" + "=" * 70)
    print("Demonstration completed!")


if __name__ == "__main__":
    main()
