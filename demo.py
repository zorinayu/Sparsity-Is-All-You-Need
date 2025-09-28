#!/usr/bin/env python3
"""
Simple demonstration of Creative-Sparse functionality
"""

import torch
import sys
import os

# Add current directory to path
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

def demo_sparse_attention():
    """Demonstrate basic sparse attention"""
    print("🔍 Demonstrating Sparse Attention")
    print("-" * 40)
    
    try:
        from spas_sage_attn import spas_sage2_attn_meansim_cuda
        
        # Create sample tensors
        device = 'cuda' if torch.cuda.is_available() else 'cpu'
        print(f"Using device: {device}")
        
        batch_size, num_heads, seq_len, head_dim = 1, 8, 512, 64
        q = torch.randn(batch_size, num_heads, seq_len, head_dim, dtype=torch.float16, device=device)
        k = torch.randn(batch_size, num_heads, seq_len, head_dim, dtype=torch.float16, device=device)
        v = torch.randn(batch_size, num_heads, seq_len, head_dim, dtype=torch.float16, device=device)
        
        print(f"Input shapes: Q={q.shape}, K={k.shape}, V={v.shape}")
        
        # Apply sparse attention
        output = spas_sage2_attn_meansim_cuda(
            q, k, v,
            simthreshd1=0.6,
            cdfthreshd=0.97,
            pvthreshd=15,
            is_causal=False,
            return_sparsity=True
        )
        
        if isinstance(output, tuple):
            attn_output, sparsity = output
            print(f"✅ Sparse attention applied successfully")
            print(f"   Output shape: {attn_output.shape}")
            print(f"   Achieved sparsity: {sparsity:.3f}")
        else:
            print(f"✅ Sparse attention applied successfully")
            print(f"   Output shape: {output.shape}")
            
    except Exception as e:
        print(f"❌ Error: {e}")

def demo_creative_sparse():
    """Demonstrate Creative-Sparse algorithm"""
    print("\n🎨 Demonstrating Creative-Sparse Algorithm")
    print("-" * 40)
    
    try:
        from spas_sage_attn import CreativeSparseWrapper, QualityWeights
        
        # Initialize wrapper
        wrapper = CreativeSparseWrapper(
            model_name="gpt2-large",
            sparsity_schedule=(0.7, 0.3),  # (divergence, refinement)
            quality_weights=QualityWeights(alpha=0.4, beta=0.3, gamma=0.3)
        )
        
        print("✅ Creative-Sparse wrapper initialized")
        
        # Test prompts
        prompts = [
            "Write a creative story about a robot learning to paint",
            "Describe a world where colors have sounds"
        ]
        
        for i, prompt in enumerate(prompts):
            print(f"\n📝 Prompt {i+1}: {prompt}")
            
            # Generate creative content
            output = wrapper.generate_creative(
                prompt=prompt,
                max_length=100,
                num_candidates=3,
                return_best=True
            )
            
            print(f"🎯 Generated: {output}")
            
            # Compute quality score
            quality_score = wrapper.quality_functional.compute_quality_score([output])
            print(f"⭐ Quality Score: {quality_score[0]:.3f}")
            
    except Exception as e:
        print(f"❌ Error: {e}")

def demo_sparsity_optimization():
    """Demonstrate sparsity optimization"""
    print("\n⚙️ Demonstrating Sparsity Optimization")
    print("-" * 40)
    
    try:
        from spas_sage_attn import find_optimal_sparsity, compute_quality_score
        
        # Test different task types
        tasks = [
            ("creative_writing", 0.8),
            ("image_generation", 0.6),
            ("video_generation", 0.7)
        ]
        
        for task_type, compute_budget in tasks:
            optimal_sparsity = find_optimal_sparsity(
                model_name="gpt2-large",
                task_type=task_type,
                compute_budget=compute_budget
            )
            
            print(f"📊 Task: {task_type}")
            print(f"   Compute Budget: {compute_budget}")
            print(f"   Optimal Sparsity: {optimal_sparsity:.3f}")
        
        # Test quality assessment
        print(f"\n📈 Quality Assessment Example:")
        outputs = [
            "The robot painted a beautiful sunset with vibrant colors.",
            "Color sounds like music, each hue a different note."
        ]
        
        scores = compute_quality_score(outputs)
        for i, (output, score) in enumerate(zip(outputs, scores)):
            print(f"   {i+1}. Score: {score:.3f} - {output[:50]}...")
            
    except Exception as e:
        print(f"❌ Error: {e}")

def main():
    """Run all demonstrations"""
    print("🚀 Creative-Sparse Demonstration")
    print("=" * 50)
    print("This demo showcases the key features of the Creative-Sparse framework")
    print("for controllable creative AI generation using sparsity.")
    print("=" * 50)
    
    # Run demonstrations
    demo_sparse_attention()
    demo_creative_sparse()
    demo_sparsity_optimization()
    
    print("\n" + "=" * 50)
    print("🎉 Demonstration completed!")
    print("\nFor more examples, see:")
    print("  - examples/creative_generation_example.py")
    print("  - evaluate/cogvideo_example.py")
    print("  - evaluate/flux_example.py")
    print("\nFor installation help, see INSTALL.md")

if __name__ == "__main__":
    main()
