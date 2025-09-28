#!/usr/bin/env python3
"""
Test script for Creative-Sparse implementation
"""

import torch
import sys
import os

# Add current directory to path
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

def test_imports():
    """Test that all imports work correctly"""
    print("Testing imports...")
    try:
        from spas_sage_attn import (
            CreativeSparseWrapper,
            SparsityConfig,
            QualityWeights,
            find_optimal_sparsity,
            compute_quality_score,
            spas_sage2_attn_meansim_cuda
        )
        print("✓ All imports successful")
        return True
    except ImportError as e:
        print(f"✗ Import error: {e}")
        return False

def test_sparse_attention():
    """Test sparse attention functionality"""
    print("\nTesting sparse attention...")
    try:
        from spas_sage_attn import spas_sage2_attn_meansim_cuda
        
        # Create test tensors
        device = 'cuda' if torch.cuda.is_available() else 'cpu'
        q = torch.randn(1, 8, 512, 64, dtype=torch.float16, device=device)
        k = torch.randn(1, 8, 512, 64, dtype=torch.float16, device=device)
        v = torch.randn(1, 8, 512, 64, dtype=torch.float16, device=device)
        
        # Test sparse attention
        output = spas_sage2_attn_meansim_cuda(
            q, k, v,
            simthreshd1=0.6,
            cdfthreshd=0.97,
            pvthreshd=15,
            is_causal=False
        )
        
        print(f"✓ Sparse attention output shape: {output.shape}")
        return True
        
    except Exception as e:
        print(f"✗ Sparse attention error: {e}")
        return False

def test_creative_sparse_wrapper():
    """Test Creative-Sparse wrapper"""
    print("\nTesting Creative-Sparse wrapper...")
    try:
        from spas_sage_attn import CreativeSparseWrapper, QualityWeights, SparsityConfig
        
        # Initialize wrapper
        wrapper = CreativeSparseWrapper(
            model_name="gpt2-large",
            sparsity_schedule=(0.7, 0.3),
            quality_weights=QualityWeights(alpha=0.4, beta=0.3, gamma=0.3)
        )
        
        print("✓ Creative-Sparse wrapper initialized")
        
        # Test quality score computation
        test_outputs = [
            "The robot painted a beautiful sunset with vibrant colors.",
            "Color sounds like music, each hue a different note."
        ]
        
        scores = wrapper.quality_functional.compute_quality_score(test_outputs)
        print(f"✓ Quality scores computed: {scores}")
        
        return True
        
    except Exception as e:
        print(f"✗ Creative-Sparse wrapper error: {e}")
        return False

def test_sparsity_controller():
    """Test sparsity controller"""
    print("\nTesting sparsity controller...")
    try:
        from spas_sage_attn import SparsityController, SparsityConfig
        
        config = SparsityConfig()
        controller = SparsityController(config)
        
        # Test sparsity parameter computation
        sparsity = 0.6
        params = controller.get_sparsity_params(sparsity)
        
        print(f"✓ Sparsity parameters for {sparsity}: {params}")
        return True
        
    except Exception as e:
        print(f"✗ Sparsity controller error: {e}")
        return False

def test_quality_assessment():
    """Test quality assessment functions"""
    print("\nTesting quality assessment...")
    try:
        from spas_sage_attn import compute_quality_score, find_optimal_sparsity
        
        # Test quality score computation
        outputs = [
            "The robot painted a beautiful sunset with vibrant colors.",
            "Color sounds like music, each hue a different note.",
            "Digital dreams dance in binary streams of light."
        ]
        
        scores = compute_quality_score(outputs)
        print(f"✓ Quality scores: {scores}")
        
        # Test optimal sparsity finding
        optimal_sparsity = find_optimal_sparsity(
            model_name="gpt2-large",
            task_type="creative_writing",
            compute_budget=0.8
        )
        print(f"✓ Optimal sparsity: {optimal_sparsity}")
        
        return True
        
    except Exception as e:
        print(f"✗ Quality assessment error: {e}")
        return False

def main():
    """Run all tests"""
    print("Creative-Sparse Test Suite")
    print("=" * 40)
    
    tests = [
        test_imports,
        test_sparse_attention,
        test_creative_sparse_wrapper,
        test_sparsity_controller,
        test_quality_assessment
    ]
    
    passed = 0
    total = len(tests)
    
    for test in tests:
        if test():
            passed += 1
    
    print("\n" + "=" * 40)
    print(f"Test Results: {passed}/{total} tests passed")
    
    if passed == total:
        print("🎉 All tests passed!")
        return 0
    else:
        print("❌ Some tests failed")
        return 1

if __name__ == "__main__":
    exit(main())
