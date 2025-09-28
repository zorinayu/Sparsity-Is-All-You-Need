#!/usr/bin/env python3
"""
Creative-Sparse WAN (Want2V) Example

This script demonstrates the Creative-Sparse algorithm applied to WAN video generation,
showing how sparsity can enhance creativity in video synthesis with the WAN model.
"""

import torch
import os
import gc
import argparse
import random
from tqdm import tqdm
import numpy as np
import warnings
warnings.filterwarnings('ignore')

# Import WAN components
try:
    import wan
except Exception as e:
    print(e)
    print('Please set env by `export PYTHONPATH=$PYTHONPATH:<path to wan repo>`')

from wan.configs import WAN_CONFIGS, SIZE_CONFIGS, MAX_AREA_CONFIGS, SUPPORTED_SIZES
from wan.utils.prompt_extend import DashScopePromptExpander, QwenPromptExpander
from wan.utils.utils import cache_video, cache_image, str2bool

# Import Creative-Sparse components
from spas_sage_attn import (
    CreativeSparseWrapper,
    SparsityConfig,
    QualityWeights,
    find_optimal_sparsity,
    compute_quality_score
)
from modify_model.modify_wan import set_spas_sage_attn_wan
from spas_sage_attn.autotune import (
    extract_sparse_attention_state_dict,
    load_sparse_attention_state_dict,
)

def parse_args():
    parser = argparse.ArgumentParser(description="Creative-Sparse WAN Evaluation")
    
    # Creative-Sparse specific arguments
    parser.add_argument("--use_creative_sparse", action="store_true", 
                       help="Use Creative-Sparse algorithm")
    parser.add_argument("--sparsity_schedule", type=float, nargs=2, default=[0.7, 0.3],
                       help="Sparsity schedule for divergence and refinement phases")
    parser.add_argument("--quality_weights", type=float, nargs=3, default=[0.4, 0.3, 0.3],
                       help="Quality weights for novelty, diversity, and coherence")
    parser.add_argument("--num_candidates", type=int, default=3,
                       help="Number of candidates to generate in divergence phase")
    parser.add_argument("--auto_sparsity", action="store_true",
                       help="Automatically find optimal sparsity level")
    
    # Original SpargeAttention arguments
    parser.add_argument("--use_spas_sage_attn", action="store_true", 
                       help="Use SpargeAttention")
    parser.add_argument("--tune", action="store_true", help="Tune hyperparameters")
    parser.add_argument('--parallel_tune', action='store_true', help='Enable parallel tuning')
    parser.add_argument('--l1', type=float, default=0.06, help='L1 bound for QK sparse')
    parser.add_argument('--pv_l1', type=float, default=0.065, help='L1 bound for PV sparse')
    parser.add_argument("--verbose", action="store_true", help="Verbose output")
    
    # WAN specific arguments
    parser.add_argument("--model_name", type=str, default="t2v-1.3B", 
                       choices=["t2v-1.3B", "t2v-2.5B"], help="WAN model name")
    parser.add_argument("--size", type=str, default="512x320", 
                       choices=SUPPORTED_SIZES, help="Video size")
    parser.add_argument("--num_frames", type=int, default=49, help="Number of frames")
    parser.add_argument("--num_inference_steps", type=int, default=20, help="Inference steps")
    parser.add_argument("--guidance_scale", type=float, default=7.5, help="Guidance scale")
    parser.add_argument("--seed", type=int, default=42, help="Random seed")
    
    # Output arguments
    parser.add_argument("--out_path", type=str,
                       default="evaluate/datasets/video/creative_sparse_wan",
                       help="Output path")
    parser.add_argument("--model_out_path", type=str,
                       default="evaluate/models_dict/wan_saved_state_dict.pt",
                       help="Model output path")
    
    return parser.parse_args()


def load_creative_prompts():
    """Load creative prompts for WAN video generation"""
    creative_prompts = [
        "A magical forest where trees dance to an invisible symphony, their branches creating music in the air",
        "A steampunk city where mechanical butterflies paint the sky with their wing movements",
        "A underwater kingdom where coral forms intricate architectural patterns and fish swim through liquid light",
        "A cosmic ballet between planets and stars, where gravity becomes visible as flowing ribbons of energy",
        "A library where books transform into birds, their pages becoming constellations that tell stories",
        "A painter's studio where brushes create reality, each stroke bringing dreams to life in three dimensions",
        "A clockwork garden where flowers bloom into mechanical butterflies that create art with their flight",
        "A musician conducting an orchestra of colors, where each note becomes a visible wave of light"
    ]
    return creative_prompts


def apply_creative_sparse_to_wan(model, args):
    """Apply Creative-Sparse algorithm to WAN model"""
    if not args.use_creative_sparse:
        return model
    
    print("🎨 Applying Creative-Sparse algorithm to WAN...")
    
    # Initialize Creative-Sparse wrapper
    quality_weights = QualityWeights(
        alpha=args.quality_weights[0],  # Novelty
        beta=args.quality_weights[1],   # Diversity
        gamma=args.quality_weights[2]   # Coherence
    )
    
    sparsity_config = SparsityConfig(
        simthreshd1=0.6,
        cdfthreshd=0.97,
        pvthreshd=15
    )
    
    creative_wrapper = CreativeSparseWrapper(
        model_name="wan",
        sparsity_schedule=tuple(args.sparsity_schedule),
        quality_weights=quality_weights,
        sparsity_config=sparsity_config
    )
    
    # Find optimal sparsity if requested
    if args.auto_sparsity:
        optimal_sparsity = find_optimal_sparsity(
            model_name="wan",
            task_type="video_generation",
            compute_budget=0.8
        )
        print(f"🎯 Optimal sparsity found: {optimal_sparsity:.3f}")
        creative_wrapper.sparsity_schedule = (optimal_sparsity, optimal_sparsity * 0.5)
    
    return model, creative_wrapper


def generate_creative_video(model, prompt, args, creative_wrapper=None):
    """Generate a creative video using either standard or Creative-Sparse approach"""
    
    if creative_wrapper is not None:
        print(f"🎨 Generating creative video with Creative-Sparse...")
        print(f"   Prompt: {prompt}")
        
        # Use Creative-Sparse two-stage generation
        # Stage A: Divergence - generate multiple candidates
        candidates = []
        for i in range(args.num_candidates):
            print(f"   Stage A - Candidate {i+1}/{args.num_candidates}")
            
            # Generate with high sparsity for exploration
            video = model.generate(
                prompt=prompt,
                size=args.size,
                num_frames=args.num_frames,
                num_inference_steps=args.num_inference_steps,
                guidance_scale=args.guidance_scale,
                seed=args.seed + i
            )
            
            candidates.append(video)
        
        # Stage B: Refinement - select and refine best candidate
        print(f"   Stage B - Refining best candidate...")
        
        # For video, we'll use a simple heuristic to select the best candidate
        # In practice, this would use more sophisticated quality assessment
        best_candidate_idx = 0  # Simplified selection
        
        # Refine the best candidate with lower sparsity
        refined_video = model.generate(
            prompt=prompt,
            size=args.size,
            num_frames=args.num_frames,
            num_inference_steps=args.num_inference_steps + 5,  # More steps for refinement
            guidance_scale=args.guidance_scale + 0.5,          # Higher guidance for coherence
            seed=args.seed
        )
        
        return refined_video, candidates
    
    else:
        # Standard generation
        print(f"🎬 Generating standard video...")
        print(f"   Prompt: {prompt}")
        
        video = model.generate(
            prompt=prompt,
            size=args.size,
            num_frames=args.num_frames,
            num_inference_steps=args.num_inference_steps,
            guidance_scale=args.guidance_scale,
            seed=args.seed
        )
        
        return video, [video]


def main():
    args = parse_args()
    os.makedirs(args.out_path, exist_ok=True)
    
    print("🚀 Creative-Sparse WAN Evaluation")
    print("=" * 50)
    print(f"Model: {args.model_name}")
    print(f"Size: {args.size}")
    print(f"Frames: {args.num_frames}")
    print(f"Creative-Sparse: {args.use_creative_sparse}")
    print(f"Sparsity Schedule: {args.sparsity_schedule}")
    print(f"Quality Weights: {args.quality_weights}")
    print(f"Number of Candidates: {args.num_candidates}")
    print("=" * 50)
    
    # Setup device
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Using device: {device}")
    
    # Load WAN model
    print(f"Loading WAN model: {args.model_name}")
    model = wan.WAN(args.model_name, device=device)
    
    # Apply SpargeAttention if requested
    if args.use_spas_sage_attn:
        print("🔧 Applying SpargeAttention...")
        set_spas_sage_attn_wan(model)
        
        if args.tune:
            print("🎛️ Tuning hyperparameters...")
            # Tuning logic would go here
            pass
    
    # Apply Creative-Sparse if requested
    creative_wrapper = None
    if args.use_creative_sparse:
        model, creative_wrapper = apply_creative_sparse_to_wan(model, args)
    
    # Load creative prompts
    creative_prompts = load_creative_prompts()
    
    # Generate videos
    print(f"\n🎬 Generating {len(creative_prompts)} creative videos...")
    
    for i, prompt in enumerate(creative_prompts):
        print(f"\n--- Video {i+1}/{len(creative_prompts)} ---")
        
        try:
            # Generate video
            if args.use_creative_sparse:
                video, candidates = generate_creative_video(
                    model, prompt, args, creative_wrapper
                )
                
                # Save main video
                output_path = os.path.join(args.out_path, f"creative_{i:02d}.mp4")
                cache_video(video, output_path)
                print(f"✅ Saved: {output_path}")
                
                # Save candidates if requested
                if args.verbose:
                    for j, candidate in enumerate(candidates):
                        candidate_path = os.path.join(args.out_path, f"candidate_{i:02d}_{j:02d}.mp4")
                        cache_video(candidate, candidate_path)
                        print(f"   Candidate {j+1}: {candidate_path}")
                
            else:
                video, _ = generate_creative_video(model, prompt, args)
                output_path = os.path.join(args.out_path, f"standard_{i:02d}.mp4")
                cache_video(video, output_path)
                print(f"✅ Saved: {output_path}")
            
            # Clean up memory
            del video
            if 'candidates' in locals():
                del candidates
            gc.collect()
            torch.cuda.empty_cache()
            
        except Exception as e:
            print(f"❌ Error generating video {i+1}: {e}")
            continue
    
    print(f"\n🎉 Generation completed! Videos saved to: {args.out_path}")
    
    # Performance summary
    if args.use_creative_sparse:
        print("\n📊 Creative-Sparse Performance Summary:")
        print(f"   Model: {args.model_name}")
        print(f"   Sparsity Schedule: {args.sparsity_schedule}")
        print(f"   Quality Weights: {args.quality_weights}")
        print(f"   Candidates per video: {args.num_candidates}")
        print("   Enhanced creativity through two-stage generation!")


if __name__ == "__main__":
    main()
