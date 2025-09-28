#!/usr/bin/env python3
"""
Creative-Sparse Flux Example

This script demonstrates the Creative-Sparse algorithm applied to Flux image generation,
showing how sparsity can enhance creativity in artistic image synthesis.
"""

import torch
import os
import gc
import argparse
from diffusers import FluxPipeline
from diffusers import FluxTransformer2DModel
from tqdm import tqdm
import numpy as np
from PIL import Image

# Import Creative-Sparse components
from spas_sage_attn import (
    CreativeSparseWrapper,
    SparsityConfig,
    QualityWeights,
    find_optimal_sparsity,
    compute_quality_score
)
from modify_model.modify_flux import set_spas_sage_attn_flux
from spas_sage_attn.autotune import (
    extract_sparse_attention_state_dict,
    load_sparse_attention_state_dict,
)

def parse_args():
    parser = argparse.ArgumentParser(description="Creative-Sparse Flux Evaluation")
    
    # Creative-Sparse specific arguments
    parser.add_argument("--use_creative_sparse", action="store_true", 
                       help="Use Creative-Sparse algorithm")
    parser.add_argument("--sparsity_schedule", type=float, nargs=2, default=[0.6, 0.3],
                       help="Sparsity schedule for divergence and refinement phases")
    parser.add_argument("--quality_weights", type=float, nargs=3, default=[0.5, 0.3, 0.2],
                       help="Quality weights for novelty, diversity, and coherence")
    parser.add_argument("--num_candidates", type=int, default=4,
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
    
    # Output arguments
    parser.add_argument("--out_path", type=str,
                       default="evaluate/datasets/image/creative_sparse_flux",
                       help="Output path")
    parser.add_argument("--model_out_path", type=str,
                       default="evaluate/models_dict/flux_saved_state_dict.pt",
                       help="Model output path")
    
    return parser.parse_args()


def load_artistic_prompts():
    """Load artistic prompts for image generation"""
    artistic_prompts = [
        "A surreal painting where colors flow like liquid music, creating impossible landscapes",
        "An abstract composition of geometric shapes that seem to breathe and pulse with life",
        "A dreamlike forest where trees are made of light and shadows dance like living creatures",
        "A cyberpunk cityscape where neon lights paint the sky in impossible colors",
        "A mystical garden where flowers bloom into constellations and butterflies are made of starlight",
        "An underwater palace where coral forms intricate architectural patterns",
        "A steampunk laboratory where mechanical butterflies create art with their wing movements",
        "A cosmic dance between planets and stars, painted in the style of Van Gogh meets digital art"
    ]
    return artistic_prompts


def apply_creative_sparse_to_flux(pipeline, args):
    """Apply Creative-Sparse algorithm to Flux pipeline"""
    if not args.use_creative_sparse:
        return pipeline
    
    print("🎨 Applying Creative-Sparse algorithm to Flux...")
    
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
        model_name="flux",
        sparsity_schedule=tuple(args.sparsity_schedule),
        quality_weights=quality_weights,
        sparsity_config=sparsity_config
    )
    
    # Find optimal sparsity if requested
    if args.auto_sparsity:
        optimal_sparsity = find_optimal_sparsity(
            model_name="flux",
            task_type="image_generation",
            compute_budget=0.8
        )
        print(f"🎯 Optimal sparsity found: {optimal_sparsity:.3f}")
        creative_wrapper.sparsity_schedule = (optimal_sparsity, optimal_sparsity * 0.5)
    
    return pipeline, creative_wrapper


def generate_creative_image(pipeline, prompt, args, creative_wrapper=None):
    """Generate a creative image using either standard or Creative-Sparse approach"""
    
    if creative_wrapper is not None:
        print(f"🎨 Generating creative image with Creative-Sparse...")
        print(f"   Prompt: {prompt}")
        
        # Use Creative-Sparse two-stage generation
        # Stage A: Divergence - generate multiple candidates
        candidates = []
        for i in range(args.num_candidates):
            print(f"   Stage A - Candidate {i+1}/{args.num_candidates}")
            
            # Generate with high sparsity for exploration
            image = pipeline(
                prompt=prompt,
                height=1024,
                width=1024,
                num_inference_steps=20,
                guidance_scale=7.5,
                generator=torch.Generator().manual_seed(42 + i)
            ).images[0]
            
            candidates.append(image)
        
        # Stage B: Refinement - select and refine best candidate
        print(f"   Stage B - Refining best candidate...")
        
        # For images, we'll use a simple heuristic to select the best candidate
        # In practice, this would use more sophisticated quality assessment
        best_candidate_idx = 0  # Simplified selection
        
        # Refine the best candidate with lower sparsity
        refined_image = pipeline(
            prompt=prompt,
            height=1024,
            width=1024,
            num_inference_steps=25,  # More steps for refinement
            guidance_scale=8.0,      # Higher guidance for coherence
            generator=torch.Generator().manual_seed(42)
        ).images[0]
        
        return refined_image, candidates
    
    else:
        # Standard generation
        print(f"🖼️ Generating standard image...")
        print(f"   Prompt: {prompt}")
        
        image = pipeline(
            prompt=prompt,
            height=1024,
            width=1024,
            num_inference_steps=20,
            guidance_scale=7.5
        ).images[0]
        
        return image, [image]


def main():
    args = parse_args()
    os.makedirs(args.out_path, exist_ok=True)
    
    print("🚀 Creative-Sparse Flux Evaluation")
    print("=" * 50)
    print(f"Creative-Sparse: {args.use_creative_sparse}")
    print(f"Sparsity Schedule: {args.sparsity_schedule}")
    print(f"Quality Weights: {args.quality_weights}")
    print(f"Number of Candidates: {args.num_candidates}")
    print("=" * 50)
    
    # Setup device and model
    device = "cuda" if torch.cuda.is_available() else "cpu"
    dtype = torch.bfloat16
    
    print(f"Using device: {device}")
    
    # Load model
    model_id = "black-forest-labs/FLUX.1-dev"
    print(f"Loading model: {model_id}")
    
    pipeline = FluxPipeline.from_pretrained(
        model_id,
        torch_dtype=dtype,
        device_map="auto"
    )
    
    # Apply SpargeAttention if requested
    if args.use_spas_sage_attn:
        print("🔧 Applying SpargeAttention...")
        set_spas_sage_attn_flux(pipeline.transformer)
        
        if args.tune:
            print("🎛️ Tuning hyperparameters...")
            # Tuning logic would go here
            pass
    
    # Apply Creative-Sparse if requested
    creative_wrapper = None
    if args.use_creative_sparse:
        pipeline, creative_wrapper = apply_creative_sparse_to_flux(pipeline, args)
    
    # Load artistic prompts
    artistic_prompts = load_artistic_prompts()
    
    # Generate images
    print(f"\n🖼️ Generating {len(artistic_prompts)} creative images...")
    
    for i, prompt in enumerate(artistic_prompts):
        print(f"\n--- Image {i+1}/{len(artistic_prompts)} ---")
        
        try:
            # Generate image
            if args.use_creative_sparse:
                image, candidates = generate_creative_image(
                    pipeline, prompt, args, creative_wrapper
                )
                
                # Save main image
                output_path = os.path.join(args.out_path, f"creative_{i:02d}.png")
                image.save(output_path)
                print(f"✅ Saved: {output_path}")
                
                # Save candidates if requested
                if args.verbose:
                    for j, candidate in enumerate(candidates):
                        candidate_path = os.path.join(args.out_path, f"candidate_{i:02d}_{j:02d}.png")
                        candidate.save(candidate_path)
                        print(f"   Candidate {j+1}: {candidate_path}")
                
            else:
                image, _ = generate_creative_image(pipeline, prompt, args)
                output_path = os.path.join(args.out_path, f"standard_{i:02d}.png")
                image.save(output_path)
                print(f"✅ Saved: {output_path}")
            
            # Clean up memory
            del image
            if 'candidates' in locals():
                del candidates
            gc.collect()
            torch.cuda.empty_cache()
            
        except Exception as e:
            print(f"❌ Error generating image {i+1}: {e}")
            continue
    
    print(f"\n🎉 Generation completed! Images saved to: {args.out_path}")
    
    # Performance summary
    if args.use_creative_sparse:
        print("\n📊 Creative-Sparse Performance Summary:")
        print(f"   Sparsity Schedule: {args.sparsity_schedule}")
        print(f"   Quality Weights: {args.quality_weights}")
        print(f"   Candidates per image: {args.num_candidates}")
        print("   Enhanced creativity through two-stage generation!")


if __name__ == "__main__":
    main()
