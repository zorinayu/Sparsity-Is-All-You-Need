#!/usr/bin/env python3
"""
Run All Creative-Sparse Examples

This script runs all Creative-Sparse examples across different models:
- CogVideoX (video generation)
- Flux (image generation) 
- HunyuanVideo (Chinese video generation)
- WAN (Want2V video generation)
"""

import os
import sys
import argparse
import subprocess
import time
from pathlib import Path

def run_command(cmd, description):
    """Run a command and handle errors"""
    print(f"\n🚀 {description}")
    print(f"Command: {' '.join(cmd)}")
    print("-" * 60)
    
    try:
        result = subprocess.run(cmd, check=True, capture_output=True, text=True)
        print("✅ Success!")
        if result.stdout:
            print("Output:", result.stdout)
        return True
    except subprocess.CalledProcessError as e:
        print(f"❌ Error: {e}")
        if e.stdout:
            print("Stdout:", e.stdout)
        if e.stderr:
            print("Stderr:", e.stderr)
        return False

def main():
    parser = argparse.ArgumentParser(description="Run All Creative-Sparse Examples")
    
    # Model selection
    parser.add_argument("--models", nargs="+", 
                       choices=["cogvideo", "flux", "hunyuan", "wan", "all"],
                       default=["all"], help="Models to run")
    
    # Creative-Sparse options
    parser.add_argument("--use_creative_sparse", action="store_true", default=True,
                       help="Use Creative-Sparse algorithm")
    parser.add_argument("--use_spas_sage_attn", action="store_true", default=True,
                       help="Use SpargeAttention")
    parser.add_argument("--sparsity_schedule", type=float, nargs=2, default=[0.7, 0.3],
                       help="Sparsity schedule for divergence and refinement phases")
    parser.add_argument("--quality_weights", type=float, nargs=3, default=[0.4, 0.3, 0.3],
                       help="Quality weights for novelty, diversity, and coherence")
    parser.add_argument("--num_candidates", type=int, default=3,
                       help="Number of candidates to generate in divergence phase")
    parser.add_argument("--auto_sparsity", action="store_true",
                       help="Automatically find optimal sparsity level")
    
    # General options
    parser.add_argument("--tune", action="store_true", help="Tune hyperparameters")
    parser.add_argument("--verbose", action="store_true", help="Verbose output")
    parser.add_argument("--skip_existing", action="store_true", 
                       help="Skip if output directory already exists")
    
    args = parser.parse_args()
    
    # Determine which models to run
    if "all" in args.models:
        models_to_run = ["cogvideo", "flux", "hunyuan", "wan"]
    else:
        models_to_run = args.models
    
    print("🎨 Creative-Sparse Comprehensive Evaluation")
    print("=" * 60)
    print(f"Models to run: {models_to_run}")
    print(f"Creative-Sparse: {args.use_creative_sparse}")
    print(f"SpargeAttention: {args.use_spas_sage_attn}")
    print(f"Sparsity Schedule: {args.sparsity_schedule}")
    print(f"Quality Weights: {args.quality_weights}")
    print(f"Number of Candidates: {args.num_candidates}")
    print("=" * 60)
    
    # Base command arguments
    base_args = []
    if args.use_creative_sparse:
        base_args.extend(["--use_creative_sparse"])
    if args.use_spas_sage_attn:
        base_args.extend(["--use_spas_sage_attn"])
    if args.tune:
        base_args.extend(["--tune"])
    if args.verbose:
        base_args.extend(["--verbose"])
    if args.auto_sparsity:
        base_args.extend(["--auto_sparsity"])
    
    base_args.extend([
        "--sparsity_schedule", str(args.sparsity_schedule[0]), str(args.sparsity_schedule[1]),
        "--quality_weights", str(args.quality_weights[0]), str(args.quality_weights[1]), str(args.quality_weights[2]),
        "--num_candidates", str(args.num_candidates)
    ])
    
    # Results tracking
    results = {}
    start_time = time.time()
    
    # Run each model
    for model in models_to_run:
        print(f"\n{'='*60}")
        print(f"🎬 Running {model.upper()} Model")
        print(f"{'='*60}")
        
        # Check if output exists and skip if requested
        output_dir = f"evaluate/datasets/{'video' if model != 'flux' else 'image'}/creative_sparse_{model}"
        if args.skip_existing and os.path.exists(output_dir):
            print(f"⏭️ Skipping {model} - output directory exists: {output_dir}")
            results[model] = "skipped"
            continue
        
        # Build command for each model
        if model == "cogvideo":
            cmd = ["python", "evaluate/creative_sparse_cogvideo.py"] + base_args
            description = "Running Creative-Sparse CogVideoX Evaluation"
            
        elif model == "flux":
            cmd = ["python", "evaluate/creative_sparse_flux.py"] + base_args
            description = "Running Creative-Sparse Flux Evaluation"
            
        elif model == "hunyuan":
            cmd = ["python", "evaluate/creative_sparse_hunyuan.py"] + base_args
            description = "Running Creative-Sparse HunyuanVideo Evaluation"
            
        elif model == "wan":
            cmd = ["python", "evaluate/creative_sparse_wan.py"] + base_args
            description = "Running Creative-Sparse WAN Evaluation"
        
        # Run the command
        success = run_command(cmd, description)
        results[model] = "success" if success else "failed"
        
        if success:
            print(f"✅ {model.upper()} completed successfully!")
        else:
            print(f"❌ {model.upper()} failed!")
    
    # Print summary
    end_time = time.time()
    total_time = end_time - start_time
    
    print(f"\n{'='*60}")
    print("📊 EVALUATION SUMMARY")
    print(f"{'='*60}")
    print(f"Total time: {total_time:.2f} seconds")
    print(f"Models run: {len(models_to_run)}")
    print()
    
    for model, status in results.items():
        status_emoji = "✅" if status == "success" else "❌" if status == "failed" else "⏭️"
        print(f"{status_emoji} {model.upper()}: {status}")
    
    # Success rate
    successful = sum(1 for status in results.values() if status == "success")
    total = len(results)
    success_rate = (successful / total * 100) if total > 0 else 0
    
    print(f"\n🎯 Success Rate: {successful}/{total} ({success_rate:.1f}%)")
    
    if successful == total:
        print("🎉 All evaluations completed successfully!")
    elif successful > 0:
        print("⚠️ Some evaluations completed successfully.")
    else:
        print("❌ All evaluations failed.")
    
    print(f"\n📁 Check output directories for results:")
    for model in models_to_run:
        output_type = "video" if model != "flux" else "image"
        output_dir = f"evaluate/datasets/{output_type}/creative_sparse_{model}"
        print(f"   {model.upper()}: {output_dir}")

if __name__ == "__main__":
    main()
