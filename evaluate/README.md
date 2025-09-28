# Creative-Sparse Evaluation Scripts

This directory contains evaluation scripts that demonstrate the Creative-Sparse algorithm applied to various generative models.

## Available Scripts

### Original SpargeAttention Scripts
- `cogvideo_example.py` - CogVideoX video generation with SpargeAttention
- `flux_example.py` - Flux image generation with SpargeAttention  
- `hunyuan_example.py` - HunyuanVideo generation with SpargeAttention
- `wan_example.py` - WAN (Want2V) video generation with SpargeAttention

### New Creative-Sparse Scripts
- `creative_sparse_cogvideo.py` - CogVideoX with Creative-Sparse algorithm
- `creative_sparse_flux.py` - Flux with Creative-Sparse algorithm
- `creative_sparse_hunyuan.py` - HunyuanVideo with Creative-Sparse algorithm
- `creative_sparse_wan.py` - WAN with Creative-Sparse algorithm
- `run_all_creative_sparse.py` - Run all Creative-Sparse evaluations

## Quick Start

### Run Individual Creative-Sparse Evaluations

```bash
# CogVideoX with Creative-Sparse
python evaluate/creative_sparse_cogvideo.py \
    --use_creative_sparse \
    --use_spas_sage_attn \
    --auto_sparsity \
    --sparsity_schedule 0.7 0.3 \
    --quality_weights 0.4 0.3 0.3

# Flux with Creative-Sparse
python evaluate/creative_sparse_flux.py \
    --use_creative_sparse \
    --use_spas_sage_attn \
    --auto_sparsity \
    --sparsity_schedule 0.6 0.3 \
    --quality_weights 0.5 0.3 0.2

# HunyuanVideo with Creative-Sparse
python evaluate/creative_sparse_hunyuan.py \
    --use_creative_sparse \
    --use_spas_sage_attn \
    --auto_sparsity \
    --sparsity_schedule 0.65 0.35

# WAN with Creative-Sparse
python evaluate/creative_sparse_wan.py \
    --use_creative_sparse \
    --use_spas_sage_attn \
    --auto_sparsity \
    --sparsity_schedule 0.7 0.3
```

### Run All Creative-Sparse Evaluations

```bash
# Run all models with Creative-Sparse
python evaluate/run_all_creative_sparse.py \
    --models all \
    --use_creative_sparse \
    --use_spas_sage_attn \
    --auto_sparsity

# Run specific models
python evaluate/run_all_creative_sparse.py \
    --models cogvideo flux \
    --use_creative_sparse \
    --use_spas_sage_attn
```

## Key Features

### Creative-Sparse Algorithm
- **Two-Stage Generation**: Divergence (high sparsity) → Refinement (low sparsity)
- **Quality Functional**: Q(s) = α Novelty(s) + β Diversity(s) − γ Incoherence(s)
- **Optimal Sparsity**: Automatic finding of Goldilocks sparsity levels
- **Multi-Candidate Generation**: Generate multiple candidates in divergence phase

### Supported Models
- **Video Generation**: CogVideoX, HunyuanVideo, WAN (Want2V)
- **Image Generation**: Flux, Flux-Dev
- **Text Generation**: GPT-2, LLaMA-2 (experimental)

### Creative Prompts
Each script includes carefully crafted creative prompts designed to showcase the algorithm's ability to enhance creativity:

- **CogVideoX**: Robot learning to paint, dancer in light city, reverse-growing tree
- **Flux**: Surreal color music, abstract breathing shapes, dreamlike light forests
- **HunyuanVideo**: Chinese creative prompts with cultural elements
- **WAN**: Magical forests, steampunk cities, cosmic ballets

## Output Structure

```
evaluate/datasets/
├── video/
│   ├── creative_sparse_cogvideo/
│   ├── creative_sparse_hunyuan/
│   └── creative_sparse_wan/
└── image/
    └── creative_sparse_flux/
```

Each output directory contains:
- `creative_XX.mp4/png` - Final refined outputs
- `candidate_XX_YY.mp4/png` - Divergence phase candidates (if --verbose)

## Parameters

### Creative-Sparse Parameters
- `--use_creative_sparse`: Enable Creative-Sparse algorithm
- `--sparsity_schedule`: Two values for divergence and refinement sparsity
- `--quality_weights`: Three values for novelty, diversity, and coherence weights
- `--num_candidates`: Number of candidates to generate in divergence phase
- `--auto_sparsity`: Automatically find optimal sparsity levels

### SpargeAttention Parameters
- `--use_spas_sage_attn`: Enable SpargeAttention acceleration
- `--tune`: Tune hyperparameters for optimal performance
- `--l1`: L1 bound for QK sparsity
- `--pv_l1`: L1 bound for PV sparsity

### General Parameters
- `--verbose`: Show detailed output and save candidates
- `--out_path`: Output directory path
- `--model_out_path`: Path to save tuned model parameters

## Performance Tips

1. **GPU Memory**: Use appropriate batch sizes to avoid OOM errors
2. **Sparsity Levels**: Start with moderate sparsity (0.5-0.7) for best results
3. **Quality Weights**: Adjust based on your priorities:
   - Higher α (novelty): More creative but potentially less coherent
   - Higher β (diversity): More varied outputs
   - Higher γ (coherence): More structured and coherent outputs
4. **Auto-Sparsity**: Use `--auto_sparsity` for optimal performance without manual tuning

## Troubleshooting

### Common Issues
1. **CUDA OOM**: Reduce batch size or use CPU
2. **Model Loading**: Ensure you have sufficient disk space and internet connection
3. **Dependencies**: Install all requirements from `requirements.txt`

### Getting Help
- Check the main [README.md](../README.md) for detailed documentation
- Run `python test_creative_sparse.py` to verify installation
- Use `--verbose` flag for detailed debugging information

## Citation

If you use these evaluation scripts, please cite:

```bibtex
@article{liu2025sparsity,
  title={Sparsity Is All You Need for High-Quality Creative Generation},
  author={Liu, Dong and Yu, Yanxuan},
  year={2025}
}
```
