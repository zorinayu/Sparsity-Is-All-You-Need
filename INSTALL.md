# Installation Guide

## Prerequisites

- Python ≥ 3.9
- CUDA ≥ 12.0 (≥ 12.8 for Blackwell, ≥ 12.4 for fp8 support)
- PyTorch ≥ 2.3.0

## Installation Steps

### 1. Clone the Repository
```bash
git clone https://github.com/zorinayu/Sparsity-Is-All-You-Need.git
cd Sparsity-Is-All-You-Need
```

### 2. Install Dependencies
```bash
pip install -r requirements.txt
```

### 3. Install the Package
```bash
pip install ninja  # for parallel compilation
python setup.py install
```

### 4. Verify Installation
```bash
python test_creative_sparse.py
```

## Quick Test

After installation, you can test the basic functionality:

```python
import torch
from spas_sage_attn import spas_sage2_attn_meansim_cuda

# Test sparse attention
q = torch.randn(1, 8, 512, 64, dtype=torch.float16, device='cuda')
k = torch.randn(1, 8, 512, 64, dtype=torch.float16, device='cuda')
v = torch.randn(1, 8, 512, 64, dtype=torch.float16, device='cuda')

output = spas_sage2_attn_meansim_cuda(q, k, v, simthreshd1=0.6, cdfthreshd=0.97, pvthreshd=15)
print(f"Output shape: {output.shape}")
```

## Troubleshooting

### CUDA Version Issues
If you encounter CUDA version issues:
1. Check your CUDA version: `nvcc --version`
2. Ensure PyTorch is compatible with your CUDA version
3. For fp8 support, you need CUDA ≥ 12.4

### Compilation Issues
If compilation fails:
1. Ensure you have the latest version of ninja: `pip install --upgrade ninja`
2. Check that your CUDA installation is complete
3. Try building with verbose output: `TORCH_CUDA_ARCH_LIST=8.0 python setup.py build_ext --verbose`

### Import Issues
If you get import errors:
1. Ensure the package is properly installed: `pip list | grep spas-sage-attn`
2. Check that you're using the correct Python environment
3. Try reinstalling: `pip uninstall spas-sage-attn && python setup.py install`

## Examples

### Basic Usage
```bash
# Run the creative generation example
python examples/creative_generation_example.py --demo all

# Run specific demonstrations
python examples/creative_generation_example.py --demo attention
python examples/creative_generation_example.py --demo generation
```

### Model Evaluation
```bash
# CogVideoX evaluation
python evaluate/cogvideo_example.py --use_spas_sage_attn --tune

# Flux evaluation
python evaluate/flux_example.py --use_spas_sage_attn --tune
```

## Performance Tips

1. **GPU Memory**: Use appropriate batch sizes to avoid OOM errors
2. **Sparsity Levels**: Start with moderate sparsity (0.5-0.7) for best results
3. **Sequence Length**: Longer sequences benefit more from sparse attention
4. **Model Size**: Larger models show more significant speedups

## Support

If you encounter issues:
1. Check the [GitHub Issues](https://github.com/dongliu/Sparsity-Is-All-You-Need/issues)
2. Review the [README.md](README.md) for detailed usage examples
3. Run the test suite: `python test_creative_sparse.py`
