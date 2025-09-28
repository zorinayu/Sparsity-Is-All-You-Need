# Sparsity Is All You Need for High-Quality Creative Generation

<div align="center">
  <h2>Creative-Sparse: A Two-Stage Algorithm for Controllable Creative AI</h2>
  <p><strong>Dong Liu</strong><sup>1</sup> | <strong>Yanxuan Yu</strong><sup>2</sup></p>
  <p><sup>1</sup>Yale University | <sup>2</sup>Columbia University</p>
  
  <a href="https://arxiv.org/abs/XXXX.XXXXX">
    <img src="https://img.shields.io/badge/arXiv-XXXX.XXXXX-b31b1b.svg" alt="arXiv">
  </a>
  <a href="https://github.com/dongliu/Sparsity-Is-All-You-Need">
    <img src="https://img.shields.io/badge/GitHub-Repository-blue.svg" alt="GitHub">
  </a>
  <a href="https://pypi.org/project/spas-sage-attn/">
    <img src="https://img.shields.io/badge/PyPI-SpargeAttention-green.svg" alt="PyPI">
  </a>
</div>

## Abstract

Dense generative models often converge to safe but unoriginal outputs, whereas overly sparse ones lose coherence and controllability. We show that **controllable sparsity is a primary dial to balance novelty, diversity, and coherence for creative AI**. 

We formalize a quality functional **Q(s) = α Novelty(s) + β Diversity(s) − γ Incoherence(s)** and prove the existence of a **Goldilocks sparsity s⋆** that maximizes Q under mild regularity assumptions. Building on this, we propose **Creative-Sparse**, a two-stage schedule: (i) **Diverge** with higher sparsity to explore; (ii) **Refine** with lower sparsity under constraint-aware decoding to consolidate coherence.

Across text, image, music, and code tasks, Creative-Sparse establishes consistent Pareto gains under equal compute budgets and yields robust improvements in human preference.

## Key Contributions

- **🎯 C1: Sparsity–Quality Model** - A measurable quality functional Q(s) predicting a Goldilocks sparsity s⋆ with theoretical guarantees
- **🚀 C2: Creative-Sparse Algorithm** - A two-stage diverge→refine schedule with joint knobs for attention sparsity, token/patch pruning, MoE expert usage, KV reuse, and sampling entropy
- **📊 C3: Evaluation Protocol** - Cross-modal tasks (text/image/music/code), automatic metrics, human-in-the-loop paired comparisons, and compute-controlled reporting
- **🛠️ C4: Toolkit** - A plug-in layer exposing unified sparsity and sampling controls under equal-FLOPs constraints

## The Goldilocks Zone Theory

Our theoretical framework establishes that there exists an optimal sparsity level s⋆ ∈ (0,1) that maximizes creative quality:

**Theorem 1 (Existence of optimal sparsity)**: Under regularity assumptions, there exists s⋆ ∈ [0,1] maximizing Q(s). If the derivative changes sign exactly once on (0,1), then s⋆ ∈ (0,1) is unique.

**Corollary 1 (Goldilocks bounds)**: The optimal sparsity satisfies:
```
(αa₁μ/γa₂ν)^(1/(ν-μ)) ≤ s⋆ ≤ min(1, (βa₃κ/γa₂ν)^(1/(ν-1)))
```

## Creative-Sparse Algorithm

Our two-stage approach mirrors human cognitive workflow: **divergent thinking** followed by **convergent refinement**.

### Stage A: Divergence Phase
- **High sparsity** (s ∈ [0.6, 0.8]) for exploration
- Enhanced randomness and subnetwork diversity
- Sparse attention, token/patch pruning, MoE routing entropy

### Stage B: Refinement Phase  
- **Low sparsity** (s ∈ [0.2, 0.4]) for consolidation
- Constraint-aware decoding
- Coherence validation and safety checks

### Unified Control Manifold
```
M(s) = {T(s), pnuc(s), K(s), kattn(s)}
```

Where:
- **T(s)** = T₀ exp(κₜ · s · (1 − s/2)) - exponential temperature modulation
- **pnuc(s)** = p₀ + κₚ · s · tanh(s/σₚ) - sigmoidal nucleus adaptation  
- **K(s)** = K₀ + κₖ · s · E · (1 − exp(−s/τₖ)) - saturating expert activation
- **kattn(s)** = k₀ + κₐ · s · d · √(1 − s²) - quadratic attention scaling

## Installation

### Requirements
- Python ≥ 3.9
- PyTorch ≥ 2.3.0
- CUDA ≥ 12.0 (≥ 12.8 for Blackwell, ≥ 12.4 for fp8 support)

### Install from Source
```bash
git clone https://github.com/dongliu/Sparsity-Is-All-You-Need.git
cd Sparsity-Is-All-You-Need
pip install ninja  # for parallel compilation
python setup.py install
```

### Install from PyPI
```bash
pip install spas-sage-attn
```

## Quick Start

### Basic Sparse Attention Usage
```python
import torch
from spas_sage_attn import spas_sage2_attn_meansim_cuda

# Simple usage without tuning
q = torch.randn(1, 16, 512, 64, dtype=torch.float16, device='cuda')
k = torch.randn(1, 16, 512, 64, dtype=torch.float16, device='cuda')
v = torch.randn(1, 16, 512, 64, dtype=torch.float16, device='cuda')

# Apply sparse attention with default parameters
attn_output = spas_sage2_attn_meansim_cuda(
    q, k, v, 
    simthreshd1=0.6,  # similarity threshold for sparsity
    cdfthreshd=0.97,  # CDF threshold for block selection
    pvthreshd=15,     # PV threshold for further sparsity
    is_causal=False
)
```

### Creative-Sparse Integration
```python
from spas_sage_attn import CreativeSparseWrapper
from spas_sage_attn.autotune import find_optimal_sparsity

# Initialize Creative-Sparse wrapper
wrapper = CreativeSparseWrapper(
    model_name="gpt2-large",
    sparsity_schedule=(0.7, 0.3),  # (divergence, refinement)
    quality_weights=(0.4, 0.3, 0.3)  # (α, β, γ)
)

# Find optimal sparsity for your task
optimal_sparsity = find_optimal_sparsity(
    model_name="gpt2-large",
    task_type="creative_writing",
    compute_budget=1.0
)

# Generate with optimal sparsity
output = wrapper.generate_creative(
    prompt="Write a creative story about a robot learning to paint",
    max_length=200,
    sparsity_level=optimal_sparsity
)
```

## Supported Models and Tasks

### Video Generation
- **Models**: CogVideoX, Want2V
- **Tasks**: Creative video synthesis with sparse attention acceleration
- **Example**: `evaluate/cogvideo_example.py`, `evaluate/wan_example.py`

### Image Generation
- **Models**: Flux, Hunyuan
- **Tasks**: High-quality image generation with sparse attention
- **Example**: `evaluate/flux_example.py`, `evaluate/hunyuan_example.py`

### Text Generation (Experimental)
- **Models**: GPT-2, GPT-Neo, LLaMA-2
- **Tasks**: Creative writing, story generation
- **Note**: Text generation support is experimental and requires custom model integration

### Available APIs
- `spas_sage2_attn_meansim_cuda`: Main sparse attention function based on SageAttention2
- `spas_sage_attn_meansim_cuda`: Legacy version based on SageAttention
- `block_sparse_sage2_attn_cuda`: Block-sparse attention with custom patterns

## Performance Results

### Optimal Sparsity Levels Across Modalities

| Modality | Small Model | Medium Model | Large Model | Average |
|----------|-------------|--------------|-------------|---------|
| Text     | 0.63        | 0.58         | 0.67        | 0.63    |
| Image    | 0.49        | 0.54         | 0.61        | 0.55    |
| Music    | 0.52        | 0.59         | 0.64        | 0.58    |
| Code     | 0.47        | 0.51         | 0.56        | 0.51    |
| **Average** | **0.53** | **0.56** | **0.62** | **0.57** |

### Compute-Controlled Results

| Method | Novelty ↑ | Diversity ↑ | Incoherence ↓ | Human Pref. ↑ | Latency ↓ | Memory ↓ |
|--------|-----------|-------------|---------------|---------------|-----------|----------|
| Dense (top-p) | 0.00 | 0.00 | 0.00 | 0.00 | 100 | 100 |
| Fixed Sparse | +0.13 | +0.09 | +0.11 | +0.16 | 73 | 87 |
| Progressive | +0.19 | +0.15 | +0.08 | +0.24 | 76 | 84 |
| Adaptive | +0.17 | +0.12 | +0.09 | +0.21 | 79 | 81 |
| MoE-Only | +0.11 | +0.08 | +0.06 | +0.14 | 83 | 92 |
| **Creative-Sparse** | **+0.31** | **+0.26** | **+0.03** | **+0.35** | **79** | **69** |

### Human Evaluation Results

| Domain | Creative-Sparse | Dense | Fixed Sparse | Δ vs Dense |
|--------|-----------------|-------|--------------|------------|
| Creative Writing | 4.2 | 3.1 | 3.5 | +1.1 |
| Poetry Generation | 4.0 | 2.8 | 3.2 | +1.2 |
| Ad Copy Creation | 4.3 | 3.3 | 3.7 | +1.0 |
| Artistic Image Gen. | 4.1 | 3.0 | 3.4 | +1.1 |
| Music Composition | 3.9 | 2.9 | 3.3 | +1.0 |
| Code Generation | 4.0 | 3.2 | 3.6 | +0.8 |
| **Average** | **4.1** | **3.1** | **3.5** | **+1.0** |

## API Reference

### Core Classes

#### `CreativeSparseWrapper`
Main wrapper class implementing the two-stage Creative-Sparse algorithm.

```python
class CreativeSparseWrapper:
    def __init__(
        self,
        model_name: str,
        sparsity_schedule: Tuple[float, float] = (0.7, 0.3),
        quality_weights: Optional[QualityWeights] = None,
        sparsity_config: Optional[SparsityConfig] = None
    )
    
    def generate_creative(
        self,
        prompt: str,
        max_length: int = 200,
        num_candidates: int = 5,
        return_best: bool = True,
        sparsity_level: Optional[float] = None
    ) -> Union[str, List[str]]
    
    def apply_sparse_attention(
        self, 
        q: torch.Tensor, 
        k: torch.Tensor, 
        v: torch.Tensor,
        sparsity: float,
        is_causal: bool = False
    ) -> torch.Tensor
```

#### `SparsityConfig`
Configuration class for fine-tuning sparsity parameters.

```python
@dataclass
class SparsityConfig:
    attention_sparsity: float = 0.6
    token_pruning_rate: float = 0.4
    moe_expert_entropy: float = 0.8
    kv_reuse_ratio: float = 0.3
    temperature_schedule: str = "exponential"
    nucleus_adaptation: str = "sigmoidal"
    simthreshd1: float = 0.6
    cdfthreshd: float = 0.97
    pvthreshd: int = 15
```

#### `QualityWeights`
Weights for the quality functional Q(s) = α Novelty(s) + β Diversity(s) − γ Incoherence(s).

```python
@dataclass
class QualityWeights:
    alpha: float = 0.4  # Novelty weight
    beta: float = 0.3   # Diversity weight
    gamma: float = 0.3  # Incoherence weight
```

### Utility Functions

#### Quality Assessment
```python
def compute_quality_score(
    outputs: List[str],
    novelty_weight: float = 0.4,
    diversity_weight: float = 0.3,
    coherence_weight: float = 0.3,
    reference_data: Optional[List[str]] = None
) -> List[float]
```

#### Sparsity Optimization
```python
def find_optimal_sparsity(
    model_name: str,
    task_type: str,
    compute_budget: float,
    quality_weights: Optional[QualityWeights] = None
) -> float
```

#### Sparse Attention Functions
```python
# Main sparse attention function
def spas_sage2_attn_meansim_cuda(
    q, k, v, 
    simthreshd1=0.6, 
    cdfthreshd=0.97, 
    pvthreshd=15, 
    is_causal=False,
    return_sparsity=False
) -> torch.Tensor

# Block-sparse attention with custom patterns
def block_sparse_sage2_attn_cuda(
    q, k, v, 
    mask_id=None,
    pvthreshd=50,
    return_sparsity=False
) -> torch.Tensor
```

## Evaluation and Benchmarking

### Running Evaluations
```bash
# Video generation evaluation (CogVideoX)
python evaluate/cogvideo_example.py --use_spas_sage_attn --tune --model_out_path evaluate/models_dict/CogVideoX-2b_0.06_0.07.pt

# Image generation evaluation (Flux)
python evaluate/flux_example.py --use_spas_sage_attn --tune --model_out_path evaluate/models_dict/flux_saved_state_dict.pt

# Video generation evaluation (Want2V)
python evaluate/wan_example.py --use_spas_sage_attn --tune

# Creative-Sparse demonstration
python examples/creative_generation_example.py --demo all
```

### Custom Benchmarking
```python
from spas_sage_attn import CreativeSparseWrapper, find_optimal_sparsity, compute_quality_score

# Benchmark different sparsity levels
sparsity_levels = [0.3, 0.5, 0.7, 0.9]
results = {}

for sparsity in sparsity_levels:
    wrapper = CreativeSparseWrapper(
        model_name="gpt2-large",
        sparsity_schedule=(sparsity, 0.3)
    )
    
    outputs = []
    for prompt in test_prompts:
        output = wrapper.generate_creative(prompt, sparsity_level=sparsity)
        outputs.append(output)
    
    quality_scores = compute_quality_score(outputs)
    results[sparsity] = {
        'avg_quality': np.mean(quality_scores),
        'outputs': outputs
    }

# Find optimal sparsity
optimal_sparsity = find_optimal_sparsity(
    model_name="gpt2-large",
    task_type="creative_writing",
    compute_budget=0.8
)
```

## Advanced Usage

### Custom Quality Functionals
```python
from spas_sage_attn import QualityFunctional, QualityWeights

class CustomQuality(QualityFunctional):
    def __init__(self, custom_weights):
        super().__init__(custom_weights)
    
    def compute_novelty(self, outputs, reference_data=None):
        # Custom novelty computation using your own metrics
        # Example: Use semantic similarity with custom embeddings
        return self._custom_novelty_score(outputs, reference_data)
    
    def compute_diversity(self, outputs):
        # Custom diversity computation
        # Example: Use custom n-gram diversity or semantic diversity
        return self._custom_diversity_score(outputs)
    
    def compute_coherence(self, outputs):
        # Custom coherence computation
        # Example: Use custom language model or rule-based scoring
        return self._custom_coherence_score(outputs)
    
    def _custom_novelty_score(self, outputs, reference_data):
        # Your custom implementation
        pass
    
    def _custom_diversity_score(self, outputs):
        # Your custom implementation
        pass
    
    def _custom_coherence_score(self, outputs):
        # Your custom implementation
        pass
```

### Custom Sparsity Control
```python
from spas_sage_attn import SparsityController, SparsityConfig

# Create custom sparsity configuration
config = SparsityConfig(
    attention_sparsity=0.7,
    simthreshd1=0.5,
    cdfthreshd=0.95,
    pvthreshd=20
)

controller = SparsityController(config)

# Get sparsity parameters for different levels
for sparsity in [0.3, 0.5, 0.7, 0.9]:
    params = controller.get_sparsity_params(sparsity)
    print(f"Sparsity {sparsity}: {params}")
```

### Integration with Existing Models
```python
import torch
from spas_sage_attn import spas_sage2_attn_meansim_cuda

# Replace standard attention in your model
class CustomModel(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.attention = spas_sage2_attn_meansim_cuda
    
    def forward(self, q, k, v, sparsity=0.6):
        # Use sparse attention instead of standard attention
        return self.attention(
            q, k, v,
            simthreshd1=0.6 * (1 - sparsity * 0.3),
            cdfthreshd=0.97 * (1 - sparsity * 0.2),
            pvthreshd=max(5, int(15 * (1 - sparsity * 0.4))),
            is_causal=True
        )
```

## Theoretical Background

### Quality Functional Formulation
The core of our approach is the quality functional:

**Q(s) = α Novelty(s) + β Diversity(s) − γ Incoherence(s)**

Where:
- **Novelty(s)** = E[min_y∈D_train ||f(x) − f(y)||²] - distance from training distribution
- **Diversity(s)** = Distinct-n(x₁:ₘ) - inter-sample diversity measures  
- **Incoherence(s)** = E[−log E(x)] + λ · Violation(x) - likelihood and constraint violations

### Mathematical Properties
Under regularity assumptions, we prove:
1. **Existence**: Q(s) has a global maximum s⋆ ∈ [0,1]
2. **Uniqueness**: If derivative changes sign once, s⋆ is unique
3. **Bounds**: Optimal sparsity satisfies theoretical bounds

### Information-Theoretic Interpretation
Sparsity reshapes mutual information I(X;Y) by selecting routing paths:
- Novelty correlates with KL(P̂_s || P_train)
- Coherence correlates with conditional mutual information under constraints

## Safety and Limitations

### Safety Measures
- **Constraint-aware refinement** reduces violation rates
- **Safety metrics monitoring** (toxicity, bias, factual accuracy)
- **Human-in-the-loop validation** for high-stakes applications

### Current Limitations
- **Safety-Robustness Trade-offs**: Higher sparsity can increase inappropriate content risk
- **Modality-specific tuning** required for optimal performance
- **Computational overhead** for sparsity optimization

### Mitigation Strategies
- Gated release of high-sparsity models
- Safety filters and content moderation
- Continuous monitoring and feedback loops

## Contributing

We welcome contributions! Please see our [Contributing Guidelines](CONTRIBUTING.md) for details.

### Development Setup
```bash
git clone https://github.com/dongliu/Sparsity-Is-All-You-Need.git
cd Sparsity-Is-All-You-Need
pip install -e ".[dev]"
pre-commit install
```

### Running Tests
```bash
pytest tests/
python -m pytest tests/test_creative_sparse.py -v
```

## Citation

If you use this code or find our work valuable, please cite:

```bibtex
@article{liu2024sparsity,
  title={Sparsity Is All You Need for High-Quality Creative Generation},
  author={Liu, Dong and Yu, Yanxuan},
  journal={arXiv preprint arXiv:XXXX.XXXXX},
  year={2024}
}
```

## License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

## Acknowledgments

- Yale University and Columbia University for research support
- The open-source community for foundational models and tools
- Contributors and users who provide feedback and improvements

## Contact

- **Dong Liu**: dong.liu.dl2367@yale.edu
- **Yanxuan Yu**: yy3523@columbia.edu
- **GitHub Issues**: [Report bugs or request features](https://github.com/dongliu/Sparsity-Is-All-You-Need/issues)

---

<div align="center">
  <p><strong>🌟 Star this repository if you find it helpful!</strong></p>
  <p>Made with ❤️ for the creative AI community</p>
</div>