"""
Creative-Sparse: A Two-Stage Algorithm for Controllable Creative AI

This module implements the Creative-Sparse algorithm based on the paper
"Sparsity Is All You Need for High-Quality Creative Generation"
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from typing import Dict, List, Tuple, Optional, Union
from dataclasses import dataclass
import math
from .core import spas_sage2_attn_meansim_cuda, spas_sage_attn_meansim_cuda


@dataclass
class SparsityConfig:
    """Configuration for sparsity parameters"""
    attention_sparsity: float = 0.6
    token_pruning_rate: float = 0.4
    moe_expert_entropy: float = 0.8
    kv_reuse_ratio: float = 0.3
    temperature_schedule: str = "exponential"
    nucleus_adaptation: str = "sigmoidal"
    simthreshd1: float = 0.6
    cdfthreshd: float = 0.97
    pvthreshd: int = 15


@dataclass
class QualityWeights:
    """Weights for quality functional Q(s) = α Novelty(s) + β Diversity(s) − γ Incoherence(s)"""
    alpha: float = 0.4  # Novelty weight
    beta: float = 0.3   # Diversity weight
    gamma: float = 0.3  # Incoherence weight


class QualityFunctional:
    """Base class for quality assessment"""
    
    def __init__(self, weights: QualityWeights):
        self.weights = weights
    
    def compute_novelty(self, outputs: List[str], reference_data: Optional[List[str]] = None) -> float:
        """Compute novelty score based on distance from training distribution"""
        # Simplified implementation using embedding distances
        if not outputs:
            return 0.0
        
        # Use sentence embeddings for novelty computation
        embeddings = self._get_embeddings(outputs)
        if reference_data:
            ref_embeddings = self._get_embeddings(reference_data)
            # Compute minimum distance to reference data
            distances = []
            for emb in embeddings:
                min_dist = min([torch.norm(emb - ref_emb).item() for ref_emb in ref_embeddings])
                distances.append(min_dist)
            return np.mean(distances)
        else:
            # Use variance as novelty proxy
            return torch.var(torch.stack(embeddings)).item()
    
    def compute_diversity(self, outputs: List[str]) -> float:
        """Compute diversity score using Distinct-n metric"""
        if len(outputs) < 2:
            return 0.0
        
        # Compute Distinct-2 (bigram diversity)
        all_bigrams = set()
        total_bigrams = 0
        
        for output in outputs:
            words = output.lower().split()
            for i in range(len(words) - 1):
                bigram = f"{words[i]} {words[i+1]}"
                all_bigrams.add(bigram)
                total_bigrams += 1
        
        if total_bigrams == 0:
            return 0.0
        
        return len(all_bigrams) / total_bigrams
    
    def compute_coherence(self, outputs: List[str]) -> float:
        """Compute coherence score (negative incoherence)"""
        if not outputs:
            return 0.0
        
        # Simplified coherence using perplexity-like metric
        # In practice, this would use a language model
        coherence_scores = []
        for output in outputs:
            # Simple heuristic: longer, more structured outputs are more coherent
            words = output.split()
            if len(words) < 5:
                score = 0.1
            else:
                # Reward longer outputs with good sentence structure
                sentences = output.split('.')
                avg_sentence_length = sum(len(s.split()) for s in sentences) / len(sentences)
                score = min(1.0, avg_sentence_length / 10.0)
            coherence_scores.append(score)
        
        return np.mean(coherence_scores)
    
    def compute_quality_score(self, outputs: List[str], reference_data: Optional[List[str]] = None) -> float:
        """Compute overall quality score Q(s)"""
        novelty = self.compute_novelty(outputs, reference_data)
        diversity = self.compute_diversity(outputs)
        coherence = self.compute_coherence(outputs)
        
        quality = (self.weights.alpha * novelty + 
                  self.weights.beta * diversity + 
                  self.weights.gamma * coherence)
        
        return quality
    
    def _get_embeddings(self, texts: List[str]) -> List[torch.Tensor]:
        """Get embeddings for texts (simplified implementation)"""
        # In practice, this would use a proper embedding model
        # For now, use simple bag-of-words representation
        embeddings = []
        for text in texts:
            words = text.lower().split()
            # Create a simple embedding based on word frequencies
            embedding = torch.zeros(100)  # Fixed size for simplicity
            for i, word in enumerate(words[:100]):
                embedding[i] = hash(word) % 1000 / 1000.0
            embeddings.append(embedding)
        return embeddings


class SparsityController:
    """Controls sparsity parameters based on the unified control manifold"""
    
    def __init__(self, config: SparsityConfig):
        self.config = config
        self.kappa_T = 0.5  # Temperature modulation coefficient
        self.kappa_p = 0.3  # Nucleus adaptation coefficient
        self.kappa_K = 0.4  # Expert activation coefficient
        self.kappa_A = 0.2  # Attention scaling coefficient
    
    def get_temperature(self, sparsity: float) -> float:
        """T(s) = T₀ exp(κₜ · s · (1 − s/2))"""
        T0 = 1.0
        return T0 * math.exp(self.kappa_T * sparsity * (1 - sparsity / 2))
    
    def get_nucleus_threshold(self, sparsity: float) -> float:
        """pnuc(s) = p₀ + κₚ · s · tanh(s/σₚ)"""
        p0 = 0.9
        sigma_p = 0.5
        return p0 + self.kappa_p * sparsity * math.tanh(sparsity / sigma_p)
    
    def get_expert_activation(self, sparsity: float, num_experts: int = 8) -> int:
        """K(s) = K₀ + κₖ · s · E · (1 − exp(−s/τₖ))"""
        K0 = 1
        tau_K = 0.3
        K = K0 + self.kappa_K * sparsity * num_experts * (1 - math.exp(-sparsity / tau_K))
        return max(1, int(K))
    
    def get_attention_scaling(self, sparsity: float, head_dim: int = 64) -> float:
        """kattn(s) = k₀ + κₐ · s · d · √(1 − s²)"""
        k0 = 1.0
        return k0 + self.kappa_A * sparsity * head_dim * math.sqrt(1 - sparsity**2)
    
    def get_sparsity_params(self, sparsity: float) -> Dict[str, float]:
        """Get all sparsity parameters for given sparsity level"""
        return {
            'temperature': self.get_temperature(sparsity),
            'nucleus_threshold': self.get_nucleus_threshold(sparsity),
            'expert_activation': self.get_expert_activation(sparsity),
            'attention_scaling': self.get_attention_scaling(sparsity),
            'simthreshd1': self.config.simthreshd1 * (1 - sparsity * 0.3),
            'cdfthreshd': self.config.cdfthreshd * (1 - sparsity * 0.2),
            'pvthreshd': max(5, int(self.config.pvthreshd * (1 - sparsity * 0.4)))
        }


class CreativeSparseWrapper:
    """Main wrapper class implementing the Creative-Sparse algorithm"""
    
    def __init__(
        self,
        model_name: str,
        sparsity_schedule: Tuple[float, float] = (0.7, 0.3),
        quality_weights: Optional[QualityWeights] = None,
        sparsity_config: Optional[SparsityConfig] = None
    ):
        self.model_name = model_name
        self.sparsity_schedule = sparsity_schedule  # (divergence, refinement)
        self.quality_weights = quality_weights or QualityWeights()
        self.sparsity_config = sparsity_config or SparsityConfig()
        
        self.quality_functional = QualityFunctional(self.quality_weights)
        self.sparsity_controller = SparsityController(self.sparsity_config)
        
        # Model-specific configurations
        self.model_configs = {
            # Text generation models
            'gpt2-large': {'max_length': 1024, 'temperature': 0.8, 'guidance_scale': 7.5},
            'llama-2-7b': {'max_length': 2048, 'temperature': 0.7, 'guidance_scale': 7.0},
            'gpt-neo': {'max_length': 1024, 'temperature': 0.8, 'guidance_scale': 7.5},
            'gpt-j': {'max_length': 1024, 'temperature': 0.8, 'guidance_scale': 7.5},
            'palm': {'max_length': 2048, 'temperature': 0.7, 'guidance_scale': 7.0},
            
            # Video generation models
            'cogvideox': {'max_length': 512, 'temperature': 0.9, 'guidance_scale': 7.5, 'num_frames': 49},
            'hunyuan': {'max_length': 512, 'temperature': 0.9, 'guidance_scale': 7.5, 'num_frames': 49},
            'wan': {'max_length': 512, 'temperature': 0.9, 'guidance_scale': 7.5, 'num_frames': 49},
            'want2v': {'max_length': 512, 'temperature': 0.9, 'guidance_scale': 7.5, 'num_frames': 49},
            
            # Image generation models
            'flux': {'max_length': 256, 'temperature': 0.8, 'guidance_scale': 7.5, 'height': 1024, 'width': 1024},
            'flux-dev': {'max_length': 256, 'temperature': 0.8, 'guidance_scale': 7.5, 'height': 1024, 'width': 1024},
            'stable-diffusion': {'max_length': 256, 'temperature': 0.8, 'guidance_scale': 7.5, 'height': 512, 'width': 512},
            'imagen': {'max_length': 256, 'temperature': 0.8, 'guidance_scale': 7.5, 'height': 1024, 'width': 1024},
            'dit-xl': {'max_length': 256, 'temperature': 0.8, 'guidance_scale': 7.5, 'height': 512, 'width': 512},
            
            # Music generation models
            'musiclm': {'max_length': 512, 'temperature': 0.9, 'guidance_scale': 8.0, 'duration': 30},
            'jukebox': {'max_length': 512, 'temperature': 0.9, 'guidance_scale': 8.0, 'duration': 30},
            'musenet': {'max_length': 512, 'temperature': 0.9, 'guidance_scale': 8.0, 'duration': 30},
            
            # Code generation models
            'codegen': {'max_length': 1024, 'temperature': 0.7, 'guidance_scale': 6.0},
            'starcoder': {'max_length': 2048, 'temperature': 0.7, 'guidance_scale': 6.0},
            'codet5': {'max_length': 1024, 'temperature': 0.7, 'guidance_scale': 6.0}
        }
    
    def generate_creative(
        self,
        prompt: str,
        max_length: int = 200,
        num_candidates: int = 5,
        return_best: bool = True,
        sparsity_level: Optional[float] = None
    ) -> Union[str, List[str]]:
        """
        Generate creative content using the two-stage Creative-Sparse algorithm
        """
        if sparsity_level is None:
            sparsity_level = self._find_optimal_sparsity()
        
        # Stage A: Divergence Phase
        candidates = self._divergence_phase(
            prompt, max_length, num_candidates, 
            self.sparsity_schedule[0]  # High sparsity for exploration
        )
        
        # Stage B: Refinement Phase
        if return_best:
            best_candidate = self._refinement_phase(
                candidates, prompt, max_length,
                self.sparsity_schedule[1]  # Low sparsity for refinement
            )
            return best_candidate
        else:
            refined_candidates = []
            for candidate in candidates:
                refined = self._refinement_phase(
                    [candidate], prompt, max_length,
                    self.sparsity_schedule[1]
                )
                refined_candidates.append(refined)
            return refined_candidates
    
    def _divergence_phase(
        self, 
        prompt: str, 
        max_length: int, 
        num_candidates: int,
        sparsity: float
    ) -> List[str]:
        """Stage A: High sparsity for exploration"""
        sparsity_params = self.sparsity_controller.get_sparsity_params(sparsity)
        
        # Generate multiple candidates with high sparsity
        candidates = []
        for _ in range(num_candidates):
            # In practice, this would use the actual model with sparse attention
            candidate = self._generate_with_sparsity(prompt, max_length, sparsity_params)
            candidates.append(candidate)
        
        return candidates
    
    def _refinement_phase(
        self, 
        candidates: List[str], 
        prompt: str, 
        max_length: int,
        sparsity: float
    ) -> str:
        """Stage B: Low sparsity for refinement"""
        sparsity_params = self.sparsity_controller.get_sparsity_params(sparsity)
        
        # Score candidates and select best
        scores = []
        for candidate in candidates:
            score = self.quality_functional.compute_quality_score([candidate])
            scores.append(score)
        
        best_idx = np.argmax(scores)
        best_candidate = candidates[best_idx]
        
        # Refine the best candidate with low sparsity
        refined = self._generate_with_sparsity(
            f"{prompt} {best_candidate}", 
            max_length, 
            sparsity_params
        )
        
        return refined
    
    def _generate_with_sparsity(
        self, 
        prompt: str, 
        max_length: int, 
        sparsity_params: Dict[str, float]
    ) -> str:
        """Generate text with given sparsity parameters"""
        # This is a simplified implementation
        # In practice, this would integrate with the actual model and use sparse attention
        
        # For demonstration, return a mock creative output
        creative_templates = [
            f"In a world where {prompt.lower()}, the possibilities are endless.",
            f"The story begins when {prompt.lower()}, leading to unexpected discoveries.",
            f"Imagine a reality where {prompt.lower()}, and everything changes.",
            f"Deep in the realm of {prompt.lower()}, magic and wonder await.",
            f"Beyond the ordinary lies {prompt.lower()}, a journey of infinite potential."
        ]
        
        # Select template based on sparsity (higher sparsity = more creative)
        template_idx = int(sparsity_params['simthreshd1'] * len(creative_templates)) % len(creative_templates)
        return creative_templates[template_idx]
    
    def _find_optimal_sparsity(self) -> float:
        """Find optimal sparsity level for the current model and task"""
        # Simplified implementation - in practice, this would use the theoretical bounds
        model_config = self.model_configs.get(self.model_name, {})
        
        # Use model-specific optimal sparsity
        optimal_sparsities = {
            'gpt2-large': 0.63,
            'llama-2-7b': 0.58,
            'cogvideox': 0.55,
            'flux': 0.54
        }
        
        return optimal_sparsities.get(self.model_name, 0.6)
    
    def apply_sparse_attention(
        self, 
        q: torch.Tensor, 
        k: torch.Tensor, 
        v: torch.Tensor,
        sparsity: float,
        is_causal: bool = False
    ) -> torch.Tensor:
        """Apply sparse attention using SpargeAttention"""
        sparsity_params = self.sparsity_controller.get_sparsity_params(sparsity)
        
        return spas_sage2_attn_meansim_cuda(
            q, k, v,
            simthreshd1=sparsity_params['simthreshd1'],
            cdfthreshd=sparsity_params['cdfthreshd'],
            pvthreshd=int(sparsity_params['pvthreshd']),
            is_causal=is_causal
        )


def find_optimal_sparsity(
    model_name: str,
    task_type: str,
    compute_budget: float,
    quality_weights: Optional[QualityWeights] = None
) -> float:
    """
    Find optimal sparsity level for given model, task, and compute budget
    
    Args:
        model_name: Name of the model
        task_type: Type of task (creative_writing, image_generation, etc.)
        compute_budget: Available compute budget (0.0 to 1.0)
        quality_weights: Weights for quality functional
    
    Returns:
        Optimal sparsity level
    """
    # Task-specific optimal sparsities based on paper results
    task_sparsities = {
        # Text generation tasks
        'creative_writing': 0.63,
        'poetry': 0.60,
        'ad_copy': 0.65,
        'story_generation': 0.62,
        'dialogue_generation': 0.58,
        
        # Image generation tasks
        'image_generation': 0.55,
        'artistic_generation': 0.57,
        'style_transfer': 0.52,
        'photorealistic_generation': 0.50,
        
        # Video generation tasks
        'video_generation': 0.58,
        'creative_video': 0.60,
        'motion_generation': 0.56,
        'scene_generation': 0.54,
        
        # Music generation tasks
        'music_composition': 0.58,
        'jazz_composition': 0.60,
        'classical_composition': 0.55,
        'electronic_music': 0.62,
        
        # Code generation tasks
        'code_generation': 0.51,
        'creative_programming': 0.53,
        'algorithm_design': 0.49,
        'debugging': 0.48
    }
    
    base_sparsity = task_sparsities.get(task_type, 0.6)
    
    # Adjust based on compute budget
    if compute_budget < 0.5:
        # Lower compute budget -> higher sparsity for efficiency
        return min(0.9, base_sparsity + 0.2)
    elif compute_budget > 0.8:
        # Higher compute budget -> lower sparsity for quality
        return max(0.3, base_sparsity - 0.1)
    else:
        return base_sparsity


def compute_quality_score(
    outputs: List[str],
    novelty_weight: float = 0.4,
    diversity_weight: float = 0.3,
    coherence_weight: float = 0.3,
    reference_data: Optional[List[str]] = None
) -> List[float]:
    """
    Compute quality scores for a list of outputs
    
    Args:
        outputs: List of generated outputs
        novelty_weight: Weight for novelty component
        diversity_weight: Weight for diversity component
        coherence_weight: Weight for coherence component
        reference_data: Reference data for novelty computation
    
    Returns:
        List of quality scores
    """
    weights = QualityWeights(novelty_weight, diversity_weight, coherence_weight)
    quality_func = QualityFunctional(weights)
    
    scores = []
    for output in outputs:
        score = quality_func.compute_quality_score([output], reference_data)
        scores.append(score)
    
    return scores
