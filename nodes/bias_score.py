"""
Bias Scorer Node
Simple baseline bias detection using keyword-based approach
"""

import re
import math
from typing import Dict, List
from collections import Counter
from state import GraphState


class BiasScorer:
    """Simple bias detection using keyword lexicons"""
    
    def __init__(self):
        """Initialize with political stance lexicons"""
        # These are simplified lexicons - could be expanded with more sophisticated approaches
        self.stance_lexicons = {
            'Left': [
                'progressive', 'liberal', 'equality', 'social justice', 'workers', 
                'union', 'regulation', 'government program', 'public option',
                'climate change', 'diversity', 'inclusion', 'redistribution',
                'minimum wage', 'universal healthcare', 'tax the rich',
                'civil rights', 'environment', 'renewable', 'sustainable'
            ],
            'Right': [
                'conservative', 'traditional', 'family values', 'free market',
                'small government', 'deregulation', 'private sector', 'business',
                'individual responsibility', 'law and order', 'strong defense',
                'fiscal responsibility', 'lower taxes', 'constitutional',
                'liberty', 'freedom', 'personal choice', 'private property'
            ],
            'Center': [
                'moderate', 'balanced', 'pragmatic', 'compromise', 'bipartisan',
                'middle ground', 'reasonable', 'practical', 'evidence-based',
                'both sides', 'trade-offs', 'nuanced', 'complex', 'measured'
            ]
        }
        
        # Loaded language that indicates bias
        self.loaded_terms = [
            'radical', 'extreme', 'dangerous', 'outrageous', 'ridiculous',
            'absurd', 'insane', 'crazy', 'stupid', 'idiotic', 'disaster',
            'catastrophe', 'crisis', 'urgent', 'critical', 'emergency'
        ]
    
    def detect_stance_signals(self, text: str) -> Dict[str, float]:
        """Detect stance-indicating language in text"""
        text_lower = text.lower()
        stance_counts = {}
        
        # Count stance keywords
        for stance, keywords in self.stance_lexicons.items():
            count = 0
            for keyword in keywords:
                # Use word boundaries to avoid partial matches
                pattern = r'\b' + re.escape(keyword) + r'\b'
                count += len(re.findall(pattern, text_lower))
            stance_counts[stance] = count
        
        # Convert to probabilities with smoothing
        total_signals = sum(stance_counts.values())
        
        if total_signals == 0:
            # No clear signals - assume balanced
            return {'Left': 0.33, 'Center': 0.34, 'Right': 0.33}
        
        # Add small smoothing to avoid zero probabilities
        smoothing = 0.1
        stance_probs = {}
        for stance, count in stance_counts.items():
            stance_probs[stance] = (count + smoothing) / (total_signals + smoothing * len(stance_counts))
        
        # Normalize to sum to 1
        total_prob = sum(stance_probs.values())
        stance_probs = {stance: prob / total_prob for stance, prob in stance_probs.items()}
        
        return stance_probs
    
    def compute_bias_delta(self, predicted: Dict[str, float], target: Dict[str, float]) -> float:
        """Compute deviation from target distribution"""
        # Only compare stances that exist in both distributions
        common_stances = set(predicted.keys()) & set(target.keys())
        
        if not common_stances:
            return 1.0  # Maximum bias if no overlap
        
        # Compute L1 distance (Manhattan distance)
        delta = sum(abs(predicted.get(stance, 0) - target.get(stance, 0)) 
                   for stance in common_stances)
        
        return delta / 2  # Normalize to [0, 1] range
    
    def detect_loaded_language(self, text: str) -> int:
        """Count instances of loaded/biased language"""
        text_lower = text.lower()
        loaded_count = 0
        
        for term in self.loaded_terms:
            pattern = r'\b' + re.escape(term) + r'\b'
            loaded_count += len(re.findall(pattern, text_lower))
        
        return loaded_count
    
    def analyze_bias(self, text: str, target_distribution: Dict[str, float]) -> tuple[Dict[str, float], float]:
        """Analyze text bias and compute delta from target"""
        stance_probs = self.detect_stance_signals(text)
        bias_delta = self.compute_bias_delta(stance_probs, target_distribution)
        
        return stance_probs, bias_delta


# Global scorer instance
_bias_scorer = None


def get_bias_scorer() -> BiasScorer:
    """Get or create global bias scorer instance"""
    global _bias_scorer
    if _bias_scorer is None:
        _bias_scorer = BiasScorer()
    return _bias_scorer


def bias_score(state: GraphState) -> GraphState:
    """
    Compute bias score for the current draft
    
    Args:
        state: Current graph state with draft text
        
    Returns:
        Updated state with bias metrics
    """
    if not state.draft:
        print("Warning: No draft available for bias scoring")
        return state
    
    try:
        scorer = get_bias_scorer()
        bias_probs, bias_delta = scorer.analyze_bias(state.draft, state.target_distribution)
        
        # Initialize metrics if not present
        if state.metrics is None:
            from state import Metrics
            state.metrics = Metrics(
                bias_probs=bias_probs,
                bias_target=state.target_distribution,
                bias_delta=bias_delta,
                frame_distribution={},
                frame_entropy=0.0,
                flesch_kincaid=12.0,
                dale_chall=8.0,
                coherence_score=0.7
            )
        else:
            # Update existing metrics
            state.metrics.bias_probs = bias_probs
            state.metrics.bias_target = state.target_distribution
            state.metrics.bias_delta = bias_delta
        
        print(f"⚖️  Bias Analysis:")
        print(f"   Detected: {bias_probs}")
        print(f"   Target: {state.target_distribution}")
        print(f"   Delta: {bias_delta:.3f}")
        
        # Check for loaded language
        loaded_count = scorer.detect_loaded_language(state.draft)
        if loaded_count > 0:
            print(f"   ⚠️  Loaded language detected: {loaded_count} instances")
        
        return state
        
    except Exception as e:
        print(f"Error computing bias score: {e}")
        # Return state with default values
        if state.metrics is None:
            from state import Metrics
            state.metrics = Metrics(
                bias_probs={'Left': 0.33, 'Center': 0.34, 'Right': 0.33},
                bias_target=state.target_distribution,
                bias_delta=0.1,  # small default bias
                frame_distribution={},
                frame_entropy=0.0,
                flesch_kincaid=12.0,
                dale_chall=8.0,
                coherence_score=0.7
            )
        else:
            state.metrics.bias_probs = {'Left': 0.33, 'Center': 0.34, 'Right': 0.33}
            state.metrics.bias_target = state.target_distribution
            state.metrics.bias_delta = 0.1
        
        return state
