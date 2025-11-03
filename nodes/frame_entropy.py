"""
Frame Entropy Scorer Node
Identifies framing categories and computes Shannon entropy
"""

import re
import math
from typing import Dict, List
from collections import Counter
from state import GraphState


class FrameEntropyScorer:
    """Frame detection and entropy calculation"""
    
    def __init__(self):
        """Initialize with frame detection keywords"""
        self.frame_lexicons = {
            'moral': [
                'right', 'wrong', 'ethical', 'moral', 'values', 'principle', 'duty',
                'virtue', 'justice', 'fairness', 'integrity', 'responsibility',
                'good', 'evil', 'righteous', 'sin', 'conscience'
            ],
            'economic': [
                'cost', 'benefit', 'profit', 'loss', 'price', 'value', 'money',
                'economic', 'financial', 'budget', 'investment', 'market',
                'tax', 'income', 'wealth', 'poverty', 'expensive', 'cheap'
            ],
            'policy': [
                'policy', 'law', 'regulation', 'government', 'legislation',
                'rule', 'requirement', 'mandate', 'ban', 'allow', 'permit',
                'legal', 'illegal', 'congress', 'senate', 'vote', 'bill'
            ],
            'conflict': [
                'conflict', 'fight', 'battle', 'oppose', 'against', 'versus',
                'enemy', 'threat', 'danger', 'attack', 'defend', 'protect',
                'struggle', 'compete', 'war', 'peace', 'negotiate'
            ],
            'human_interest': [
                'people', 'individual', 'person', 'family', 'community',
                'story', 'experience', 'life', 'human', 'personal',
                'emotion', 'feel', 'suffer', 'hope', 'dream', 'struggle'
            ],
            'consequence': [
                'result', 'outcome', 'consequence', 'effect', 'impact',
                'lead to', 'cause', 'because', 'therefore', 'thus',
                'implication', 'future', 'long-term', 'short-term'
            ],
            'attribution': [
                'blame', 'fault', 'responsible', 'credit', 'due to',
                'caused by', 'thanks to', 'because of', 'reason',
                'explanation', 'why', 'who', 'accountability'
            ]
        }
    
    def detect_frames(self, text: str) -> Dict[str, float]:
        """Detect frame usage in text"""
        text_lower = text.lower()
        frame_counts = {}
        
        # Count frame keywords
        for frame_name, keywords in self.frame_lexicons.items():
            count = 0
            for keyword in keywords:
                # Use word boundaries to avoid partial matches
                pattern = r'\b' + re.escape(keyword) + r'\b'
                count += len(re.findall(pattern, text_lower))
            frame_counts[frame_name] = count
        
        # Convert to proportions
        total_frame_words = sum(frame_counts.values())
        if total_frame_words == 0:
            # Equal distribution as fallback
            return {frame: 1.0 / len(self.frame_lexicons) 
                   for frame in self.frame_lexicons.keys()}
        
        frame_distribution = {
            frame: count / total_frame_words 
            for frame, count in frame_counts.items()
        }
        
        return frame_distribution
    
    def compute_shannon_entropy(self, distribution: Dict[str, float]) -> float:
        """Compute Shannon entropy of frame distribution"""
        # Filter out zero probabilities
        probs = [p for p in distribution.values() if p > 0]
        
        if not probs:
            return 0.0
        
        # Shannon entropy: H = -Σ(p * log2(p))
        entropy = -sum(p * math.log2(p) for p in probs)
        
        return entropy
    
    def analyze_frames(self, text: str) -> tuple[Dict[str, float], float]:
        """Analyze text and return frame distribution and entropy"""
        frame_distribution = self.detect_frames(text)
        entropy = self.compute_shannon_entropy(frame_distribution)
        
        return frame_distribution, entropy


# Global scorer instance
_frame_scorer = None


def get_frame_scorer() -> FrameEntropyScorer:
    """Get or create global frame scorer instance"""
    global _frame_scorer
    if _frame_scorer is None:
        _frame_scorer = FrameEntropyScorer()
    return _frame_scorer


def frame_entropy(state: GraphState) -> GraphState:
    """
    Compute frame entropy for the current draft
    
    Args:
        state: Current graph state with draft text
        
    Returns:
        Updated state with frame entropy metrics
    """
    if not state.draft:
        print("Warning: No draft available for frame entropy scoring")
        return state
    
    try:
        scorer = get_frame_scorer()
        frame_distribution, entropy = scorer.analyze_frames(state.draft)
        
        # Initialize metrics if not present
        if state.metrics is None:
            from state import Metrics
            state.metrics = Metrics(
                bias_probs={},
                bias_target=state.target_distribution,
                bias_delta=0.0,
                frame_distribution=frame_distribution,
                frame_entropy=entropy,
                flesch_kincaid=12.0,
                dale_chall=8.0,
                coherence_score=0.7
            )
        else:
            # Update existing metrics
            state.metrics.frame_distribution = frame_distribution
            state.metrics.frame_entropy = entropy
        
        print(f"🖼️  Frame Entropy: {entropy:.3f}")
        print(f"   Top frames: {dict(sorted(frame_distribution.items(), key=lambda x: x[1], reverse=True)[:3])}")
        
        return state
        
    except Exception as e:
        print(f"Error computing frame entropy: {e}")
        # Return state with default values
        if state.metrics is None:
            from state import Metrics
            default_frames = {frame: 1.0/7 for frame in ['moral', 'economic', 'policy', 'conflict', 'human_interest', 'consequence', 'attribution']}
            state.metrics = Metrics(
                bias_probs={},
                bias_target=state.target_distribution,
                bias_delta=0.0,
                frame_distribution=default_frames,
                frame_entropy=2.8,  # log2(7) for uniform distribution
                flesch_kincaid=12.0,
                dale_chall=8.0,
                coherence_score=0.7
            )
        else:
            default_frames = {frame: 1.0/7 for frame in ['moral', 'economic', 'policy', 'conflict', 'human_interest', 'consequence', 'attribution']}
            state.metrics.frame_distribution = default_frames
            state.metrics.frame_entropy = 2.8
        
        return state
