"""
Coherence Scorer Node
Uses sentence embeddings to compute paragraph coherence
"""

import numpy as np
from sentence_transformers import SentenceTransformer
from sklearn.metrics.pairwise import cosine_similarity
from typing import List
from state import GraphState


class CoherenceScorer:
    """Coherence evaluation using sentence embeddings"""
    
    def __init__(self, model_name: str = "all-MiniLM-L6-v2"):
        """Initialize with a lightweight sentence transformer model"""
        try:
            self.model = SentenceTransformer(model_name)
        except Exception as e:
            print(f"Warning: Could not load {model_name}, using simpler fallback")
            # Fallback to a simpler approach if model loading fails
            self.model = None
    
    def split_into_paragraphs(self, text: str) -> List[str]:
        """Split text into paragraphs"""
        paragraphs = [p.strip() for p in text.split('\n\n') if p.strip()]
        return [p for p in paragraphs if len(p) > 20]  # Filter very short paragraphs
    
    def compute_coherence(self, text: str) -> float:
        """Compute coherence score from paragraph embeddings"""
        if not self.model:
            # Fallback: simple lexical overlap measure
            return self._simple_coherence_fallback(text)
        
        try:
            paragraphs = self.split_into_paragraphs(text)
            
            if len(paragraphs) < 2:
                return 1.0  # Single paragraph is perfectly coherent
            
            # Get embeddings for each paragraph
            embeddings = self.model.encode(paragraphs)
            
            # Compute pairwise cosine similarities
            similarities = cosine_similarity(embeddings)
            
            # Get mean similarity excluding diagonal (self-similarity)
            mask = np.ones_like(similarities, dtype=bool)
            np.fill_diagonal(mask, False)
            mean_similarity = similarities[mask].mean()
            
            # Ensure result is between 0 and 1
            return max(0.0, min(1.0, mean_similarity))
            
        except Exception as e:
            print(f"Error computing coherence: {e}")
            return self._simple_coherence_fallback(text)
    
    def _simple_coherence_fallback(self, text: str) -> float:
        """Simple lexical overlap fallback when embeddings fail"""
        paragraphs = self.split_into_paragraphs(text)
        
        if len(paragraphs) < 2:
            return 1.0
        
        total_overlap = 0
        comparisons = 0
        
        for i in range(len(paragraphs)):
            for j in range(i + 1, len(paragraphs)):
                words_i = set(paragraphs[i].lower().split())
                words_j = set(paragraphs[j].lower().split())
                
                if len(words_i) > 0 and len(words_j) > 0:
                    overlap = len(words_i & words_j) / len(words_i | words_j)
                    total_overlap += overlap
                    comparisons += 1
        
        return total_overlap / comparisons if comparisons > 0 else 0.5


# Global scorer instance
_coherence_scorer = None


def get_coherence_scorer() -> CoherenceScorer:
    """Get or create global coherence scorer instance"""
    global _coherence_scorer
    if _coherence_scorer is None:
        _coherence_scorer = CoherenceScorer()
    return _coherence_scorer


def coherence(state: GraphState) -> GraphState:
    """
    Compute coherence score for the current draft
    
    Args:
        state: Current graph state with draft text
        
    Returns:
        Updated state with coherence metrics
    """
    if not state.draft:
        print("Warning: No draft available for coherence scoring")
        return state
    
    try:
        scorer = get_coherence_scorer()
        coherence_score = scorer.compute_coherence(state.draft)
        
        # Initialize metrics if not present
        if state.metrics is None:
            from state import Metrics
            state.metrics = Metrics(
                bias_probs={},
                bias_target=state.target_distribution,
                bias_delta=0.0,
                frame_distribution={},
                frame_entropy=0.0,
                flesch_kincaid=12.0,
                dale_chall=8.0,
                coherence_score=coherence_score
            )
        else:
            # Update existing metrics
            state.metrics.coherence_score = coherence_score
        
        print(f"🔗 Coherence Score: {coherence_score:.3f}")
        
        return state
        
    except Exception as e:
        print(f"Error computing coherence: {e}")
        # Return state with default coherence
        if state.metrics is None:
            from state import Metrics
            state.metrics = Metrics(
                bias_probs={},
                bias_target=state.target_distribution,
                bias_delta=0.0,
                frame_distribution={},
                frame_entropy=0.0,
                flesch_kincaid=12.0,
                dale_chall=8.0,
                coherence_score=0.7  # default coherence
            )
        else:
            state.metrics.coherence_score = 0.7
        
        return state
