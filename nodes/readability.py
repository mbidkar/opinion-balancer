"""
Readability Scorer Node
Computes Flesch-Kincaid and Dale-Chall readability metrics
"""

import textstat
from typing import Dict
from state import GraphState


def readability(state: GraphState) -> GraphState:
    """
    Compute readability scores for the current draft
    
    Args:
        state: Current graph state with draft text
        
    Returns:
        Updated state with readability metrics
    """
    if not state.draft:
        print("Warning: No draft available for readability scoring")
        return state
    
    try:
        # Compute readability metrics
        flesch_kincaid = textstat.flesch_kincaid().flesch_kincaid_grade(state.draft)
        dale_chall = textstat.dale_chall_readability_score(state.draft)
        
        # Initialize metrics if not present
        if state.metrics is None:
            from state import Metrics
            state.metrics = Metrics(
                bias_probs={},
                bias_target=state.target_distribution,
                bias_delta=0.0,
                frame_distribution={},
                frame_entropy=0.0,
                flesch_kincaid=flesch_kincaid,
                dale_chall=dale_chall,
                coherence_score=0.0
            )
        else:
            # Update existing metrics
            state.metrics.flesch_kincaid = flesch_kincaid
            state.metrics.dale_chall = dale_chall
        
        print(f"📊 Readability Scores:")
        print(f"   Flesch-Kincaid Grade: {flesch_kincaid:.1f}")
        print(f"   Dale-Chall Score: {dale_chall:.1f}")
        
        return state
        
    except Exception as e:
        print(f"Error computing readability: {e}")
        # Return state with default values
        if state.metrics is None:
            from state import Metrics
            state.metrics = Metrics(
                bias_probs={},
                bias_target=state.target_distribution,
                bias_delta=0.0,
                frame_distribution={},
                frame_entropy=0.0,
                flesch_kincaid=12.0,  # default grade level
                dale_chall=8.0,       # default score
                coherence_score=0.0
            )
        else:
            state.metrics.flesch_kincaid = 12.0
            state.metrics.dale_chall = 8.0
        
        return state


def analyze_readability_detailed(text: str) -> Dict[str, float]:
    """Extended readability analysis for development/debugging"""
    return {
        'flesch_kincaid_grade': textstat.flesch_kincaid().flesch_kincaid_grade(text),
        'flesch_reading_ease': textstat.flesch_reading_ease(text),
        'dale_chall': textstat.dale_chall_readability_score(text),
        'gunning_fog': textstat.gunning_fog(text),
        'automated_readability': textstat.automated_readability_index(text),
        'avg_sentence_length': textstat.avg_sentence_length(text),
        'avg_syllables_per_word': textstat.avg_syllables_per_word(text)
    }
