"""
Topic Intake Node
Normalizes incoming topic and sets up initial state
"""

import re
from typing import Dict, Any
from state import GraphState, TopicIntakeInput


def parse_target_distribution(target_string: str) -> Dict[str, float]:
    """Parse target distribution string like 'Left=0.5,Right=0.5'"""
    if not target_string:
        return {"Left": 0.5, "Right": 0.5}
    
    try:
        distribution = {}
        pairs = target_string.split(',')
        
        for pair in pairs:
            if '=' in pair:
                stance, value = pair.split('=')
                distribution[stance.strip()] = float(value.strip())
        
        # Normalize to sum to 1.0
        total = sum(distribution.values())
        if total > 0:
            distribution = {k: v/total for k, v in distribution.items()}
        
        return distribution
        
    except Exception as e:
        print(f"Warning: Could not parse target distribution '{target_string}': {e}")
        return {"Left": 0.5, "Right": 0.5}


def normalize_topic(raw_topic: str) -> str:
    """Normalize topic to single sentence"""
    # Remove extra whitespace and newlines
    topic = re.sub(r'\s+', ' ', raw_topic.strip())
    
    # Ensure it ends with appropriate punctuation
    if not topic.endswith(('.', '?', '!')):
        topic += '.'
    
    # Capitalize first letter
    if topic:
        topic = topic[0].upper() + topic[1:]
    
    return topic


def topic_intake(state: GraphState) -> GraphState:
    """
    Normalize incoming topic and set up initial parameters
    
    Args:
        state: Initial graph state with raw topic information
        
    Returns:
        Updated state with normalized parameters
    """
    try:
        print("📝 Topic Intake & Normalization")
        print("=" * 40)
        
        # Normalize the topic
        if state.topic:
            normalized_topic = normalize_topic(state.topic)
            state.topic = normalized_topic
            print(f"Topic: {normalized_topic}")
        else:
            # Default topic for testing
            state.topic = "The benefits and drawbacks of remote work policies."
            print(f"Topic (default): {state.topic}")
        
        # Set defaults if not already specified
        if not hasattr(state, 'audience') or not state.audience:
            state.audience = "general US reader"
        print(f"Audience: {state.audience}")
        
        if not hasattr(state, 'length') or not state.length:
            state.length = 750
        print(f"Target length: {state.length} words")
        
        # Ensure target distribution is set
        if not state.target_distribution:
            state.target_distribution = {"Left": 0.5, "Right": 0.5}
        print(f"Target distribution: {state.target_distribution}")
        
        # Initialize constraints if not set
        default_constraints = {
            "grade_min": 10,
            "grade_max": 13,
            "frame_entropy_min": 0.6,
            "bias_delta_max": 0.05,
            "max_passes": 3
        }
        
        if not state.constraints:
            state.constraints = default_constraints
        else:
            # Fill in missing constraints with defaults
            for key, value in default_constraints.items():
                if key not in state.constraints:
                    state.constraints[key] = value
        
        print(f"Constraints: {state.constraints}")
        
        # Initialize run metadata
        import uuid
        from datetime import datetime
        
        if not state.run_id:
            state.run_id = str(uuid.uuid4())[:8]
        
        if not hasattr(state, 'start_time') or not state.start_time:
            state.start_time = datetime.now()
        
        print(f"Run ID: {state.run_id}")
        print("✅ Topic intake complete")
        
        return state
        
    except Exception as e:
        print(f"❌ Error in topic intake: {e}")
        # Set safe defaults
        state.topic = "The benefits and drawbacks of remote work policies."
        state.audience = "general US reader"
        state.length = 750
        state.target_distribution = {"Left": 0.5, "Right": 0.5}
        state.constraints = {
            "grade_min": 10,
            "grade_max": 13,
            "frame_entropy_min": 0.6,
            "bias_delta_max": 0.05,
            "max_passes": 3
        }
        
        import uuid
        state.run_id = str(uuid.uuid4())[:8]
        
        return state


def create_intake_from_args(topic: str, 
                          audience: str = None,
                          length: int = None,
                          target_stance: str = None) -> GraphState:
    """
    Create initial graph state from command line arguments
    
    Args:
        topic: The opinion topic to write about
        audience: Target audience (default: "general US reader")
        length: Target word count (default: 750)
        target_stance: Target distribution string like "Left=0.5,Right=0.5"
        
    Returns:
        Initialized GraphState ready for processing
    """
    state = GraphState(topic=topic)
    
    if audience:
        state.audience = audience
    
    if length:
        state.length = length
    
    if target_stance:
        state.target_distribution = parse_target_distribution(target_stance)
    
    return topic_intake(state)
