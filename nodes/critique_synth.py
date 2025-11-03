"""
Critique Synthesizer Node
Converts metrics into actionable edit instructions using Ollama LLM
"""

import yaml
from state import GraphState
from llm_client import make_llm_client


def load_prompts(config_path: str = "prompts.yaml") -> dict:
    """Load prompt templates"""
    try:
        with open(config_path, 'r') as f:
            return yaml.safe_load(f)
    except Exception as e:
        print(f"Warning: Could not load prompts from {config_path}: {e}")
        return {}


def format_critique_prompt(state: GraphState, prompts: dict) -> str:
    """Format the critique prompt with current metrics"""
    if not state.metrics:
        return "No metrics available for critique."
    
    template = prompts.get('prompts', {}).get('bias_critique_adapter', 
        "Analyze these metrics and provide edit suggestions: {metrics}")
    
    # Get draft excerpt (first 500 characters)
    draft_excerpt = state.draft[:500] + "..." if len(state.draft) > 500 else state.draft
    
    # Load constraints for targets
    bias_delta_max = state.constraints.get('bias_delta_max', 0.05)
    frame_entropy_min = state.constraints.get('frame_entropy_min', 0.6)
    
    formatted_prompt = template.format(
        bias_delta=state.metrics.bias_delta,
        bias_delta_max=bias_delta_max,
        bias_probs=state.metrics.bias_probs,
        target_distribution=state.target_distribution,
        frame_entropy=state.metrics.frame_entropy,
        frame_entropy_min=frame_entropy_min,
        flesch_kincaid=state.metrics.flesch_kincaid,
        dale_chall=state.metrics.dale_chall,
        coherence_score=state.metrics.coherence_score,
        draft_excerpt=draft_excerpt
    )
    
    return formatted_prompt


def parse_edit_instructions(critique: str) -> list:
    """Parse numbered edit instructions from critique"""
    import re
    
    # Find numbered items (1., 2., etc.)
    pattern = r'(\d+)\.\s*(.+?)(?=\d+\.|$)'
    matches = re.findall(pattern, critique, re.DOTALL)
    
    instructions = []
    for number, instruction in matches:
        instructions.append({
            'number': int(number),
            'instruction': instruction.strip()
        })
    
    return instructions


def critique_synth(state: GraphState) -> GraphState:
    """
    Generate critique and edit instructions based on current metrics
    
    Args:
        state: Current graph state with draft and metrics
        
    Returns:
        Updated state with critique instructions
    """
    try:
        print("🔍 Critique Synthesis")
        print("=" * 40)
        
        if not state.metrics:
            print("❌ No metrics available for critique")
            state.critique = "No specific edits needed - metrics not available."
            return state
        
        if not state.draft:
            print("❌ No draft available for critique")
            state.critique = "No draft available to critique."
            return state
        
        # Load prompts and LLM client
        prompts = load_prompts()
        llm_client = make_llm_client()
        
        # Test LLM connection
        if not llm_client.test_connection():
            print("❌ Cannot connect to Ollama for critique generation")
            state.critique = create_fallback_critique(state)
            return state
        
        # Analyze current metrics vs targets
        print("📊 Current Metrics:")
        print(f"   Bias delta: {state.metrics.bias_delta:.3f} (target: ≤{state.constraints.get('bias_delta_max', 0.05)})")
        print(f"   Frame entropy: {state.metrics.frame_entropy:.3f} (target: ≥{state.constraints.get('frame_entropy_min', 0.6)})")
        print(f"   Readability: FK={state.metrics.flesch_kincaid:.1f}, DC={state.metrics.dale_chall:.1f}")
        print(f"   Coherence: {state.metrics.coherence_score:.3f}")
        
        # Format the critique prompt
        prompt = format_critique_prompt(state, prompts)
        
        print("Generating critique...")
        
        # Generate critique
        critique = llm_client.generate(
            prompt=prompt,
            role="critic",
            system_message="You are an expert editor focused on creating balanced, readable content."
        )
        
        if not critique or len(critique.strip()) < 20:
            print("❌ Generated critique too short, using fallback")
            critique = create_fallback_critique(state)
        
        # Update state
        state.critique = critique.strip()
        
        # Parse and count edit instructions
        instructions = parse_edit_instructions(state.critique)
        print(f"📝 Generated {len(instructions)} edit instructions")
        
        print("✅ Critique synthesis complete")
        return state
        
    except Exception as e:
        print(f"❌ Error in critique synthesis: {e}")
        state.critique = create_fallback_critique(state)
        return state


def create_fallback_critique(state: GraphState) -> str:
    """Create fallback critique when LLM fails"""
    if not state.metrics:
        return "1. Review overall balance and ensure multiple perspectives are fairly represented."
    
    instructions = []
    
    # Check bias delta
    if state.metrics.bias_delta > state.constraints.get('bias_delta_max', 0.05):
        target_dist = state.target_distribution
        current_dist = state.metrics.bias_probs
        
        # Find which stance is over-represented
        max_stance = max(current_dist.items(), key=lambda x: x[1])
        instructions.append(f"1. [Overall]: Reduce {max_stance[0]} bias by balancing arguments. Current distribution: {current_dist}, Target: {target_dist}")
    
    # Check frame entropy
    if state.metrics.frame_entropy < state.constraints.get('frame_entropy_min', 0.6):
        instructions.append(f"2. [Throughout]: Increase frame diversity by incorporating more varied perspectives (economic, moral, policy, etc.). Current entropy: {state.metrics.frame_entropy:.3f}")
    
    # Check readability
    if state.metrics.flesch_kincaid > state.constraints.get('grade_max', 13):
        instructions.append(f"3. [Throughout]: Simplify language and sentence structure. Current grade level: {state.metrics.flesch_kincaid:.1f}")
    elif state.metrics.flesch_kincaid < state.constraints.get('grade_min', 10):
        instructions.append(f"3. [Throughout]: Add more sophisticated vocabulary and complex sentences. Current grade level: {state.metrics.flesch_kincaid:.1f}")
    
    # Check coherence
    if state.metrics.coherence_score < state.constraints.get('coherence_min', 0.7):
        instructions.append(f"4. [Transitions]: Improve paragraph connections and logical flow. Current coherence: {state.metrics.coherence_score:.3f}")
    
    if not instructions:
        instructions = ["1. [Overall]: Review for balance, clarity, and fair representation of multiple viewpoints."]
    
    return "\n".join(instructions)


def identify_problem_areas(state: GraphState) -> dict:
    """Identify which metrics need improvement"""
    problems = {}
    
    if not state.metrics:
        return problems
    
    constraints = state.constraints
    
    if state.metrics.bias_delta > constraints.get('bias_delta_max', 0.05):
        problems['bias'] = {
            'current': state.metrics.bias_delta,
            'target': constraints.get('bias_delta_max', 0.05),
            'severity': 'high' if state.metrics.bias_delta > 0.1 else 'medium'
        }
    
    if state.metrics.frame_entropy < constraints.get('frame_entropy_min', 0.6):
        problems['frame_diversity'] = {
            'current': state.metrics.frame_entropy,
            'target': constraints.get('frame_entropy_min', 0.6),
            'severity': 'high' if state.metrics.frame_entropy < 0.4 else 'medium'
        }
    
    grade_min = constraints.get('grade_min', 10)
    grade_max = constraints.get('grade_max', 13)
    
    if state.metrics.flesch_kincaid < grade_min or state.metrics.flesch_kincaid > grade_max:
        problems['readability'] = {
            'current': state.metrics.flesch_kincaid,
            'target_range': f"{grade_min}-{grade_max}",
            'severity': 'medium'
        }
    
    if state.metrics.coherence_score < constraints.get('coherence_min', 0.7):
        problems['coherence'] = {
            'current': state.metrics.coherence_score,
            'target': constraints.get('coherence_min', 0.7),
            'severity': 'medium'
        }
    
    return problems
