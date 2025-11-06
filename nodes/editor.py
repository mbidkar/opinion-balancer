"""
Editor Node
Applies edit instructions to improve the draft using OpenAI API
"""

import yaml
from state import GraphState
from llm_client_openai import OpenAILLMClient


def load_prompts(config_path: str = "prompts.yaml") -> dict:
    """Load prompt templates"""
    try:
        with open(config_path, 'r') as f:
            return yaml.safe_load(f)
    except Exception as e:
        print(f"Warning: Could not load prompts from {config_path}: {e}")
        return {}


def load_config(config_path: str = "config.yaml") -> dict:
    """Load configuration settings"""
    try:
        with open(config_path, 'r') as f:
            return yaml.safe_load(f)
    except Exception as e:
        print(f"Warning: Could not load config from {config_path}: {e}")
        return {}


def format_editor_prompt(state: GraphState, prompts: dict) -> str:
    """Format the editor prompt with current draft and critique"""
    template = prompts.get('prompts', {}).get('editor', 
        "Apply these edits to the draft:\n\nDRAFT:\n{draft}\n\nEDITS:\n{critique}")
    
    formatted_prompt = template.format(
        draft=state.draft,
        critique=state.critique,
        target_distribution=state.target_distribution
    )
    
    return formatted_prompt


def editor(state: GraphState) -> GraphState:
    """
    Apply edit instructions to improve the current draft
    
    Args:
        state: Current graph state with draft and critique
        
    Returns:
        Updated state with edited draft
    """
    try:
        print("✏️  Editor (Applying Edits)")
        print("=" * 40)
        
        if not state.draft:
            print("❌ No draft available for editing")
            return state
        
        if not state.critique:
            print("⚠️  No critique available - returning original draft")
            return state
        
        # Load prompts and config
        prompts = load_prompts()
        config = load_config()
        
        # Get model settings from config
        model_config = config.get('openai', {})
        model_name = model_config.get('model', 'gpt-4')
        temperature = 0.7
        max_tokens = 1000
        
        llm_client = OpenAILLMClient(model=model_name, temperature=temperature, max_completion_tokens=max_tokens)
        
        # Test LLM connection
        if not llm_client.test_connection():
            print("❌ Cannot connect to OpenAI API for editing")
            # Return original draft if can't edit
            return state
        
        print("Applying edits...")
        
        # Format the editor prompt
        prompt = format_editor_prompt(state, prompts)
        
        # Generate edited draft
        edited_draft = llm_client.generate(
            prompt=prompt,
            system_message="You are an expert editor focused on applying specific edits while maintaining quality and balance."
        )
        
        if not edited_draft or len(edited_draft.strip()) < 100:
            print("❌ Edited draft too short, keeping original")
            return state
        
        # Compare word counts
        original_words = len(state.draft.split())
        edited_words = len(edited_draft.split())
        
        print(f"📊 Word count: {original_words} → {edited_words} ({edited_words - original_words:+d})")
        
        # Update state with edited draft
        state.draft = edited_draft.strip()
        
        print("✅ Edits applied successfully")
        return state
        
    except Exception as e:
        print(f"❌ Error in editor: {e}")
        # Return original state if editing fails
        return state


def analyze_edits_applied(original: str, edited: str) -> dict:
    """Analyze what changes were made during editing"""
    original_words = original.split()
    edited_words = edited.split()
    
    # Simple change analysis
    word_count_change = len(edited_words) - len(original_words)
    
    # Paragraph count change
    original_paras = len([p for p in original.split('\n\n') if p.strip()])
    edited_paras = len([p for p in edited.split('\n\n') if p.strip()])
    paragraph_change = edited_paras - original_paras
    
    # Character-level similarity (rough estimate)
    similarity = calculate_similarity(original, edited)
    
    return {
        'word_count_change': word_count_change,
        'paragraph_change': paragraph_change,
        'similarity': similarity,
        'substantial_change': similarity < 0.8  # 80% similarity threshold
    }


def calculate_similarity(text1: str, text2: str) -> float:
    """Calculate simple character-level similarity between texts"""
    if not text1 or not text2:
        return 0.0
    
    # Simple Jaccard similarity on character 3-grams
    def get_ngrams(text: str, n: int = 3) -> set:
        text = text.lower().replace(' ', '')
        return set(text[i:i+n] for i in range(len(text) - n + 1))
    
    ngrams1 = get_ngrams(text1)
    ngrams2 = get_ngrams(text2)
    
    if not ngrams1 and not ngrams2:
        return 1.0
    
    intersection = ngrams1 & ngrams2
    union = ngrams1 | ngrams2
    
    return len(intersection) / len(union) if union else 0.0


def validate_edit_quality(original: str, edited: str, target_length: int) -> dict:
    """Validate that edits maintain quality standards"""
    issues = []
    
    # Check for dramatic length changes
    original_words = len(original.split())
    edited_words = len(edited.split())
    length_change_pct = abs(edited_words - original_words) / original_words * 100
    
    if length_change_pct > 50:
        issues.append(f"Dramatic length change: {length_change_pct:.1f}%")
    
    # Check for target length deviation
    target_deviation = abs(edited_words - target_length) / target_length * 100
    if target_deviation > 30:
        issues.append(f"Target length deviation: {target_deviation:.1f}%")
    
    # Check for structure preservation (paragraph count)
    original_paras = len([p for p in original.split('\n\n') if p.strip()])
    edited_paras = len([p for p in edited.split('\n\n') if p.strip()])
    
    if abs(edited_paras - original_paras) > 2:
        issues.append(f"Structure change: {original_paras} → {edited_paras} paragraphs")
    
    # Check for content preservation
    similarity = calculate_similarity(original, edited)
    if similarity < 0.5:
        issues.append(f"Low content similarity: {similarity:.2f}")
    
    return {
        'valid': len(issues) == 0,
        'issues': issues,
        'similarity': similarity,
        'length_change_pct': length_change_pct
    }
