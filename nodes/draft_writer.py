"""
Draft Writer Node
Uses OpenAI API to generate balanced opinion drafts
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


def format_draft_prompt(state: GraphState, prompts: dict) -> str:
    """Format the draft writer prompt with current state"""
    template = prompts.get('prompts', {}).get('draft_writer', "Write a balanced opinion piece on: {topic}")
    
    # Prepare sections for dynamic insertion
    if state.pass_count > 0 and state.draft:
        previous_draft_section = prompts.get('template_sections', {}).get('previous_draft', 
            "- Previous draft (pass {pass_id}):\n{previous_draft}").format(
            pass_id=state.pass_count,
            previous_draft=state.draft
        )
    else:
        previous_draft_section = prompts.get('template_sections', {}).get('no_previous_draft',
            "- This is the initial draft (no previous version)")
    
    # Format edit instructions if available
    if state.critique:
        edit_instructions_section = prompts.get('template_sections', {}).get('edit_instructions',
            "SPECIFIC EDITS TO APPLY:\n{edit_instructions}").format(
            edit_instructions=state.critique
        )
    else:
        edit_instructions_section = prompts.get('template_sections', {}).get('no_edit_instructions',
            "- Write a fresh, balanced draft on the topic")
    
    # Fill in the main template
    formatted_prompt = template.format(
        topic=state.topic,
        audience=state.audience,
        length=state.length,
        target_distribution=state.target_distribution,
        pass_count=state.pass_count + 1,  # Next pass number
        previous_draft_section=previous_draft_section,
        edit_instructions_section=edit_instructions_section
    )
    
    return formatted_prompt


def draft_writer(state: GraphState) -> GraphState:
    """
    Generate or revise an opinion draft using Ollama LLM
    
    Args:
        state: Current graph state with topic and parameters
        
    Returns:
        Updated state with new draft
    """
    try:
        print(f"✍️  Draft Writer (Pass {state.pass_count + 1})")
        print("=" * 40)
        
        # Load prompts and configuration
        prompts = load_prompts()
        
        # Load model configuration from config.yaml
        try:
            with open("config.yaml", 'r') as f:
                config = yaml.safe_load(f)
            model_config = config.get('models', {})
            model_name = model_config.get('model', 'gpt-5-nano')
            temperature = model_config.get('temperatures', {}).get('writer', 0.8)
            max_tokens = model_config.get('max_completion_tokens', 1000)
        except:
            # Fallback defaults
            model_name = 'gpt-5-nano'
            temperature = 0.8
            max_tokens = 1000
        
        llm_client = OpenAILLMClient(model=model_name, temperature=temperature, max_completion_tokens=max_tokens)
        
        # Test LLM connection
        if not llm_client.test_connection():
            print("❌ Cannot connect to OpenAI API. Check your API key and internet connection.")
            # Provide a fallback draft
            state.draft = create_fallback_draft(state)
            return state
        
        # Format the prompt
        prompt = format_draft_prompt(state, prompts)
        
        print("Generating draft...")
        
        # Generate the draft
        draft = llm_client.generate(
            prompt=prompt,
            system_message="You are an expert opinion writer focused on balanced, fair reporting."
        )
        
        print(f"🔍 Generated draft length: {len(draft.strip()) if draft else 0} characters")
        if draft:
            print(f"🔍 Draft preview: {draft[:200]}...")
        
        if not draft or len(draft.strip()) < 100:
            print("❌ Generated draft too short, using fallback")
            draft = create_fallback_draft(state)
        
        # Update state
        state.draft = draft.strip()
        
        # Word count check
        word_count = len(state.draft.split())
        print(f"📊 Draft generated: {word_count} words")
        
        if abs(word_count - state.length) > 200:
            print(f"⚠️  Word count deviation: target {state.length}, actual {word_count}")
        
        print("✅ Draft complete")
        return state
        
    except Exception as e:
        print(f"❌ Error in draft writer: {e}")
        # Provide fallback draft
        state.draft = create_fallback_draft(state)
        return state


def create_fallback_draft(state: GraphState) -> str:
    """Create a simple fallback draft when LLM fails"""
    return f"""The topic of {state.topic.lower()} presents multiple perspectives worth considering.

From one viewpoint, supporters argue that this approach offers significant benefits including improved outcomes and greater efficiency. They point to evidence suggesting positive impacts on stakeholders and alignment with important values.

However, critics raise valid concerns about potential drawbacks and unintended consequences. They emphasize the importance of considering alternative approaches and question whether the proposed benefits outweigh the costs.

A more moderate perspective suggests that the reality likely lies somewhere between these positions. Both sides make valid points that deserve careful consideration.

The complexity of this issue means that simple solutions are unlikely to address all concerns. Moving forward will require thoughtful dialogue, evidence-based analysis, and willingness to find common ground among different viewpoints.

Ultimately, the best path forward may involve elements from multiple perspectives, adapted to specific circumstances and community needs."""


def count_words(text: str) -> int:
    """Count words in text"""
    return len(text.split())


def extract_draft_structure(draft: str) -> dict:
    """Analyze draft structure for debugging"""
    paragraphs = [p.strip() for p in draft.split('\n\n') if p.strip()]
    
    return {
        'total_paragraphs': len(paragraphs),
        'words_per_paragraph': [len(p.split()) for p in paragraphs],
        'total_words': count_words(draft),
        'avg_paragraph_length': sum(len(p.split()) for p in paragraphs) / len(paragraphs) if paragraphs else 0
    }
