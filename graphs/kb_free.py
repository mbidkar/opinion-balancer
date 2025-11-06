"""
KB-Free OpinionBalancer Graph
LangGraph workflow implementation for local opinion balancing
"""

from typing import Dict, Any
from langgraph.graph import StateGraph, END
from langgraph.graph.state import CompiledStateGraph

from state import GraphState
from nodes import (
    topic_intake, draft_writer, bias_score, frame_entropy,
    readability, coherence, critique_synth, editor, convergence_check, logger
)


def create_opinion_balancer_graph() -> CompiledStateGraph:
    """
    Create the complete OpinionBalancer workflow graph with iterative refinement
    
    Returns:
        Compiled LangGraph ready for execution with convergence loops
    """
    
    # Create the state graph
    graph = StateGraph(GraphState)
    
    # Add all nodes
    graph.add_node("topic_intake", topic_intake)
    graph.add_node("draft_writer", draft_writer)
    graph.add_node("bias_score", bias_score)
    graph.add_node("frame_entropy", frame_entropy)
    graph.add_node("readability", readability)
    graph.add_node("coherence", coherence)
    graph.add_node("critique_synth", critique_synth)
    graph.add_node("editor", editor)
    graph.add_node("convergence_check", convergence_check)
    graph.add_node("logger", logger)
    
    # Add coordination node for metrics evaluation
    graph.add_node("evaluate_all", evaluate_all_metrics)
    
    # Add pass increment node
    graph.add_node("increment_pass", increment_pass_counter)
    
    # Set entry point
    graph.set_entry_point("topic_intake")
    
    # Initial flow: intake → draft_writer → evaluate metrics
    graph.add_edge("topic_intake", "draft_writer")
    graph.add_edge("draft_writer", "evaluate_all")
    
    # After evaluation, synthesize critique and edit
    graph.add_edge("evaluate_all", "critique_synth")
    graph.add_edge("critique_synth", "editor")
    
    # After editing, increment pass counter
    graph.add_edge("editor", "increment_pass")
    
    # Check convergence after each pass
    graph.add_edge("increment_pass", "convergence_check")
    
    # Conditional routing based on convergence decision
    graph.add_conditional_edges(
        "convergence_check",
        route_convergence_decision,
        {
            "continue": "evaluate_all",  # Loop back to re-evaluate metrics
            "end": "logger"              # Finish and log results
        }
    )
    
    # Final logging and end
    graph.add_edge("logger", END)
    
    # Compile the graph
    return graph.compile()


def evaluate_all_metrics(state: GraphState) -> GraphState:
    """
    Coordinate node that runs all evaluators in sequence
    
    Args:
        state: Current graph state
        
    Returns:
        State with all metrics computed
    """
    try:
        print("📊 Evaluating All Metrics")
        print("=" * 40)
        
        # Run each evaluator in sequence
        # Note: We do this sequentially to avoid concurrency issues with state updates
        
        state = bias_score(state)
        state = frame_entropy(state)
        state = readability(state)
        state = coherence(state)
        
        print("✅ All metrics evaluated")
        return state
        
    except Exception as e:
        print(f"❌ Error in metric evaluation: {e}")
        return state


def increment_pass_counter(state: GraphState) -> GraphState:
    """
    Increment the pass counter and update history
    
    Args:
        state: Current graph state
        
    Returns:
        State with incremented pass counter and updated history
    """
    try:
        # Increment pass counter
        state.pass_count += 1
        
        # Create a pass log entry
        from state import PassLog
        from datetime import datetime
        
        # Only create pass log if we have metrics
        if state.metrics:
            pass_log = PassLog(
                pass_id=state.pass_count,
                draft=state.draft if state.draft else "",
                critique=state.critique if state.critique else "",
                metrics=state.metrics.model_copy(),
                timestamp=datetime.now()
            )
            
            # Add to history
            if state.history is None:
                state.history = []
            state.history.append(pass_log)
        
        print(f"📈 Pass {state.pass_count} completed")
        
        return state
        
    except Exception as e:
        print(f"❌ Error incrementing pass counter: {e}")
        return state


def route_convergence_decision(state: GraphState) -> str:
    """
    Route based on convergence check decision
    
    Args:
        state: Current graph state with convergence decision
        
    Returns:
        Next node name: "continue" or "end"
    """
    try:
        # The convergence_check node should have set the convergence status
        if hasattr(state, 'converged') and state.converged:
            print("🎯 Convergence achieved - ending refinement")
            return "end"
        else:
            print("🔄 Continuing refinement loop")
            return "continue"
            
    except Exception as e:
        print(f"❌ Error in routing decision: {e}")
        # Default to ending on error to prevent infinite loops
        return "end"


# Removed routing function - using linear flow only


def run_opinion_balancer(
    topic: str,
    audience: str = "general US reader",
    length: int = 500,
    target_distribution: Dict[str, float] = None,
    config_overrides: Dict[str, Any] = None
) -> GraphState:
    """
    Run the complete opinion balancing workflow with iterative refinement
    
    Args:
        topic: The opinion topic to write about
        audience: Target audience
        length: Target word count (max 500)
        target_distribution: Stance distribution (e.g., {"Left": 0.5, "Right": 0.5})
        config_overrides: Override default configuration values
        
    Returns:
        Final GraphState with balanced opinion piece
    """
    
    # Load configuration for constraints
    import yaml
    import os
    
    config_path = os.path.join(os.path.dirname(__file__), '..', 'config.yaml')
    try:
        with open(config_path, 'r') as f:
            config = yaml.safe_load(f)
    except:
        # Fallback to default config
        config = {
            'run': {
                'max_passes': 3,
                'thresholds': {
                    'bias_delta_max': 0.05,
                    'frame_entropy_min': 0.6,
                    'readability_grade_min': 10,
                    'readability_grade_max': 13,
                    'coherence_min': 0.7
                }
            }
        }
    
    # Extract constraints from config
    constraints = config.get('run', {}).get('thresholds', {})
    constraints['max_passes'] = config.get('run', {}).get('max_passes', 3)
    
    # Apply any config overrides
    if config_overrides:
        constraints.update(config_overrides)
    
    # Create initial state
    initial_state = GraphState(
        topic=topic,
        audience=audience,
        length=length,
        constraints=constraints,
        pass_count=0,
        converged=False
    )
    
    # Set target distribution
    if target_distribution:
        initial_state.target_distribution = target_distribution
    else:
        initial_state.target_distribution = {"Left": 0.5, "Right": 0.5}
    
    # Create and run the graph
    graph = create_opinion_balancer_graph()
    
    print("🚀 Starting OpinionBalancer Workflow with Iterative Refinement")
    print(f"📋 Max passes: {constraints.get('max_passes', 3)}")
    print(f"🎯 Target: {target_distribution or {'Left': 0.5, 'Right': 0.5}}")
    print("=" * 60)
    
    try:
        # Execute the workflow
        result = graph.invoke(initial_state)
        
        # LangGraph sometimes returns a dict, ensure we have a GraphState
        if isinstance(result, dict):
            # Extract the state data and create a new GraphState
            final_state = GraphState(**result)
        else:
            final_state = result
        
        print(f"\n🎉 OpinionBalancer Complete!")
        print(f"📊 Final pass count: {final_state.pass_count}")
        print(f"✅ Converged: {final_state.converged}")
        
        return final_state
        
    except Exception as e:
        print(f"\n❌ Workflow failed: {e}")
        # Return the current state even if workflow fails
        return initial_state


def create_simple_test_graph() -> CompiledStateGraph:
    """
    Create a simplified graph for testing individual components (with convergence)
    
    Returns:
        Simplified graph for debugging with convergence loop
    """
    graph = StateGraph(GraphState)
    
    # Add minimal nodes for testing
    graph.add_node("topic_intake", topic_intake)
    graph.add_node("draft_writer", draft_writer)
    graph.add_node("evaluate_all", evaluate_all_metrics)
    graph.add_node("convergence_check", convergence_check)
    graph.add_node("increment_pass", increment_pass_counter)
    graph.add_node("logger", logger)
    
    # Simple flow with convergence
    graph.set_entry_point("topic_intake")
    graph.add_edge("topic_intake", "draft_writer")
    graph.add_edge("draft_writer", "evaluate_all")
    graph.add_edge("evaluate_all", "increment_pass")
    graph.add_edge("increment_pass", "convergence_check")
    
    # Conditional routing
    graph.add_conditional_edges(
        "convergence_check",
        route_convergence_decision,
        {
            "continue": "evaluate_all",
            "end": "logger"
        }
    )
    
    graph.add_edge("logger", END)
    
    return graph.compile()


def test_individual_nodes():
    """Test individual nodes in isolation"""
    from nodes.topic_intake import create_intake_from_args
    
    print("🧪 Testing Individual Nodes")
    print("=" * 40)
    
    # Create test state
    state = create_intake_from_args(
        topic="The benefits and drawbacks of remote work",
        audience="general reader",
        length=600,
        target_stance="Left=0.5,Right=0.5"
    )
    
    print("\n1. Testing Draft Writer...")
    state = draft_writer(state)
    print(f"Draft length: {len(state.draft.split())} words")
    
    print("\n2. Testing Evaluators...")
    state = evaluate_all_metrics(state)
    if state.metrics:
        print(f"Bias delta: {state.metrics.bias_delta:.3f}")
        print(f"Frame entropy: {state.metrics.frame_entropy:.3f}")
        print(f"Readability: {state.metrics.flesch_kincaid:.1f}")
        print(f"Coherence: {state.metrics.coherence_score:.3f}")
    
    print("\n3. Testing Critique Synthesis...")
    state = critique_synth(state)
    print(f"Critique length: {len(state.critique)} characters")
    
    print("\n4. Testing Editor...")
    original_length = len(state.draft.split())
    state = editor(state)
    new_length = len(state.draft.split())
    print(f"Draft edited: {original_length} → {new_length} words")
    
    print("\n5. Testing Logger...")
    state = logger(state)
    
    print("✅ All nodes tested successfully!")
    return state


if __name__ == "__main__":
    # Run a simple test
    test_state = test_individual_nodes()
