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
    readability, coherence, critique_synth, editor,
    convergence_check, logger
)


def create_opinion_balancer_graph() -> CompiledStateGraph:
    """
    Create the complete OpinionBalancer workflow graph
    
    Returns:
        Compiled LangGraph ready for execution
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
    
    # Set entry point
    graph.set_entry_point("topic_intake")
    
    # Linear flow: intake → draft_writer
    graph.add_edge("topic_intake", "draft_writer")
    
    # Fan-out: draft_writer → all evaluators
    # Note: In LangGraph, we need to create a coordination node for parallel execution
    graph.add_node("evaluate_all", evaluate_all_metrics)
    graph.add_edge("draft_writer", "evaluate_all")
    
    # Flow: evaluation → critique → editor → convergence
    graph.add_edge("evaluate_all", "critique_synth")
    graph.add_edge("critique_synth", "editor")
    
    # Conditional edge from editor based on convergence
    graph.add_conditional_edges(
        source="editor",
        path=routing_function,
        path_map={
            "continue": "draft_writer",  # Loop back for another pass
            "converged": "logger",       # Final logging and END
        }
    )
    
    # Logger to END
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


def routing_function(state: GraphState) -> str:
    """
    Determine whether to continue iteration or finish
    
    Args:
        state: Current graph state
        
    Returns:
        "continue" to loop back to draft_writer, "converged" to finish
    """
    try:
        # Update pass count first
        state.pass_count += 1
        
        # Run convergence check logic
        decision = convergence_check(state)
        
        # Map convergence check result to routing decision
        if decision == "END":
            return "converged"
        else:
            return "continue"
            
    except Exception as e:
        print(f"❌ Error in routing: {e}")
        # Default to converged on error to avoid infinite loops
        return "converged"


def run_opinion_balancer(
    topic: str,
    audience: str = "general US reader",
    length: int = 750,
    target_distribution: Dict[str, float] = None,
    config_overrides: Dict[str, Any] = None
) -> GraphState:
    """
    Run the complete opinion balancing workflow
    
    Args:
        topic: The opinion topic to write about
        audience: Target audience
        length: Target word count
        target_distribution: Stance distribution (e.g., {"Left": 0.5, "Right": 0.5})
        config_overrides: Override default configuration values
        
    Returns:
        Final GraphState with balanced opinion piece
    """
    
    # Create initial state
    initial_state = GraphState(
        topic=topic,
        audience=audience,
        length=length
    )
    
    # Set target distribution
    if target_distribution:
        initial_state.target_distribution = target_distribution
    else:
        initial_state.target_distribution = {"Left": 0.5, "Right": 0.5}
    
    # Apply config overrides
    if config_overrides:
        if 'constraints' in config_overrides:
            initial_state.constraints.update(config_overrides['constraints'])
    
    # Create and run the graph
    graph = create_opinion_balancer_graph()
    
    print("🚀 Starting OpinionBalancer Workflow")
    print("=" * 50)
    
    try:
        # Execute the workflow
        final_state = graph.invoke(initial_state)
        
        print("\n🎉 OpinionBalancer Complete!")
        return final_state
        
    except Exception as e:
        print(f"\n❌ Workflow failed: {e}")
        # Return the current state even if workflow fails
        return initial_state


def create_simple_test_graph() -> CompiledStateGraph:
    """
    Create a simplified graph for testing individual components
    
    Returns:
        Simplified graph for debugging
    """
    graph = StateGraph(GraphState)
    
    # Add minimal nodes for testing
    graph.add_node("topic_intake", topic_intake)
    graph.add_node("draft_writer", draft_writer)
    graph.add_node("evaluate_all", evaluate_all_metrics)
    graph.add_node("logger", logger)
    
    # Simple linear flow
    graph.set_entry_point("topic_intake")
    graph.add_edge("topic_intake", "draft_writer")
    graph.add_edge("draft_writer", "evaluate_all")
    graph.add_edge("evaluate_all", "logger")
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
