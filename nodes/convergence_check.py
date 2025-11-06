"""
Convergence Check Node
Determines whether to stop iteration or continue refinement
"""

from state import GraphState, Metrics, PassLog
from datetime import datetime


def check_thresholds(metrics: Metrics, constraints: dict) -> dict:
    """Check if all quality thresholds are met"""
    results = {}
    
    # Bias delta check
    bias_threshold = constraints.get('bias_delta_max', 0.05)
    results['bias_delta'] = {
        'met': metrics.bias_delta <= bias_threshold,
        'current': metrics.bias_delta,
        'threshold': bias_threshold
    }
    
    # Frame entropy check
    frame_threshold = constraints.get('frame_entropy_min', 0.6)
    results['frame_entropy'] = {
        'met': metrics.frame_entropy >= frame_threshold,
        'current': metrics.frame_entropy,
        'threshold': frame_threshold
    }
    
    # Readability checks
    grade_min = constraints.get('grade_min', 10)
    grade_max = constraints.get('grade_max', 13)
    results['readability'] = {
        'met': grade_min <= metrics.flesch_kincaid <= grade_max,
        'current': metrics.flesch_kincaid,
        'threshold_range': f"{grade_min}-{grade_max}"
    }
    
    # Coherence check
    coherence_threshold = constraints.get('coherence_min', 0.7)
    results['coherence'] = {
        'met': metrics.coherence_score >= coherence_threshold,
        'current': metrics.coherence_score,
        'threshold': coherence_threshold
    }
    
    # Overall assessment
    all_met = all(result['met'] for result in results.values())
    critical_met = results['bias_delta']['met'] and results['frame_entropy']['met']
    
    return {
        'individual': results,
        'all_met': all_met,
        'critical_met': critical_met,  # Bias and frame diversity are most important
        'met_count': sum(1 for result in results.values() if result['met'])
    }


def convergence_check(state: GraphState) -> GraphState:
    """
    Check convergence criteria and update state with decision
    
    Args:
        state: Current graph state with metrics and history
        
    Returns:
        Updated state with convergence decision set
    """
    try:
        print("🎯 Convergence Check")
        print("=" * 40)
        
        # Initialize convergence flag if not present
        if not hasattr(state, 'converged'):
            state.converged = False
        
        # Check pass limit
        max_passes = state.constraints.get('max_passes', 3)
        print(f"Pass {state.pass_count} of {max_passes}")
        
        if state.pass_count >= max_passes:
            print(f"🛑 Maximum passes ({max_passes}) reached")
            state.converged = True
            return state
        
        # Check if we have metrics to evaluate
        if not state.metrics:
            if state.pass_count == 0:
                print("▶️  First pass - continuing to evaluation")
                state.converged = False
                return state
            else:
                print("❌ No metrics available - stopping")
                state.converged = True
                return state
        
        # Evaluate thresholds
        threshold_results = check_thresholds(state.metrics, state.constraints)
        
        print("📊 Threshold Analysis:")
        for metric_name, result in threshold_results['individual'].items():
            status = "✅" if result['met'] else "❌"
            if 'threshold_range' in result:
                print(f"   {status} {metric_name}: {result['current']:.3f} (target: {result['threshold_range']})")
            else:
                comparison = "≥" if metric_name == 'frame_entropy' else "≤"
                print(f"   {status} {metric_name}: {result['current']:.3f} (target: {comparison}{result['threshold']})")
        
        print(f"Thresholds met: {threshold_results['met_count']}/4")
        
        # Decision logic
        if threshold_results['all_met']:
            print("🎉 All thresholds met - converged!")
            state.converged = True
            return state
        
        if threshold_results['critical_met'] and state.pass_count >= 2:
            print("✅ Critical thresholds met after 2+ passes - acceptable quality")
            state.converged = True
            return state
        
        # Check for improvement plateau
        if len(state.history) >= 2:
            improvement = analyze_improvement_trend(state.history[-2:])
            if improvement['plateau'] and state.pass_count >= 2:
                print("📈 Improvement plateau detected - stopping to avoid overprocessing")
                state.converged = True
                return state
        
        # Check emergency stop conditions
        if should_force_stop(state):
            print("⛔ Emergency stop condition met")
            state.converged = True
            return state
        
        # Continue iteration
        print("▶️  Continuing refinement")
        state.converged = False
        return state
        
    except Exception as e:
        print(f"❌ Error in convergence check: {e}")
        # Default to stopping on error
        state.converged = True
        return state


def analyze_improvement_trend(recent_history: list) -> dict:
    """Analyze recent improvement trends"""
    if len(recent_history) < 2:
        return {'plateau': False, 'trend': 'insufficient_data'}
    
    # Compare bias delta improvements
    current_bias = recent_history[-1].metrics.bias_delta
    previous_bias = recent_history[-2].metrics.bias_delta
    bias_improvement = previous_bias - current_bias
    
    # Compare frame entropy improvements
    current_entropy = recent_history[-1].metrics.frame_entropy
    previous_entropy = recent_history[-2].metrics.frame_entropy
    entropy_improvement = current_entropy - previous_entropy
    
    # Detect plateau (minimal improvement)
    bias_plateau = abs(bias_improvement) < 0.01
    entropy_plateau = abs(entropy_improvement) < 0.1
    
    plateau = bias_plateau and entropy_plateau
    
    return {
        'plateau': plateau,
        'bias_improvement': bias_improvement,
        'entropy_improvement': entropy_improvement,
        'trend': 'plateau' if plateau else 'improving'
    }


def should_force_stop(state: GraphState) -> bool:
    """Check for emergency stop conditions"""
    
    # Stop if draft becomes too short or too long
    if state.draft:
        word_count = len(state.draft.split())
        target = state.length
        
        if word_count < target * 0.3:  # Less than 30% of target
            print(f"⛔ Draft too short ({word_count} words)")
            return True
        
        if word_count > target * 2.5:  # More than 250% of target
            print(f"⛔ Draft too long ({word_count} words)")
            return True
    
    # Stop if metrics show extreme bias
    if state.metrics and state.metrics.bias_delta > 0.5:
        print(f"⛔ Extreme bias detected ({state.metrics.bias_delta:.3f})")
        return True
    
    return False


def create_convergence_summary(state: GraphState) -> dict:
    """Create summary of convergence analysis"""
    if not state.metrics:
        return {'status': 'no_metrics', 'reason': 'No metrics available'}
    
    threshold_results = check_thresholds(state.metrics, state.constraints)
    
    # Determine convergence reason
    if threshold_results['all_met']:
        reason = 'all_thresholds_met'
    elif threshold_results['critical_met']:
        reason = 'critical_thresholds_met'
    elif state.pass_count >= state.constraints.get('max_passes', 3):
        reason = 'max_passes_reached'
    elif len(state.history) >= 2:
        improvement = analyze_improvement_trend(state.history[-2:])
        if improvement['plateau']:
            reason = 'improvement_plateau'
        else:
            reason = 'continuing'
    else:
        reason = 'continuing'
    
    return {
        'status': 'converged' if state.converged else 'continuing',
        'reason': reason,
        'pass_count': state.pass_count,
        'thresholds_met': threshold_results['met_count'],
        'total_thresholds': 4,
        'final_metrics': {
            'bias_delta': state.metrics.bias_delta,
            'frame_entropy': state.metrics.frame_entropy,
            'readability': state.metrics.flesch_kincaid,
            'coherence': state.metrics.coherence_score
        }
    }
