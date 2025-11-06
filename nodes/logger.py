"""
Logger Node
Persists metrics, drafts, and artifacts for each pass
"""

import os
import json
import csv
import numpy as np
from datetime import datetime
from typing import Dict, Any
from state import GraphState, PassLog


def convert_numpy_types(obj):
    """Convert numpy types to Python native types for JSON serialization"""
    if isinstance(obj, np.floating):
        return float(obj)
    elif isinstance(obj, np.integer):
        return int(obj)
    elif isinstance(obj, np.ndarray):
        return obj.tolist()
    elif isinstance(obj, dict):
        return {key: convert_numpy_types(value) for key, value in obj.items()}
    elif isinstance(obj, list):
        return [convert_numpy_types(item) for item in obj]
    return obj


def create_output_directory(base_dir: str = "./runs") -> str:
    """Create timestamped output directory"""
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    run_dir = os.path.join(base_dir, timestamp)
    
    os.makedirs(run_dir, exist_ok=True)
    return run_dir


def save_pass_log(state: GraphState, run_dir: str) -> None:
    """Save current pass data to log files"""
    if not state.metrics:
        return
    
    # Create pass log entry
    pass_log = PassLog(
        pass_id=state.pass_count,
        draft=state.draft,
        critique=state.critique,
        metrics=state.metrics
    )
    
    # Add to history
    state.history.append(pass_log)
    
    # Save individual pass file
    pass_file = os.path.join(run_dir, f"pass_{state.pass_count:02d}.json")
    with open(pass_file, 'w') as f:
        pass_data = {
            'pass_id': pass_log.pass_id,
            'timestamp': pass_log.timestamp.isoformat(),
            'draft': pass_log.draft,
            'critique': pass_log.critique,
            'metrics': convert_numpy_types(pass_log.metrics.dict()),
            'word_count': len(pass_log.draft.split()) if pass_log.draft else 0
        }
        json.dump(pass_data, f, indent=2)


def save_metrics_csv(state: GraphState, run_dir: str) -> None:
    """Save metrics progression as CSV"""
    if not state.history:
        return
    
    csv_file = os.path.join(run_dir, "metrics_progression.csv")
    
    with open(csv_file, 'w', newline='') as f:
        writer = csv.writer(f)
        
        # Header
        writer.writerow([
            'pass_id', 'timestamp', 'bias_delta', 'frame_entropy', 
            'flesch_kincaid', 'dale_chall', 'coherence_score', 'word_count'
        ])
        
        # Data rows
        for log_entry in state.history:
            writer.writerow([
                log_entry.pass_id,
                log_entry.timestamp.isoformat(),
                log_entry.metrics.bias_delta,
                log_entry.metrics.frame_entropy,
                log_entry.metrics.flesch_kincaid,
                log_entry.metrics.dale_chall,
                log_entry.metrics.coherence_score,
                len(log_entry.draft.split()) if log_entry.draft else 0
            ])


def save_final_artifacts(state: GraphState, run_dir: str) -> None:
    """Save final outputs and summary"""
    
    # Final draft
    final_draft_file = os.path.join(run_dir, "final_draft.txt")
    with open(final_draft_file, 'w') as f:
        f.write(f"# OpinionBalancer Output\n")
        f.write(f"Topic: {state.topic}\n")
        f.write(f"Target Distribution: {state.target_distribution}\n")
        f.write(f"Run ID: {state.run_id}\n")
        f.write(f"Generated: {datetime.now().isoformat()}\n\n")
        f.write("---\n\n")
        f.write(state.draft)
    
    # Run summary
    summary = create_run_summary(state)
    summary_file = os.path.join(run_dir, "run_summary.json")
    with open(summary_file, 'w') as f:
        json.dump(convert_numpy_types(summary), f, indent=2, default=str)
    
    # Complete history as JSONL
    jsonl_file = os.path.join(run_dir, "complete_history.jsonl")
    with open(jsonl_file, 'w') as f:
        for log_entry in state.history:
            json_line = {
                'pass_id': log_entry.pass_id,
                'timestamp': log_entry.timestamp.isoformat(),
                'metrics': log_entry.metrics.dict(),
                'draft_length': len(log_entry.draft.split()) if log_entry.draft else 0,
                'critique_length': len(log_entry.critique.split()) if log_entry.critique else 0
            }
            f.write(json.dumps(json_line) + '\n')


def create_run_summary(state: GraphState) -> Dict[str, Any]:
    """Create comprehensive run summary"""
    summary = {
        'run_metadata': {
            'run_id': state.run_id,
            'topic': state.topic,
            'audience': state.audience,
            'target_length': state.length,
            'target_distribution': state.target_distribution,
            'start_time': state.start_time.isoformat() if state.start_time else None,
            'end_time': datetime.now().isoformat(),
            'total_passes': state.pass_count,
            'converged': state.converged
        },
        'constraints': state.constraints,
        'final_metrics': convert_numpy_types(state.metrics.dict()) if state.metrics else None,
        'final_draft_stats': {
            'word_count': len(state.draft.split()) if state.draft else 0,
            'paragraph_count': len([p for p in state.draft.split('\n\n') if p.strip()]) if state.draft else 0,
            'character_count': len(state.draft) if state.draft else 0
        }
    }
    
    # Add convergence analysis
    if state.metrics:
        from nodes.convergence_check import check_thresholds
        threshold_results = check_thresholds(state.metrics, state.constraints)
        summary['convergence_analysis'] = threshold_results
    
    # Add improvement progression
    if len(state.history) > 1:
        summary['improvement_progression'] = analyze_improvement_progression(state.history)
    
    return summary


def analyze_improvement_progression(history: list) -> Dict[str, Any]:
    """Analyze how metrics improved over iterations"""
    if len(history) < 2:
        return {}
    
    first_metrics = history[0].metrics
    final_metrics = history[-1].metrics
    
    improvements = {
        'bias_delta': {
            'initial': first_metrics.bias_delta,
            'final': final_metrics.bias_delta,
            'improvement': first_metrics.bias_delta - final_metrics.bias_delta,
            'improved': first_metrics.bias_delta > final_metrics.bias_delta
        },
        'frame_entropy': {
            'initial': first_metrics.frame_entropy,
            'final': final_metrics.frame_entropy,
            'improvement': final_metrics.frame_entropy - first_metrics.frame_entropy,
            'improved': final_metrics.frame_entropy > first_metrics.frame_entropy
        },
        'coherence': {
            'initial': first_metrics.coherence_score,
            'final': final_metrics.coherence_score,
            'improvement': final_metrics.coherence_score - first_metrics.coherence_score,
            'improved': final_metrics.coherence_score > first_metrics.coherence_score
        }
    }
    
    # Calculate overall improvement score
    improvement_count = sum(1 for metric in improvements.values() if metric['improved'])
    improvements['overall_improvement_score'] = improvement_count / len(improvements)
    
    return improvements


def logger(state: GraphState) -> GraphState:
    """
    Log current pass data and create artifacts
    
    Args:
        state: Current graph state with results
        
    Returns:
        Updated state (unchanged, just logging side effects)
    """
    try:
        print("📊 Logger (Saving Artifacts)")
        print("=" * 40)
        
        # Create run directory if not exists
        if not hasattr(state, '_run_dir'):
            state._run_dir = create_output_directory()
            print(f"📁 Output directory: {state._run_dir}")
        
        run_dir = state._run_dir
        
        # Save current pass data
        if state.metrics:
            save_pass_log(state, run_dir)
            print(f"💾 Pass {state.pass_count} data saved")
        
        # Update CSV with current metrics
        save_metrics_csv(state, run_dir)
        
        # If this is the final pass, save complete artifacts
        if state.converged or state.pass_count >= state.constraints.get('max_passes', 3):
            save_final_artifacts(state, run_dir)
            print(f"📋 Final artifacts saved to: {run_dir}")
            
            # Print summary
            print_run_summary(state)
        
        print("✅ Logging complete")
        return state
        
    except Exception as e:
        print(f"❌ Error in logger: {e}")
        # Don't fail the whole process if logging fails
        return state


def print_run_summary(state: GraphState) -> None:
    """Print a concise run summary to console"""
    if not state.metrics:
        return
    
    print("\n" + "="*50)
    print("📈 OPINION BALANCER RUN SUMMARY")
    print("="*50)
    print(f"Topic: {state.topic}")
    print(f"Passes: {state.pass_count}")
    print(f"Converged: {'✅ Yes' if state.converged else '❌ No'}")
    print(f"Final word count: {len(state.draft.split()) if state.draft else 0}")
    
    print("\n📊 Final Metrics:")
    constraints = state.constraints
    
    # Bias delta
    bias_status = "✅" if state.metrics.bias_delta <= constraints.get('bias_delta_max', 0.05) else "❌"
    print(f"  {bias_status} Bias Delta: {state.metrics.bias_delta:.3f} (target: ≤{constraints.get('bias_delta_max', 0.05)})")
    
    # Frame entropy
    frame_status = "✅" if state.metrics.frame_entropy >= constraints.get('frame_entropy_min', 0.6) else "❌"
    print(f"  {frame_status} Frame Entropy: {state.metrics.frame_entropy:.3f} (target: ≥{constraints.get('frame_entropy_min', 0.6)})")
    
    # Readability
    grade_min = constraints.get('grade_min', 10)
    grade_max = constraints.get('grade_max', 13)
    read_status = "✅" if grade_min <= state.metrics.flesch_kincaid <= grade_max else "❌"
    print(f"  {read_status} Readability: {state.metrics.flesch_kincaid:.1f} (target: {grade_min}-{grade_max})")
    
    # Coherence
    coherence_status = "✅" if state.metrics.coherence_score >= constraints.get('coherence_min', 0.7) else "❌"
    print(f"  {coherence_status} Coherence: {state.metrics.coherence_score:.3f} (target: ≥{constraints.get('coherence_min', 0.7)})")
    
    print("\n🎯 Stance Distribution:")
    for stance, prob in state.metrics.bias_probs.items():
        target_prob = state.target_distribution.get(stance, 0)
        print(f"  {stance}: {prob:.1%} (target: {target_prob:.1%})")
    
    print("="*50)


def get_latest_run_directory(base_dir: str = "./runs") -> str:
    """Get the most recent run directory"""
    if not os.path.exists(base_dir):
        return None
    
    run_dirs = [d for d in os.listdir(base_dir) 
               if os.path.isdir(os.path.join(base_dir, d)) and d.replace('_', '').isdigit()]
    
    if not run_dirs:
        return None
    
    run_dirs.sort(reverse=True)  # Most recent first
    return os.path.join(base_dir, run_dirs[0])
