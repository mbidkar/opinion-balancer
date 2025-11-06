#!/usr/bin/env python3
"""
Simple script to run OpinionBalancer evaluation
"""

import sys
import os
from evaluator import run_evaluation

def main():
    """Run evaluation with command line options"""
    
    # Default parameters
    sample_size = 10
    output_dir = "./evaluation_results"
    
    # Parse command line arguments
    if len(sys.argv) > 1:
        try:
            sample_size = int(sys.argv[1])
        except ValueError:
            print("❌ First argument must be a number (sample size)")
            sys.exit(1)
            
    if len(sys.argv) > 2:
        output_dir = sys.argv[2]
    
    print(f"🔬 Running OpinionBalancer Evaluation")
    print(f"📊 Testing {sample_size} topics from AllSides dataset")
    print(f"📁 Results will be saved to: {output_dir}")
    print("-" * 50)
    
    # Check if we need the API key
    if not os.getenv('OPENAI_API_KEY'):
        print("⚠️  Make sure to set OPENAI_API_KEY environment variable")
        print("   export OPENAI_API_KEY=\"your-key-here\"")
        print()
    
    # Run the evaluation
    try:
        results = run_evaluation(sample_size, output_dir)
        
        # Print quick summary
        if 'summary' in results and 'error' not in results['summary']:
            summary = results['summary']
            print(f"\n🎯 EVALUATION RESULTS:")
            print(f"   ✅ Success Rate: {summary.get('success_rate', 0):.1%}")
            print(f"   ⚖️  Avg Balance Score: {summary.get('balance_performance', {}).get('avg_balance_score', 0):.3f} / 1.0")
            print(f"   🎭 Well-Balanced Rate: {summary.get('balance_performance', {}).get('well_balanced_rate', 0):.1%}")
            print(f"   🔄 Convergence Rate: {summary.get('convergence_analysis', {}).get('convergence_rate', 0):.1%}")
            print(f"   ⏱️  Avg Time: {summary.get('performance', {}).get('avg_execution_time', 0):.1f}s")
            
            # Performance assessment
            balance_score = summary.get('balance_performance', {}).get('avg_balance_score', 0)
            balanced_rate = summary.get('balance_performance', {}).get('well_balanced_rate', 0)
            
            print(f"\n📈 PERFORMANCE ASSESSMENT:")
            if balance_score >= 0.8 and balanced_rate >= 0.8:
                print("   🏆 EXCELLENT - System produces highly balanced content!")
            elif balance_score >= 0.7 and balanced_rate >= 0.6:
                print("   ✨ GOOD - System generally produces balanced content")
            elif balance_score >= 0.5 and balanced_rate >= 0.4:
                print("   ⚠️  FAIR - System sometimes produces balanced content")
            else:
                print("   ❌ POOR - System struggles to produce balanced content")
                
        elif 'error' in results.get('summary', {}):
            print(f"\n❌ Evaluation failed: {results['summary']['error']}")
        else:
            print(f"\n❌ No evaluation results available")
        
        print(f"\n📋 Full results and detailed report saved to: {output_dir}")
        
    except Exception as e:
        print(f"\n❌ Evaluation failed with error: {e}")
        print("Make sure your OpinionBalancer system is working correctly")
        sys.exit(1)

def print_help():
    """Print help information"""
    print("""
OpinionBalancer Evaluation Tool
=============================

Usage: python run_evaluation.py [sample_size] [output_directory]

Arguments:
  sample_size     Number of topics to test (default: 10)
  output_directory Directory to save results (default: ./evaluation_results)

Examples:
  python run_evaluation.py                    # Test 10 topics
  python run_evaluation.py 25                 # Test 25 topics  
  python run_evaluation.py 50 ./my_results    # Test 50 topics, save to ./my_results

Requirements:
  - OPENAI_API_KEY environment variable set
  - All dependencies installed (pip install -r requirements-simple.txt)
  - Internet connection (to download AllSides dataset)

Output:
  - detailed_results_*.json     Full evaluation data
  - evaluation_summary_*.json   Summary statistics
  - evaluation_report_*.txt     Human-readable report
""")

if __name__ == "__main__":
    if len(sys.argv) > 1 and sys.argv[1] in ['-h', '--help', 'help']:
        print_help()
    else:
        main()