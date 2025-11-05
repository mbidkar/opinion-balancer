#!/usr/bin/env python3
"""
OpinionBalancer CLI
Main entry point for running opinion balancing workflows
"""

import argparse
import sys
import os
from typing import Dict, Any

# Add current directory to Python path for imports
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from graphs.kb_free import run_opinion_balancer, test_individual_nodes
from nodes.topic_intake import parse_target_distribution


def parse_arguments():
    """Parse command line arguments"""
    parser = argparse.ArgumentParser(
        description="OpinionBalancer: Local multi-agent opinion writing system",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Basic usage
  python run.py --topic "Universal basic income in the US"
  
  # With custom parameters
  python run.py --topic "Climate change policy" \\
                --audience "policymakers" \\
                --length 800 \\
                --target "Left=0.4,Center=0.2,Right=0.4"
  
  # Test mode
  python run.py --test
        """
    )
    
    # Main arguments
    parser.add_argument(
        '--topic',
        type=str,
        help='Opinion topic to write about'
    )
    
    parser.add_argument(
        '--audience',
        type=str,
        default='general US reader',
        help='Target audience (default: "general US reader")'
    )
    
    parser.add_argument(
        '--length',
        type=int,
        default=750,
        help='Target word count (default: 750)'
    )
    
    parser.add_argument(
        '--target',
        type=str,
        default='Left=0.5,Right=0.5',
        help='Target stance distribution (default: "Left=0.5,Right=0.5")'
    )
    
    # Configuration overrides
    parser.add_argument(
        '--max-passes',
        type=int,
        default=3,
        help='Maximum refinement passes (default: 3)'
    )
    
    parser.add_argument(
        '--bias-threshold',
        type=float,
        default=0.05,
        help='Maximum allowed bias delta (default: 0.05)'
    )
    
    parser.add_argument(
        '--frame-threshold',
        type=float,
        default=0.6,
        help='Minimum frame entropy (default: 0.6)'
    )
    
    parser.add_argument(
        '--grade-min',
        type=int,
        default=10,
        help='Minimum reading grade level (default: 10)'
    )
    
    parser.add_argument(
        '--grade-max',
        type=int,
        default=13,
        help='Maximum reading grade level (default: 13)'
    )
    
    # System options
    parser.add_argument(
        '--test',
        action='store_true',
        help='Run in test mode (individual node testing)'
    )
    
    parser.add_argument(
        '--debug',
        action='store_true',
        help='Enable debug output'
    )
    
    parser.add_argument(
        '--config',
        type=str,
        default='config.yaml',
        help='Configuration file path (default: config.yaml)'
    )
    
    return parser.parse_args()


def validate_arguments(args) -> bool:
    """Validate command line arguments"""
    
    if args.test:
        return True  # Test mode doesn't need topic
    
    if not args.topic:
        print("❌ Error: --topic is required (unless using --test)")
        return False
    
    if args.length < 100 or args.length > 2000:
        print("❌ Error: --length must be between 100 and 2000 words")
        return False
    
    if args.max_passes < 1 or args.max_passes > 10:
        print("❌ Error: --max-passes must be between 1 and 10")
        return False
    
    if args.bias_threshold < 0 or args.bias_threshold > 1:
        print("❌ Error: --bias-threshold must be between 0 and 1")
        return False
    
    if args.frame_threshold < 0 or args.frame_threshold > 5:
        print("❌ Error: --frame-threshold must be between 0 and 5")
        return False
    
    if args.grade_min < 1 or args.grade_max > 20 or args.grade_min >= args.grade_max:
        print("❌ Error: Invalid grade level range")
        return False
    
    # Validate target distribution
    try:
        target_dist = parse_target_distribution(args.target)
        total = sum(target_dist.values())
        if abs(total - 1.0) > 0.01:  # Allow small floating point errors
            print(f"❌ Error: Target distribution must sum to 1.0 (got {total:.3f})")
            return False
    except Exception as e:
        print(f"❌ Error: Invalid target distribution format: {e}")
        return False
    
    return True


def check_prerequisites() -> bool:
    """Check if all prerequisites are available"""
    
    print("🔍 Checking Prerequisites...")
    
    # Check LLM connection (GPT-2 or Ollama)
    try:
        from llm_client import make_llm_client
        llm_client = make_llm_client()
        
        if not llm_client.test_connection():
            print("❌ Cannot connect to LLM")
            print("   Check your model configuration in config.yaml")
            return False
        
        print("✅ LLM connection successful")
        
    except Exception as e:
        print(f"❌ LLM client error: {e}")
        return False
    
    # Check required directories
    os.makedirs("./runs", exist_ok=True)
    
    # Check configuration files
    if not os.path.exists("config.yaml"):
        print("⚠️  Warning: config.yaml not found, using defaults")
    
    if not os.path.exists("prompts.yaml"):
        print("⚠️  Warning: prompts.yaml not found, using fallback prompts")
    
    print("✅ Prerequisites check complete")
    return True


def create_config_overrides(args) -> Dict[str, Any]:
    """Create configuration overrides from arguments"""
    
    overrides = {
        'constraints': {
            'max_passes': args.max_passes,
            'bias_delta_max': args.bias_threshold,
            'frame_entropy_min': args.frame_threshold,
            'grade_min': args.grade_min,
            'grade_max': args.grade_max
        }
    }
    
    return overrides


def run_test_mode():
    """Run the system in test mode"""
    print("🧪 OpinionBalancer Test Mode")
    print("=" * 50)
    
    try:
        final_state = test_individual_nodes()
        
        if final_state and final_state.draft:
            print(f"\n📝 Final test draft ({len(final_state.draft.split())} words):")
            print("-" * 40)
            print(final_state.draft[:300] + "..." if len(final_state.draft) > 300 else final_state.draft)
            print("-" * 40)
        
        print("✅ Test mode completed successfully!")
        return True
        
    except Exception as e:
        print(f"❌ Test mode failed: {e}")
        if hasattr(e, '__traceback__'):
            import traceback
            traceback.print_exc()
        return False


def run_production_mode(args):
    """Run the full opinion balancing workflow"""
    print("🚀 OpinionBalancer Production Mode")
    print("=" * 50)
    
    # Parse target distribution
    target_distribution = parse_target_distribution(args.target)
    
    # Create config overrides
    config_overrides = create_config_overrides(args)
    
    print(f"📝 Topic: {args.topic}")
    print(f"👥 Audience: {args.audience}")
    print(f"📏 Length: {args.length} words")
    print(f"🎯 Target: {target_distribution}")
    print(f"⚙️  Max passes: {args.max_passes}")
    print()
    
    try:
        # Run the workflow
        final_state = run_opinion_balancer(
            topic=args.topic,
            audience=args.audience,
            length=args.length,
            target_distribution=target_distribution,
            config_overrides=config_overrides
        )
        
        if final_state and final_state.draft:
            print(f"\n📖 Final Opinion Piece:")
            print("=" * 50)
            print(final_state.draft)
            print("=" * 50)
            
            # Show run directory
            if hasattr(final_state, '_run_dir'):
                print(f"\n📁 Complete results saved to: {final_state._run_dir}")
        
        print("✅ OpinionBalancer completed successfully!")
        return True
        
    except Exception as e:
        print(f"❌ OpinionBalancer failed: {e}")
        if args.debug:
            import traceback
            traceback.print_exc()
        return False


def main():
    """Main entry point"""
    args = parse_arguments()
    
    # Validate arguments
    if not validate_arguments(args):
        sys.exit(1)
    
    # Check prerequisites
    if not check_prerequisites():
        sys.exit(1)
    
    # Run appropriate mode
    if args.test:
        success = run_test_mode()
    else:
        success = run_production_mode(args)
    
    sys.exit(0 if success else 1)


if __name__ == "__main__":
    main()
