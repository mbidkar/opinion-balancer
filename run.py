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
    
    # Fixed configuration - no customization allowed
    
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
    
    return True


def check_prerequisites() -> bool:
    """Check if all prerequisites are available"""
    
    print("🔍 Checking Prerequisites...")
    
    # Check LLM connection (GPT-2 or Ollama)
    try:
        from llm_client_openai import OpenAILLMClient
        llm_client = OpenAILLMClient(model="gpt-5")
        
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


def get_fixed_config() -> Dict[str, Any]:
    """Return fixed configuration settings"""
    
    return {
        'audience': 'general US reader',
        'length': 500,  # Fixed at 500 words maximum
        'target_distribution': {'Left': 0.5, 'Right': 0.5},
        'bias_delta_max': 0.05,
        'frame_entropy_min': 0.6,
        'grade_min': 10,
        'grade_max': 13
    }


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
    
    # Get fixed configuration
    config = get_fixed_config()
    
    print(f"📝 Topic: {args.topic}")
    print(f"👥 Audience: {config['audience']}")
    print(f"📏 Length: {config['length']} words (max)")
    print(f"🎯 Target: {config['target_distribution']}")
    print(f"⚙️  Workflow: Single-pass linear execution")
    print()
    
    try:
        # Run the workflow
        final_state = run_opinion_balancer(
            topic=args.topic,
            audience=config['audience'],
            length=config['length'],
            target_distribution=config['target_distribution'],
            config_overrides=config
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
