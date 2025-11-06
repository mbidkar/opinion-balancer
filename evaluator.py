"""
OpinionBalancer Evaluation Framework
Evaluates system performance against AllSides dataset
"""

import os
import json
import numpy as np
import pandas as pd
from typing import Dict, List, Tuple, Any, Optional
from datetime import datetime
from datasets import load_dataset
import re
import yaml

from graphs.kb_free import run_opinion_balancer
from nodes.bias_score import bias_score
from nodes.readability import readability
from nodes.coherence import coherence
from nodes.frame_entropy import frame_entropy
from state import GraphState


class OpinionBalancerEvaluator:
    """
    Comprehensive evaluation framework for OpinionBalancer system
    Tests against AllSides dataset to measure political balance achievement
    """
    
    def __init__(self, output_dir: str = "./evaluation_results"):
        """Initialize the evaluator"""
        self.output_dir = output_dir
        self.results = []
        
        # Create output directory
        os.makedirs(output_dir, exist_ok=True)
        
        print("🔬 OpinionBalancer Evaluator Initialized")
        print(f"📁 Results will be saved to: {output_dir}")
    
    def load_test_topics(self, sample_size: int = 50) -> List[Dict]:
        """
        Load test topics from AllSides dataset
        
        Args:
            sample_size: Number of articles to sample for testing
            
        Returns:
            List of test topics with their original bias labels
        """
        print("📚 Loading AllSides dataset...")
        
        try:
            ds = load_dataset("liyucheng/allsides")
            
            # Use training split for testing
            data = ds['train']
            print(f"📊 Dataset contains {len(data)} articles")
            
            # Sample random articles
            indices = np.random.choice(len(data), min(sample_size, len(data)), replace=False)
            
            test_topics = []
            for idx in indices:
                article = data[int(idx)]
                
                # Extract topic/headline as test input
                topic = self._extract_topic_from_article(article)
                
                if topic and len(topic.strip()) > 10:  # Valid topic
                    test_topics.append({
                        'topic': topic,
                        'original_bias': article.get('bias', 'unknown'),
                        'original_outlet': article.get('outlet', 'unknown'),
                        'article_id': int(idx),
                        'original_text': str(article.get('content', ''))[:500]  # First 500 chars as reference
                    })
            
            print(f"✅ Loaded {len(test_topics)} valid test topics")
            return test_topics
            
        except Exception as e:
            print(f"❌ Error loading dataset: {e}")
            return []
    
    def _extract_topic_from_article(self, article: Dict) -> str:
        """Extract a neutral topic from the article for testing"""
        
        # Option 1: Use title/headline if available
        if 'title' in article and article['title']:
            title = str(article['title']).strip()
            if len(title) > 10 and len(title) < 200:
                return title
        
        # Option 2: Use first sentence of content
        if 'content' in article and article['content']:
            content = str(article['content'])
            sentences = content.split('.')
            if sentences and len(sentences[0]) > 20:
                first_sentence = sentences[0].strip() + '.'
                if len(first_sentence) < 300:
                    return first_sentence
        
        # Option 3: Look for common article fields
        for field in ['headline', 'summary', 'description', 'text']:
            if field in article and article[field]:
                text = str(article[field]).strip()
                if 20 < len(text) < 200:
                    return text
        
        return None
    
    def evaluate_single_topic(self, topic: str, original_bias: str, article_id: int) -> Dict:
        """
        Test OpinionBalancer on a single topic
        
        Args:
            topic: The topic to generate balanced content for
            original_bias: Original bias label from AllSides
            article_id: ID of the original article
            
        Returns:
            Evaluation results for this topic
        """
        print(f"🧪 Testing: {topic[:60]}...")
        
        start_time = datetime.now()
        
        try:
            # Run OpinionBalancer with target 50/50 distribution
            result = run_opinion_balancer(
                topic=topic,
                target_distribution={"Left": 0.5, "Right": 0.5},
                audience="general US reader",
                length=500
            )
            
            # Extract the final balanced article
            balanced_article = result.draft if hasattr(result, 'draft') else ""
            
            # Analyze the generated content
            analysis = self._analyze_generated_content(
                topic=topic,
                generated_text=balanced_article,
                original_bias=original_bias,
                final_state=result,
                article_id=article_id
            )
            
            analysis['execution_time'] = (datetime.now() - start_time).total_seconds()
            analysis['status'] = 'success'
            
            return analysis
            
        except Exception as e:
            print(f"❌ Error processing topic: {e}")
            return {
                'topic': topic,
                'article_id': article_id,
                'original_bias': original_bias,
                'status': 'error',
                'error': str(e),
                'execution_time': (datetime.now() - start_time).total_seconds()
            }
    
    def _analyze_generated_content(self, topic: str, generated_text: str, 
                                 original_bias: str, final_state: GraphState, article_id: int) -> Dict:
        """Comprehensive analysis of generated content"""
        
        analysis = {
            'topic': topic,
            'article_id': article_id,
            'original_bias': original_bias,
            'generated_length': len(generated_text.split()),
            'generated_text': generated_text[:1000]  # Store first 1000 chars
        }
        
        # 1. Political Balance Analysis
        if hasattr(final_state, 'metrics') and final_state.metrics:
            metrics = final_state.metrics
            analysis.update({
                'left_prob': metrics.bias_probs.get('Left', 0.5),
                'right_prob': metrics.bias_probs.get('Right', 0.5),
                'bias_delta': metrics.bias_delta,
                'frame_entropy': metrics.frame_entropy,
                'coherence_score': metrics.coherence_score,
                'flesch_kincaid': metrics.flesch_kincaid,
                'dale_chall': getattr(metrics, 'dale_chall', 0),
                'word_count': getattr(metrics, 'word_count', len(generated_text.split()))
            })
        else:
            # Fallback: analyze the text directly
            temp_state = GraphState(topic=topic, draft=generated_text)
            temp_state = bias_score(temp_state)
            temp_state = frame_entropy(temp_state)
            temp_state = readability(temp_state)
            temp_state = coherence(temp_state)
            
            if temp_state.metrics:
                analysis.update({
                    'left_prob': temp_state.metrics.bias_probs.get('Left', 0.5),
                    'right_prob': temp_state.metrics.bias_probs.get('Right', 0.5),
                    'bias_delta': temp_state.metrics.bias_delta,
                    'frame_entropy': temp_state.metrics.frame_entropy,
                    'coherence_score': temp_state.metrics.coherence_score,
                    'flesch_kincaid': temp_state.metrics.flesch_kincaid,
                    'word_count': len(generated_text.split())
                })
        
        # 2. Convergence Analysis
        if hasattr(final_state, 'history') and final_state.history:
            analysis.update({
                'total_passes': len(final_state.history),
                'converged': getattr(final_state, 'converged', False),
                'pass_count': getattr(final_state, 'pass_count', 0)
            })
        else:
            analysis.update({
                'total_passes': 1,
                'converged': True,
                'pass_count': 1
            })
        
        # 3. Content Quality Assessment
        analysis.update(self._assess_content_quality(generated_text))
        
        # 4. Balance Achievement Score
        analysis['balance_score'] = self._calculate_balance_score(analysis)
        
        return analysis
    
    def _assess_content_quality(self, text: str) -> Dict:
        """Assess various quality aspects of generated text"""
        
        if not text or len(text.strip()) == 0:
            return {
                'sentence_count': 0,
                'unique_sentences': 0,
                'repetition_ratio': 1.0,
                'left_keywords': 0,
                'right_keywords': 0,
                'neutral_keywords': 0
            }
        
        # Content diversity (simple metrics)
        sentences = [s.strip() for s in text.split('.') if s.strip()]
        unique_sentences = len(set(sentences))
        repetition_ratio = 1 - (unique_sentences / max(len(sentences), 1)) if sentences else 1.0
        
        # Perspective indicators
        perspective_keywords = {
            'left': ['progressive', 'liberal', 'social justice', 'inequality', 'regulation', 'government intervention', 'social programs'],
            'right': ['conservative', 'traditional', 'free market', 'individual responsibility', 'limited government', 'private sector', 'fiscal responsibility'],
            'neutral': ['however', 'while', 'both sides', 'on the other hand', 'balanced', 'perspective', 'viewpoint', 'according to']
        }
        
        keyword_counts = {}
        text_lower = text.lower()
        for category, keywords in perspective_keywords.items():
            count = sum(1 for kw in keywords if kw in text_lower)
            keyword_counts[f'{category}_keywords'] = count
        
        return {
            'sentence_count': len(sentences),
            'unique_sentences': unique_sentences,
            'repetition_ratio': repetition_ratio,
            **keyword_counts
        }
    
    def _calculate_balance_score(self, analysis: Dict) -> float:
        """
        Calculate overall balance achievement score (0-1)
        Higher score = better balance and quality
        """
        
        # Political balance component (0-1, higher = more balanced)
        bias_delta = analysis.get('bias_delta', 0.5)
        balance_component = max(0, 1 - (bias_delta * 2))  # Perfect balance = bias_delta of 0
        
        # Quality component (0-1)
        readability_grade = analysis.get('flesch_kincaid', 15)
        # Target grade level is 10-13
        if 10 <= readability_grade <= 13:
            readability_component = 1.0
        else:
            readability_component = max(0, 1 - abs(readability_grade - 11.5) / 10)
        
        # Coherence component (0-1)
        coherence = analysis.get('coherence_score', 0.5)
        coherence_component = min(1.0, coherence)
        
        # Frame diversity component (0-1)
        frame_entropy = analysis.get('frame_entropy', 0.0)
        frame_component = min(1.0, frame_entropy / 0.6)  # Target is 0.6+
        
        # Content diversity component (0-1)
        repetition_ratio = analysis.get('repetition_ratio', 1.0)
        diversity_component = 1 - repetition_ratio
        
        # Perspective coverage (both left and right keywords present)
        left_kw = analysis.get('left_keywords', 0)
        right_kw = analysis.get('right_keywords', 0)
        perspective_component = 1.0 if left_kw > 0 and right_kw > 0 else 0.5
        
        # Weighted combination
        overall_score = (
            balance_component * 0.4 +       # Political balance is most important
            readability_component * 0.15 +   # Readability
            coherence_component * 0.15 +     # Coherence  
            frame_component * 0.1 +         # Frame diversity
            diversity_component * 0.1 +     # Content diversity
            perspective_component * 0.1     # Perspective coverage
        )
        
        return round(min(1.0, max(0.0, overall_score)), 3)
    
    def run_evaluation_suite(self, sample_size: int = 20) -> Dict:
        """
        Run complete evaluation suite
        
        Args:
            sample_size: Number of topics to test
            
        Returns:
            Comprehensive evaluation results
        """
        print("🚀 Starting OpinionBalancer Evaluation Suite")
        print("=" * 60)
        
        # Load test topics
        test_topics = self.load_test_topics(sample_size)
        
        if not test_topics:
            print("❌ No test topics loaded. Evaluation aborted.")
            return {'error': 'No test topics available'}
        
        # Run evaluation on each topic
        results = []
        for i, topic_data in enumerate(test_topics):
            print(f"\n📊 Progress: {i+1}/{len(test_topics)}")
            
            result = self.evaluate_single_topic(
                topic=topic_data['topic'],
                original_bias=topic_data['original_bias'],
                article_id=topic_data['article_id']
            )
            result['test_id'] = i
            result.update({k: v for k, v in topic_data.items() if k != 'original_text'})  # Add metadata
            
            results.append(result)
            self.results.append(result)
        
        # Generate summary statistics
        summary = self._generate_evaluation_summary(results)
        
        # Save results
        self._save_results(results, summary)
        
        print("\n🎉 Evaluation Complete!")
        print(f"📊 Results saved to: {self.output_dir}")
        
        return {
            'summary': summary,
            'detailed_results': results,
            'total_topics': len(results)
        }
    
    def _generate_evaluation_summary(self, results: List[Dict]) -> Dict:
        """Generate summary statistics from evaluation results"""
        
        successful_results = [r for r in results if r.get('status') == 'success']
        
        if not successful_results:
            return {'error': 'No successful evaluations', 'total_tests': len(results)}
        
        # Balance performance
        balance_scores = [r['balance_score'] for r in successful_results]
        bias_deltas = [r['bias_delta'] for r in successful_results]
        
        # Quality metrics
        readability_grades = [r.get('flesch_kincaid', 12) for r in successful_results]
        coherence_scores = [r.get('coherence_score', 0.5) for r in successful_results]
        frame_entropies = [r.get('frame_entropy', 0.3) for r in successful_results]
        
        # Convergence analysis
        converged_count = sum(1 for r in successful_results if r.get('converged', True))
        total_passes = [r.get('total_passes', 1) for r in successful_results]
        
        # Execution performance
        execution_times = [r['execution_time'] for r in successful_results]
        
        # Calculate key thresholds
        well_balanced_count = sum(1 for delta in bias_deltas if delta <= 0.05)
        target_readability_count = sum(1 for grade in readability_grades if 10 <= grade <= 13)
        good_frame_diversity_count = sum(1 for entropy in frame_entropies if entropy >= 0.6)
        
        summary = {
            'total_tests': len(results),
            'successful_tests': len(successful_results),
            'success_rate': len(successful_results) / len(results) if results else 0,
            
            'balance_performance': {
                'avg_balance_score': float(np.mean(balance_scores)),
                'median_balance_score': float(np.median(balance_scores)),
                'max_balance_score': float(np.max(balance_scores)),
                'min_balance_score': float(np.min(balance_scores)),
                'avg_bias_delta': float(np.mean(bias_deltas)),
                'well_balanced_count': int(well_balanced_count),
                'well_balanced_rate': float(well_balanced_count / len(bias_deltas)) if bias_deltas else 0
            },
            
            'quality_metrics': {
                'avg_readability_grade': float(np.mean(readability_grades)),
                'target_readability_count': int(target_readability_count),
                'target_readability_rate': float(target_readability_count / len(readability_grades)) if readability_grades else 0,
                'avg_coherence': float(np.mean(coherence_scores)),
                'avg_frame_entropy': float(np.mean(frame_entropies)),
                'good_frame_diversity_count': int(good_frame_diversity_count)
            },
            
            'convergence_analysis': {
                'convergence_rate': float(converged_count / len(successful_results)) if successful_results else 0,
                'avg_passes': float(np.mean(total_passes)),
                'max_passes': int(np.max(total_passes)),
                'min_passes': int(np.min(total_passes)),
                'efficient_convergence_count': sum(1 for r in successful_results 
                                                 if r.get('converged', True) and r.get('total_passes', 1) <= 3)
            },
            
            'performance': {
                'avg_execution_time': float(np.mean(execution_times)),
                'median_execution_time': float(np.median(execution_times)),
                'max_execution_time': float(np.max(execution_times)),
                'min_execution_time': float(np.min(execution_times))
            }
        }
        
        return summary
    
    def _save_results(self, results: List[Dict], summary: Dict):
        """Save evaluation results to files"""
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        
        # Save detailed results
        results_file = os.path.join(self.output_dir, f"detailed_results_{timestamp}.json")
        with open(results_file, 'w') as f:
            json.dump(results, f, indent=2, default=str)
        
        # Save summary
        summary_file = os.path.join(self.output_dir, f"evaluation_summary_{timestamp}.json")
        with open(summary_file, 'w') as f:
            json.dump(summary, f, indent=2, default=str)
        
        # Save human-readable report
        report_file = os.path.join(self.output_dir, f"evaluation_report_{timestamp}.txt")
        with open(report_file, 'w') as f:
            f.write(self._generate_human_readable_report(summary, results))
        
        print(f"📄 Detailed results: {results_file}")
        print(f"📊 Summary: {summary_file}")
        print(f"📋 Report: {report_file}")
    
    def _generate_human_readable_report(self, summary: Dict, results: List[Dict]) -> str:
        """Generate a human-readable evaluation report"""
        
        if 'error' in summary:
            return f"Evaluation failed: {summary['error']}"
        
        report = f"""
OpinionBalancer Evaluation Report
Generated: {datetime.now().strftime("%Y-%m-%d %H:%M:%S")}
{"=" * 60}

OVERVIEW
--------
Total Tests: {summary['total_tests']}
Successful: {summary['successful_tests']} ({summary['success_rate']:.1%})

POLITICAL BALANCE PERFORMANCE
----------------------------
Average Balance Score: {summary['balance_performance']['avg_balance_score']:.3f} / 1.000
Best Balance Score: {summary['balance_performance']['max_balance_score']:.3f}
Worst Balance Score: {summary['balance_performance']['min_balance_score']:.3f}
Average Bias Delta: {summary['balance_performance']['avg_bias_delta']:.3f}
Well-Balanced Articles (δ ≤ 0.05): {summary['balance_performance']['well_balanced_count']} ({summary['balance_performance']['well_balanced_rate']:.1%})

CONTENT QUALITY
---------------
Average Readability Grade: {summary['quality_metrics']['avg_readability_grade']:.1f}
Target Grade Level (10-13): {summary['quality_metrics']['target_readability_count']} articles ({summary['quality_metrics']['target_readability_rate']:.1%})
Average Coherence Score: {summary['quality_metrics']['avg_coherence']:.3f}
Average Frame Entropy: {summary['quality_metrics']['avg_frame_entropy']:.3f}
Good Frame Diversity (≥0.6): {summary['quality_metrics']['good_frame_diversity_count']} articles

CONVERGENCE ANALYSIS
-------------------
Convergence Rate: {summary['convergence_analysis']['convergence_rate']:.1%}
Average Passes: {summary['convergence_analysis']['avg_passes']:.1f}
Pass Range: {summary['convergence_analysis']['min_passes']}-{summary['convergence_analysis']['max_passes']}
Efficient Convergence (≤3 passes): {summary['convergence_analysis']['efficient_convergence_count']} articles

PERFORMANCE
-----------
Average Execution Time: {summary['performance']['avg_execution_time']:.1f} seconds
Median Execution Time: {summary['performance']['median_execution_time']:.1f} seconds
Fastest: {summary['performance']['min_execution_time']:.1f}s | Slowest: {summary['performance']['max_execution_time']:.1f}s

TOP PERFORMING TOPICS
--------------------
"""
        
        # Add top 5 best-balanced topics
        successful_results = [r for r in results if r.get('status') == 'success']
        if successful_results:
            top_balanced = sorted(successful_results, key=lambda x: x.get('balance_score', 0), reverse=True)[:5]
            
            for i, result in enumerate(top_balanced, 1):
                topic_short = result['topic'][:70] + "..." if len(result['topic']) > 70 else result['topic']
                report += f"{i}. Score: {result['balance_score']:.3f} - {topic_short}\n"
        
        report += "\n\nWORST PERFORMING TOPICS\n"
        report += "----------------------\n"
        
        if successful_results:
            worst_balanced = sorted(successful_results, key=lambda x: x.get('balance_score', 1))[:3]
            
            for i, result in enumerate(worst_balanced, 1):
                topic_short = result['topic'][:70] + "..." if len(result['topic']) > 70 else result['topic']
                report += f"{i}. Score: {result['balance_score']:.3f} - {topic_short}\n"
        
        return report


# Simple runner function
def run_evaluation(sample_size: int = 20, output_dir: str = "./evaluation_results") -> Dict:
    """
    Simple function to run the evaluation
    
    Args:
        sample_size: Number of topics to test
        output_dir: Where to save results
    """
    evaluator = OpinionBalancerEvaluator(output_dir)
    return evaluator.run_evaluation_suite(sample_size)


if __name__ == "__main__":
    # Quick test with 5 topics
    print("🧪 Running quick evaluation test...")
    results = run_evaluation(sample_size=5)
    
    if 'summary' in results:
        print("\n📊 QUICK RESULTS:")
        print(f"   Success Rate: {results['summary']['success_rate']:.1%}")
        print(f"   Avg Balance Score: {results['summary']['balance_performance']['avg_balance_score']:.3f}")
        print(f"   Well-Balanced Rate: {results['summary']['balance_performance']['well_balanced_rate']:.1%}")