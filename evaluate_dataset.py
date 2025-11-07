"""
evaluate_dataset.py

Small CLI utility to run the OpinionBalancer evaluation suite against either
the built-in AllSides dataset or a local dataset file (JSON, JSONL, CSV).

Usage examples:
  python evaluate_dataset.py --dataset allsides --sample-size 10
  python evaluate_dataset.py --dataset data/my_topics.jsonl --sample-size 50

The script uses OpinionBalancerEvaluator from `evaluator.py` and saves results
to the output directory (default: ./evaluation_results).
"""

import argparse
import json
import csv
import os
from typing import List, Dict, Any
from datetime import datetime

from evaluator import OpinionBalancerEvaluator


def load_local_dataset(path: str) -> List[Dict[str, Any]]:
    """Load a local dataset in JSON, JSONL or CSV format.

    Returns a list of records with at least a 'topic' key. If the source has
    other keys (e.g., 'bias') they will be preserved.
    """
    path = os.path.expanduser(path)
    if not os.path.exists(path):
        raise FileNotFoundError(f"Dataset file not found: {path}")

    _, ext = os.path.splitext(path)
    ext = ext.lower()
    items: List[Dict[str, Any]] = []

    if ext in ('.jsonl', '.ndjson'):
        with open(path, 'r', encoding='utf-8') as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                items.append(json.loads(line))

    elif ext == '.json':
        with open(path, 'r', encoding='utf-8') as f:
            data = json.load(f)
            if isinstance(data, list):
                items = data
            elif isinstance(data, dict):
                # If it's a dict with a top-level list under a common key, try to find it
                for k in ('data', 'items', 'records'):
                    if k in data and isinstance(data[k], list):
                        items = data[k]
                        break
                else:
                    # Last resort: treat the dict itself as a single item
                    items = [data]

    elif ext in ('.csv', '.tsv'):
        delimiter = '\t' if ext == '.tsv' else ','
        with open(path, 'r', encoding='utf-8') as f:
            reader = csv.DictReader(f, delimiter=delimiter)
            for row in reader:
                items.append(row)

    else:
        raise ValueError(f"Unsupported dataset extension: {ext}")

    # Normalize to ensure each item has a 'topic' field if possible
    normalized = []
    for i, it in enumerate(items):
        if not isinstance(it, dict):
            continue

        topic = (
            it.get('topic') or it.get('text') or it.get('title') or it.get('headline')
        )
        if not topic:
            # skip records without a usable topic
            continue

        normalized.append({
            'topic': str(topic).strip(),
            'original_bias': it.get('bias', 'unknown'),
            'article_id': it.get('id', i),
            **{k: v for k, v in it.items() if k not in ('topic', 'text', 'title', 'headline')}
        })

    return normalized


def evaluate_local_dataset(dataset_path: str, sample_size: int, output_dir: str) -> Dict[str, Any]:
    """Load a local dataset and evaluate topics using OpinionBalancerEvaluator."""
    topics = load_local_dataset(dataset_path)
    if not topics:
        raise ValueError("No valid topics found in the provided dataset.")

    evaluator = OpinionBalancerEvaluator(output_dir=output_dir)

    results = []
    for i, item in enumerate(topics[:sample_size]):
        print(f"\n📊 Progress: {i+1}/{min(len(topics), sample_size)} - Topic preview: {item['topic'][:80]}")
        try:
            res = evaluator.evaluate_single_topic(
                topic=item['topic'],
                original_bias=item.get('original_bias', 'unknown'),
                article_id=item.get('article_id', i)
            )
            res.update({k: v for k, v in item.items() if k not in ('topic',)})
            results.append(res)
        except Exception as e:
            print(f"Error evaluating topic {i}: {e}")
            results.append({'topic': item.get('topic', ''), 'status': 'error', 'error': str(e)})

    # Use evaluator internals to summarize and save results
    try:
        summary = evaluator._generate_evaluation_summary(results)  # reuse existing logic
    except Exception:
        summary = {'error': 'Could not generate summary.'}

    evaluator._save_results(results, summary)

    return {'summary': summary, 'detailed_results': results}


def main():
    parser = argparse.ArgumentParser(description="Run OpinionBalancer evaluation on a dataset")
    parser.add_argument('--dataset', type=str, default='allsides',
                        help="Path to a local dataset (JSON/JSONL/CSV) or 'allsides' to use the built-in AllSides dataset")
    parser.add_argument('--sample-size', type=int, default=20, help="Number of topics to evaluate")
    parser.add_argument('--output-dir', type=str, default='./evaluation_results', help="Directory to save results")
    parser.add_argument('--dry-run', action='store_true', help="Only load dataset and print stats, do not run evaluations")

    args = parser.parse_args()

    os.makedirs(args.output_dir, exist_ok=True)

    if args.dataset.lower() in ('allsides', 'all-sides', 'all_sides'):
        print("Using built-in AllSides dataset via OpinionBalancerEvaluator")
        if args.dry_run:
            print("Dry-run requested: skipping evaluation run")
            return

        evaluator = OpinionBalancerEvaluator(output_dir=args.output_dir)
        result = evaluator.run_evaluation_suite(sample_size=args.sample_size)
        print("\nDone. Summary:\n", result.get('summary'))

    else:
        print(f"Loading local dataset: {args.dataset}")
        try:
            records = load_local_dataset(args.dataset)
        except Exception as e:
            print(f"Failed to load dataset: {e}")
            return

        print(f"Loaded {len(records)} usable topics from dataset")
        if args.dry_run:
            print("Dry-run: exiting after load")
            return

        result = evaluate_local_dataset(args.dataset, args.sample_size, args.output_dir)
        print("\nDone. Summary:\n", result.get('summary'))


if __name__ == '__main__':
    main()
