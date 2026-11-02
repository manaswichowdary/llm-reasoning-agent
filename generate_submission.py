#!/usr/bin/env python3
"""
Run the agent on the CSE 476 test set and write answers in the auto grader format.

Reads cse476_final_project_submission/cse_476_final_project_test_data.json and writes
one {"output": ...} entry per question. Progress is saved as it goes, so you can stop
with Ctrl+C and rerun to resume where you left off.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Dict, List
from tqdm import tqdm
from src.agent import Agent


import os
import signal
import sys

INPUT_PATH = Path("cse476_final_project_submission/cse_476_final_project_test_data.json")
OUTPUT_PATH = Path("cse476_final_project_submission/cse_476_final_project_answers.json")

answers = []
interrupted = False

def save_results(partial=False):
    """Saves current results to the output file."""
    if partial:
        print(f"\nSaving partial results ({len(answers)} items)...")
    
    with OUTPUT_PATH.open("w") as fp:
        json.dump(answers, fp, ensure_ascii=False, indent=2)

def handle_interrupt(sig, frame):
    """Handles Ctrl+C to save before exiting."""
    global interrupted
    print("\n\nProcess interrupted! Saving progress...")
    save_results(partial=True)
    interrupted = True
    sys.exit(0)

# Register signal handler
signal.signal(signal.SIGINT, handle_interrupt)

def load_questions(path: Path) -> List[Dict[str, Any]]:
    with path.open("r") as fp:
        data = json.load(fp)
    if not isinstance(data, list):
        raise ValueError("Input file must contain a list of question objects.")
    return data


def build_answers(questions: List[Dict[str, Any]], limit: int = None) -> List[Dict[str, str]]:
    global answers
    
    # Check for existing progress
    if OUTPUT_PATH.exists():
        try:
            with OUTPUT_PATH.open("r") as fp:
                existing_data = json.load(fp)
                if isinstance(existing_data, list) and len(existing_data) > 0:
                    answers = existing_data
                    print(f"Resuming from existing output file. Found {len(answers)} already processed items.")
        except Exception:
            print("Warning: Could not read existing output file. Starting from scratch.")

    processed_count = len(answers)
    
    if limit:
        print(f"Limiting to first {limit} items for testing.")
        questions = questions[:limit]
    
    # Only process remaining items
    remaining_questions = questions[processed_count:]
    
    if not remaining_questions:
        print("All items already processed!")
        return answers

    agent = Agent()
    print(f"Generating answers for {len(remaining_questions)} remaining items...")
    
    for idx, question in enumerate(tqdm(remaining_questions, initial=processed_count, total=len(questions)), start=processed_count+1):
        if interrupted:
            break
            
        try:
            # Agent prediction logic
            prediction = agent.solve(question)
            
            # Ensure prediction is a string
            prediction_str = str(prediction)
            
            # Truncate if too long (per template validation rules)
            if len(prediction_str) >= 5000:
                print(f"Warning: Answer {idx} too long. Truncating.")
                prediction_str = prediction_str[:4999]
                
            answers.append({"output": prediction_str})
            
            # Incremental save per item
            save_results()
            
        except Exception as e:
            print(f"Error processing item {idx}: {e}")
            answers.append({"output": "Error processing this question."})
            save_results()
            
    return answers


def validate_results(
    questions: List[Dict[str, Any]], answers: List[Dict[str, Any]], limit: int = None
) -> None:
    expected_len = len(questions) if limit is None else limit
    
    # Adjust validation to handle partial results if run was interrupted (though usually validation runs after full completion)
    # But strictly speaking, for full validation:
    if len(answers) != expected_len:
         print(f"Note: Generated {len(answers)} answers vs {expected_len} questions. (Match incomplete if limit not met)")

    for idx, answer in enumerate(answers):
        if "output" not in answer:
            raise ValueError(f"Missing 'output' field for answer index {idx}.")
        if not isinstance(answer["output"], str):
            raise TypeError(
                f"Answer at index {idx} has non-string output: {type(answer['output'])}"
            )
        if len(answer["output"]) >= 5000:
            raise ValueError(
                f"Answer at index {idx} exceeds 5000 characters "
                f"({len(answer['output'])} chars)."
            )



import argparse

def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--limit", type=int, default=None, help="Number of items to process")
    args = parser.parse_args()
    
    limit = args.limit
    
    questions = load_questions(INPUT_PATH)
    answers = build_answers(questions, limit=limit)

    # Final save is handled in loop but good to ensure
    save_results()

    # Validate
    validate_results(questions, answers, limit=limit)
    
    print(
        f"Wrote {len(answers)} answers to {OUTPUT_PATH} "
        "and results format seems valid."
    )




if __name__ == "__main__":
    main()
