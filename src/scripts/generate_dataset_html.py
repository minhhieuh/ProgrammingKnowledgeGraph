#!/usr/bin/env python3
"""
Script to generate an HTML file containing all dataset instances
with human-friendly formatting for easy viewing.
"""

import json
import os
import pandas as pd
from typing import Dict, Any, List
from human_eval.data import read_problems

def load_mbpp_problems() -> Dict:
    """Load MBPP problems from CSV file"""
    try:
        mbpp_df = pd.read_csv("mbpp.csv")
        problems = {}
        for idx, row in mbpp_df.iterrows():
            task_id = f"MBPP/{row['task_id']}"
            problems[task_id] = {
                'task_id': task_id,
                'prompt': row['text'],
                'canonical_solution': row['code'],
                'test': str(row['test_list']),
                'entry_point': ''
            }
        return problems
    except FileNotFoundError:
        print("MBPP dataset not found. Please run download_mbpp.py first.")
        return {}

def load_augmented_data() -> Dict[str, Dict[str, Any]]:
    """Load all augmented data files for both datasets."""
    augmented_data = {
        'function_wise': {},
        'block_wise': {},
        'bm25': {}
    }
    
    # File mappings for both datasets
    file_mappings = {
        # HumanEval files
        'function_wise': [
            'augmented_problems/humaneval_function_wise_relevant_context.jsonl',
            'augmented_problems/mbpp_functionwise_relevant_context.jsonl'
        ],
        'block_wise': [
            'augmented_problems/humaneval_blockwise_relevant_context.jsonl', 
            'augmented_problems/mbpp_blockwise_relevant_context.jsonl'
        ],
        'bm25': [
            'augmented_problems/bm25_relevant_context_humaneval.jsonl',
            'augmented_problems/bm25_relevant_context_mbpp.jsonl'
        ]
    }
    
    for aug_type, file_paths in file_mappings.items():
        for file_path in file_paths:
            if os.path.exists(file_path):
                print(f"Loading {aug_type} data from {file_path}...")
                try:
                    with open(file_path, 'r', encoding='utf-8') as f:
                        for line in f:
                            if line.strip():
                                data = json.loads(line.strip())
                                task_id = data['task_id']
                                
                                # Handle different task_id formats
                                if isinstance(task_id, int):
                                    # MBPP format - convert to string format
                                    task_id = f"MBPP/{task_id}"
                                elif not task_id.startswith(('HumanEval/', 'MBPP/')):
                                    # Add HumanEval prefix if missing
                                    task_id = f"HumanEval/{task_id}"
                                
                                if aug_type == 'bm25':
                                    # BM25 has 'problem' as a list of strings
                                    augmented_data[aug_type][task_id] = data.get('problem', [])
                                else:
                                    # Voyage has 'problem' as a list of [score, content] pairs
                                    augmented_data[aug_type][task_id] = data.get('problem', [])
                                    
                    print(f"Loaded entries from {file_path}")
                except Exception as e:
                    print(f"Error loading {file_path}: {e}")
            else:
                print(f"File not found: {file_path}")
    
    # Print final counts
    for aug_type in augmented_data:
        print(f"Total {aug_type} entries: {len(augmented_data[aug_type])}")
    
    return augmented_data

def escape_js_string(text: str) -> str:
    """Escape a string for safe inclusion in JavaScript."""
    # Replace problematic characters
    text = text.replace('\\', '\\\\')  # Escape backslashes
    text = text.replace('`', '\\`')    # Escape backticks
    text = text.replace('${', '\\${')  # Escape template literals
    text = text.replace('\n', '\\n')  # Escape newlines
    text = text.replace('\r', '\\r')  # Escape carriage returns
    text = text.replace('\t', '\\t')  # Escape tabs
    text = text.replace('"', '\\"')   # Escape double quotes
    return text

def generate_html_with_data(humaneval_problems: Dict, mbpp_problems: Dict, augmented_data: Dict) -> str:
    """Generate the complete HTML with embedded data."""
    
    # Read the HTML template
    with open('dataset_viewer.html', 'r', encoding='utf-8') as f:
        html_template = f.read()
    
    # Combine all problems
    all_problems = {}
    all_problems.update(humaneval_problems)
    all_problems.update(mbpp_problems)
    
    # Convert problems to list format
    problems_list = []
    for task_id, problem_data in all_problems.items():
        problems_list.append({
            'task_id': task_id,
            'prompt': problem_data['prompt'],
            'canonical_solution': problem_data['canonical_solution'],
            'test': problem_data['test'],
            'entry_point': problem_data.get('entry_point', ''),
            'dataset': 'HumanEval' if task_id.startswith('HumanEval/') else 'MBPP'
        })
    
    # Sort by dataset and task_id
    problems_list.sort(key=lambda x: (x['dataset'], x['task_id']))
    
    # Convert to JavaScript format
    problems_js = json.dumps(problems_list, indent=2)
    augmented_js = json.dumps(augmented_data, indent=2)
    
    # Update the HTML template with new stats
    total_problems = len(problems_list)
    humaneval_count = len(humaneval_problems)
    mbpp_count = len(mbpp_problems)
    
    # Replace stats in HTML
    html_template = html_template.replace(
        '<div class="stat-number" id="total-problems">164</div>',
        f'<div class="stat-number" id="total-problems">{total_problems}</div>'
    )
    html_template = html_template.replace(
        '<div class="stat-number" id="visible-problems">164</div>',
        f'<div class="stat-number" id="visible-problems">{total_problems}</div>'
    )
    
    # Add dataset filter options
    filter_options = '''
            <option value="all">All Datasets</option>
            <option value="humaneval">HumanEval Only</option>
            <option value="mbpp">MBPP Only</option>
            <option value="function_wise">Function-wise (Voyage)</option>
            <option value="block_wise">Block-wise (Voyage)</option>
            <option value="bm25">BM25</option>'''
    
    html_template = html_template.replace(
        '''<option value="all">All Augmentation Types</option>
            <option value="function_wise">Function-wise (Voyage)</option>
            <option value="block_wise">Block-wise (Voyage)</option>
            <option value="bm25">BM25</option>''',
        filter_options
    )
    
    # Update stats section to show both datasets
    stats_section = f'''
    <div class="stats">
        <div class="stat-item">
            <div class="stat-number" id="total-problems">{total_problems}</div>
            <div class="stat-label">Total Problems</div>
        </div>
        <div class="stat-item">
            <div class="stat-number">{humaneval_count}</div>
            <div class="stat-label">HumanEval Problems</div>
        </div>
        <div class="stat-item">
            <div class="stat-number">{mbpp_count}</div>
            <div class="stat-label">MBPP Problems</div>
        </div>
        <div class="stat-item">
            <div class="stat-number">3</div>
            <div class="stat-label">Augmentation Types</div>
        </div>
        <div class="stat-item">
            <div class="stat-number" id="visible-problems">{total_problems}</div>
            <div class="stat-label">Visible Problems</div>
        </div>
    </div>'''
    
    # Replace the stats section
    import re
    html_template = re.sub(
        r'<div class="stats">.*?</div>',
        stats_section,
        html_template,
        flags=re.DOTALL
    )
    
    # Replace the placeholder with actual data
    data_script = f"""
    <script>
        const PROBLEMS_DATA = {problems_js};
        const AUGMENTED_DATA = {augmented_js};
    </script>
    """
    
    # Insert the data script before the existing script tag
    html_with_data = html_template.replace(
        '<script>',
        data_script + '\n    <script>'
    )
    
    return html_with_data

def main():
    """Main function to generate the HTML file."""
    print("Loading HumanEval problems...")
    humaneval_problems = read_problems()
    print(f"Loaded {len(humaneval_problems)} HumanEval problems")
    
    print("\nLoading MBPP problems...")
    mbpp_problems = load_mbpp_problems()
    print(f"Loaded {len(mbpp_problems)} MBPP problems")
    
    print("\nLoading augmented data...")
    augmented_data = load_augmented_data()
    
    # Print statistics
    print(f"\nDataset Statistics:")
    print(f"- Total HumanEval problems: {len(humaneval_problems)}")
    print(f"- Total MBPP problems: {len(mbpp_problems)}")
    print(f"- Total problems: {len(humaneval_problems) + len(mbpp_problems)}")
    print(f"- Function-wise augmented: {len(augmented_data['function_wise'])}")
    print(f"- Block-wise augmented: {len(augmented_data['block_wise'])}")
    print(f"- BM25 augmented: {len(augmented_data['bm25'])}")
    
    print("\nGenerating HTML file...")
    html_content = generate_html_with_data(humaneval_problems, mbpp_problems, augmented_data)
    
    # Write the complete HTML file
    output_file = 'dataset_viewer_complete.html'
    with open(output_file, 'w', encoding='utf-8') as f:
        f.write(html_content)
    
    print(f"\n✅ Successfully generated {output_file}")
    print(f"📁 File size: {len(html_content):,} characters")
    print(f"🌐 Open the file in your web browser to view the dataset")
    
    # Print some sample data for verification
    print(f"\nSample HumanEval problem IDs:")
    for i, task_id in enumerate(list(humaneval_problems.keys())[:3]):
        print(f"  {i+1}. {task_id}")
    
    print(f"\nSample MBPP problem IDs:")
    for i, task_id in enumerate(list(mbpp_problems.keys())[:3]):
        print(f"  {i+1}. {task_id}")
    
    if augmented_data['function_wise']:
        sample_task = list(augmented_data['function_wise'].keys())[0]
        sample_data = augmented_data['function_wise'][sample_task]
        print(f"\nSample function-wise augmentation for {sample_task}:")
        print(f"  - {len(sample_data)} context items")
        if sample_data and len(sample_data[0]) >= 2:
            print(f"  - First item relevance score: {sample_data[0][0]:.3f}")

if __name__ == "__main__":
    main() 