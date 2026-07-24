#!/usr/bin/env python3
"""
HTML Report Generator for Mock Prompt Tests

This script converts JSON results from mock prompt tests into beautiful,
interactive HTML reports for easy human review and verification.

Features:
- Interactive navigation between different configurations
- Collapsible sections for each augmentation type
- Syntax highlighting for prompts
- Search and filter functionality
- Responsive design
- Export individual prompts
"""

import os
import json
import argparse
from pathlib import Path
from datetime import datetime
from typing import Dict, List, Any, Optional
import html


def escape_html(text: str) -> str:
    """Escape HTML characters in text"""
    return html.escape(text)


def generate_html_template() -> str:
    """Generate the base HTML template with CSS and JavaScript"""
    return """
<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>Mock Prompt Test Results</title>
    <style>
        * {
            margin: 0;
            padding: 0;
            box-sizing: border-box;
        }
        
        body {
            font-family: 'Segoe UI', Tahoma, Geneva, Verdana, sans-serif;
            line-height: 1.6;
            color: #333;
            background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
            min-height: 100vh;
        }
        
        .container {
            max-width: 1400px;
            margin: 0 auto;
            padding: 20px;
            background: rgba(255, 255, 255, 0.95);
            backdrop-filter: blur(10px);
            border-radius: 15px;
            box-shadow: 0 8px 32px rgba(31, 38, 135, 0.37);
            margin-top: 20px;
            margin-bottom: 20px;
        }
        
        .header {
            text-align: center;
            margin-bottom: 30px;
            padding: 20px;
            background: linear-gradient(135deg, #4facfe 0%, #00f2fe 100%);
            border-radius: 10px;
            color: white;
            box-shadow: 0 4px 15px rgba(79, 172, 254, 0.3);
        }
        
        .header h1 {
            font-size: 2.5rem;
            margin-bottom: 10px;
            text-shadow: 2px 2px 4px rgba(0,0,0,0.3);
        }
        
        .header p {
            font-size: 1.1rem;
            opacity: 0.9;
        }
        
        .stats-grid {
            display: grid;
            grid-template-columns: repeat(auto-fit, minmax(250px, 1fr));
            gap: 20px;
            margin-bottom: 30px;
        }
        
        .stat-card {
            background: white;
            padding: 20px;
            border-radius: 10px;
            box-shadow: 0 4px 15px rgba(0,0,0,0.1);
            border-left: 4px solid #4facfe;
            transition: transform 0.3s ease;
        }
        
        .stat-card:hover {
            transform: translateY(-5px);
        }
        
        .stat-card h3 {
            color: #4facfe;
            margin-bottom: 10px;
            font-size: 1.2rem;
        }
        
        .stat-card .value {
            font-size: 2rem;
            font-weight: bold;
            color: #333;
        }
        
        .controls {
            margin-bottom: 30px;
            padding: 20px;
            background: white;
            border-radius: 10px;
            box-shadow: 0 4px 15px rgba(0,0,0,0.1);
        }
        
        .search-box {
            width: 100%;
            padding: 12px 20px;
            border: 2px solid #e0e0e0;
            border-radius: 25px;
            font-size: 1rem;
            transition: border-color 0.3s ease;
        }
        
        .search-box:focus {
            outline: none;
            border-color: #4facfe;
            box-shadow: 0 0 10px rgba(79, 172, 254, 0.3);
        }
        
        .filter-buttons {
            margin-top: 15px;
            display: flex;
            flex-wrap: wrap;
            gap: 10px;
        }
        
        .filter-btn {
            padding: 8px 16px;
            border: 2px solid #4facfe;
            background: white;
            color: #4facfe;
            border-radius: 20px;
            cursor: pointer;
            transition: all 0.3s ease;
            font-size: 0.9rem;
        }
        
        .filter-btn:hover, .filter-btn.active {
            background: #4facfe;
            color: white;
            transform: translateY(-2px);
        }
        
        .experiment-section {
            margin-bottom: 40px;
            background: white;
            border-radius: 10px;
            box-shadow: 0 4px 15px rgba(0,0,0,0.1);
            overflow: hidden;
        }
        
        .experiment-header {
            background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
            color: white;
            padding: 20px;
            cursor: pointer;
            transition: all 0.3s ease;
        }
        
        .experiment-header:hover {
            background: linear-gradient(135deg, #764ba2 0%, #667eea 100%);
        }
        
        .experiment-header h2 {
            font-size: 1.5rem;
            margin-bottom: 5px;
        }
        
        .experiment-meta {
            opacity: 0.9;
            font-size: 0.9rem;
        }
        
        .experiment-content {
            display: none;
            padding: 20px;
        }
        
        .experiment-content.active {
            display: block;
        }
        
        .augmentation-section {
            margin-bottom: 30px;
            border: 1px solid #e0e0e0;
            border-radius: 8px;
            overflow: hidden;
        }
        
        .augmentation-header {
            background: #f8f9fa;
            padding: 15px 20px;
            cursor: pointer;
            transition: background-color 0.3s ease;
            border-bottom: 1px solid #e0e0e0;
        }
        
        .augmentation-header:hover {
            background: #e9ecef;
        }
        
        .augmentation-header h3 {
            color: #495057;
            display: flex;
            justify-content: space-between;
            align-items: center;
        }
        
        .augmentation-content {
            display: none;
            max-height: 600px;
            overflow-y: auto;
        }
        
        .augmentation-content.active {
            display: block;
        }
        
        .prompt-item {
            padding: 15px 20px;
            border-bottom: 1px solid #f0f0f0;
            transition: background-color 0.3s ease;
        }
        
        .prompt-item:hover {
            background: #f8f9fa;
        }
        
        .prompt-item:last-child {
            border-bottom: none;
        }
        
        .prompt-header {
            display: flex;
            justify-content: space-between;
            align-items: center;
            margin-bottom: 10px;
        }
        
        .task-id {
            font-weight: bold;
            color: #4facfe;
            font-size: 1.1rem;
        }
        
        .prompt-stats {
            font-size: 0.9rem;
            color: #666;
        }
        
        .prompt-content {
            background: #f8f9fa;
            border-radius: 5px;
            padding: 15px;
            margin-top: 10px;
            position: relative;
        }
        
        .prompt-text {
            font-family: 'Courier New', monospace;
            font-size: 0.9rem;
            line-height: 1.4;
            white-space: pre-wrap;
            word-wrap: break-word;
            max-height: 300px;
            overflow-y: auto;
            background: white;
            padding: 10px;
            border-radius: 3px;
            border: 1px solid #e0e0e0;
        }
        
        .copy-btn {
            position: absolute;
            top: 10px;
            right: 10px;
            padding: 5px 10px;
            background: #4facfe;
            color: white;
            border: none;
            border-radius: 3px;
            cursor: pointer;
            font-size: 0.8rem;
            transition: background-color 0.3s ease;
        }
        
        .copy-btn:hover {
            background: #3a8bfd;
        }
        
        .context-indicator {
            display: inline-block;
            padding: 2px 8px;
            border-radius: 10px;
            font-size: 0.8rem;
            font-weight: bold;
            margin-left: 10px;
        }
        
        .has-context {
            background: #d4edda;
            color: #155724;
        }
        
        .no-context {
            background: #f8d7da;
            color: #721c24;
        }
        
        .toggle-icon {
            transition: transform 0.3s ease;
        }
        
        .toggle-icon.rotated {
            transform: rotate(90deg);
        }
        
        .footer {
            text-align: center;
            margin-top: 40px;
            padding: 20px;
            color: #666;
            font-size: 0.9rem;
        }
        
        @media (max-width: 768px) {
            .container {
                margin: 10px;
                padding: 15px;
            }
            
            .header h1 {
                font-size: 2rem;
            }
            
            .stats-grid {
                grid-template-columns: 1fr;
            }
            
            .prompt-header {
                flex-direction: column;
                align-items: flex-start;
                gap: 10px;
            }
        }
        
        .highlight {
            background-color: yellow;
            padding: 2px 4px;
            border-radius: 2px;
        }
    </style>
</head>
<body>
    <div class="container">
        <div class="header">
            <h1>🧪 Mock Prompt Test Results</h1>
            <p>Interactive report for PKG experiment prompt verification</p>
        </div>
        
        <div class="stats-grid" id="statsGrid">
            <!-- Stats will be populated by JavaScript -->
        </div>
        
        <div class="controls">
            <input type="text" class="search-box" id="searchBox" placeholder="🔍 Search prompts, task IDs, or content...">
            <div class="filter-buttons" id="filterButtons">
                <!-- Filter buttons will be populated by JavaScript -->
            </div>
        </div>
        
        <div id="experimentsContainer">
            <!-- Experiment sections will be populated by JavaScript -->
        </div>
        
        <div class="footer">
            <p>Generated on {generation_time} | Mock Prompt Test Report</p>
        </div>
    </div>
    
    <script>
        // Global data will be injected here
        const mockData = {mock_data};
        
        // Initialize the report
        document.addEventListener('DOMContentLoaded', function() {
            initializeReport();
            setupEventListeners();
        });
        
        function initializeReport() {
            generateStats();
            generateFilterButtons();
            generateExperiments();
        }
        
        function setupEventListeners() {
            // Search functionality
            document.getElementById('searchBox').addEventListener('input', function(e) {
                filterPrompts(e.target.value);
            });
            
            // Filter buttons
            document.addEventListener('click', function(e) {
                if (e.target.classList.contains('filter-btn')) {
                    toggleFilter(e.target);
                }
            });
            
            // Collapsible sections
            document.addEventListener('click', function(e) {
                if (e.target.closest('.experiment-header')) {
                    toggleExperiment(e.target.closest('.experiment-section'));
                }
                if (e.target.closest('.augmentation-header')) {
                    toggleAugmentation(e.target.closest('.augmentation-section'));
                }
                if (e.target.classList.contains('copy-btn')) {
                    copyToClipboard(e.target);
                }
            });
        }
        
        function generateStats() {
            const statsContainer = document.getElementById('statsGrid');
            let totalPrompts = 0;
            let totalExperiments = 0;
            let augmentationTypes = new Set();
            let modelsTypes = new Set();
            
            for (const [expName, expData] of Object.entries(mockData)) {
                totalExperiments++;
                modelsTypes.add(expData.config.model_type);
                
                for (const [augType, prompts] of Object.entries(expData.results)) {
                    augmentationTypes.add(augType);
                    totalPrompts += prompts.length;
                }
            }
            
            const stats = [
                { title: 'Total Experiments', value: totalExperiments, icon: '🧪' },
                { title: 'Total Prompts', value: totalPrompts.toLocaleString(), icon: '📝' },
                { title: 'Augmentation Types', value: augmentationTypes.size, icon: '🔧' },
                { title: 'Model Types', value: modelsTypes.size, icon: '🤖' }
            ];
            
            statsContainer.innerHTML = stats.map(stat => `
                <div class="stat-card">
                    <h3>${stat.icon} ${stat.title}</h3>
                    <div class="value">${stat.value}</div>
                </div>
            `).join('');
        }
        
        function generateFilterButtons() {
            const filterContainer = document.getElementById('filterButtons');
            const augmentationTypes = new Set();
            const modelTypes = new Set();
            
            for (const [expName, expData] of Object.entries(mockData)) {
                modelTypes.add(expData.config.model_type);
                for (const augType of Object.keys(expData.results)) {
                    augmentationTypes.add(augType);
                }
            }
            
            const filters = [
                { label: 'All', value: 'all', active: true },
                ...Array.from(augmentationTypes).map(type => ({ label: type.toUpperCase(), value: type })),
                ...Array.from(modelTypes).map(type => ({ label: type.charAt(0).toUpperCase() + type.slice(1), value: type }))
            ];
            
            filterContainer.innerHTML = filters.map(filter => `
                <button class="filter-btn ${filter.active ? 'active' : ''}" data-filter="${filter.value}">
                    ${filter.label}
                </button>
            `).join('');
        }
        
        function generateExperiments() {
            const container = document.getElementById('experimentsContainer');
            let html = '';
            
            for (const [expName, expData] of Object.entries(mockData)) {
                html += generateExperimentSection(expName, expData);
            }
            
            container.innerHTML = html;
        }
        
        function generateExperimentSection(expName, expData) {
            const config = expData.config;
            const results = expData.results;
            
            let html = `
                <div class="experiment-section" data-experiment="${expName}">
                    <div class="experiment-header">
                        <h2>🚀 ${config.model_name} - ${config.benchmark.toUpperCase()}</h2>
                        <div class="experiment-meta">
                            Model Type: ${config.model_type} | 
                            Total Prompts: ${Object.values(results).reduce((sum, prompts) => sum + prompts.length, 0)} |
                            Augmentations: ${Object.keys(results).length}
                        </div>
                    </div>
                    <div class="experiment-content">
            `;
            
            for (const [augType, prompts] of Object.entries(results)) {
                html += generateAugmentationSection(augType, prompts, expName);
            }
            
            html += `
                    </div>
                </div>
            `;
            
            return html;
        }
        
        function generateAugmentationSection(augType, prompts, expName) {
            const withContext = prompts.filter(p => p.prompt_stats.has_context).length;
            const withoutContext = prompts.length - withContext;
            
            let html = `
                <div class="augmentation-section" data-augmentation="${augType}" data-experiment="${expName}">
                    <div class="augmentation-header">
                        <h3>
                            <span>
                                <span class="toggle-icon">▶</span>
                                ${augType.toUpperCase()} 
                                <span class="context-indicator ${withContext > 0 ? 'has-context' : 'no-context'}">
                                    ${withContext} with context, ${withoutContext} without
                                </span>
                            </span>
                            <span>${prompts.length} prompts</span>
                        </h3>
                    </div>
                    <div class="augmentation-content">
            `;
            
            for (const prompt of prompts.slice(0, 50)) { // Limit to first 50 for performance
                html += generatePromptItem(prompt, augType, expName);
            }
            
            if (prompts.length > 50) {
                html += `
                    <div class="prompt-item">
                        <div style="text-align: center; color: #666; font-style: italic;">
                            ... and ${prompts.length - 50} more prompts (showing first 50 for performance)
                        </div>
                    </div>
                `;
            }
            
            html += `
                    </div>
                </div>
            `;
            
            return html;
        }
        
        function generatePromptItem(prompt, augType, expName) {
            const stats = prompt.prompt_stats;
            const hasContext = stats.has_context;
            
            return `
                <div class="prompt-item" data-task-id="${prompt.task_id}" data-augmentation="${augType}" data-experiment="${expName}">
                    <div class="prompt-header">
                        <div class="task-id">${prompt.task_id}</div>
                        <div class="prompt-stats">
                            Length: ${stats.full_prompt_length} chars | 
                            Words: ${stats.full_prompt_words} |
                            ${hasContext ? `Context: ${stats.context_length} chars` : 'No context'}
                        </div>
                    </div>
                    <div class="prompt-content">
                        <button class="copy-btn" data-content="${escapeHtml(prompt.full_prompt)}">Copy</button>
                        <div class="prompt-text">${escapeHtml(prompt.full_prompt)}</div>
                    </div>
                </div>
            `;
        }
        
        function toggleExperiment(experimentSection) {
            const content = experimentSection.querySelector('.experiment-content');
            content.classList.toggle('active');
        }
        
        function toggleAugmentation(augmentationSection) {
            const content = augmentationSection.querySelector('.augmentation-content');
            const icon = augmentationSection.querySelector('.toggle-icon');
            
            content.classList.toggle('active');
            icon.classList.toggle('rotated');
        }
        
        function toggleFilter(button) {
            const filterValue = button.dataset.filter;
            
            // Toggle active state
            document.querySelectorAll('.filter-btn').forEach(btn => btn.classList.remove('active'));
            button.classList.add('active');
            
            // Apply filter
            applyFilter(filterValue);
        }
        
        function applyFilter(filterValue) {
            const experiments = document.querySelectorAll('.experiment-section');
            
            experiments.forEach(exp => {
                if (filterValue === 'all') {
                    exp.style.display = 'block';
                    exp.querySelectorAll('.augmentation-section').forEach(aug => {
                        aug.style.display = 'block';
                    });
                } else {
                    const config = mockData[exp.dataset.experiment].config;
                    const hasMatchingAug = Object.keys(mockData[exp.dataset.experiment].results).includes(filterValue);
                    const hasMatchingModel = config.model_type === filterValue;
                    
                    if (hasMatchingAug || hasMatchingModel) {
                        exp.style.display = 'block';
                        exp.querySelectorAll('.augmentation-section').forEach(aug => {
                            const augType = aug.dataset.augmentation;
                            if (filterValue === augType || filterValue === config.model_type || filterValue === 'all') {
                                aug.style.display = 'block';
                            } else {
                                aug.style.display = 'none';
                            }
                        });
                    } else {
                        exp.style.display = 'none';
                    }
                }
            });
        }
        
        function filterPrompts(searchTerm) {
            const term = searchTerm.toLowerCase();
            const promptItems = document.querySelectorAll('.prompt-item');
            
            promptItems.forEach(item => {
                const taskId = item.dataset.taskId.toLowerCase();
                const promptText = item.querySelector('.prompt-text').textContent.toLowerCase();
                
                if (taskId.includes(term) || promptText.includes(term)) {
                    item.style.display = 'block';
                    highlightText(item, term);
                } else {
                    item.style.display = 'none';
                }
            });
        }
        
        function highlightText(element, term) {
            if (!term) return;
            
            const textElement = element.querySelector('.prompt-text');
            const originalText = textElement.textContent;
            const regex = new RegExp(`(${term})`, 'gi');
            const highlightedText = originalText.replace(regex, '<span class="highlight">$1</span>');
            textElement.innerHTML = highlightedText;
        }
        
        function copyToClipboard(button) {
            const content = button.dataset.content;
            navigator.clipboard.writeText(content).then(() => {
                const originalText = button.textContent;
                button.textContent = 'Copied!';
                button.style.background = '#28a745';
                
                setTimeout(() => {
                    button.textContent = originalText;
                    button.style.background = '#4facfe';
                }, 2000);
            });
        }
        
        function escapeHtml(text) {
            const div = document.createElement('div');
            div.textContent = text;
            return div.innerHTML;
        }
    </script>
</body>
</html>
"""


def load_mock_results(results_dir: Path) -> Dict[str, Any]:
    """Load all mock experiment results from a directory"""
    experiments = {}
    
    for exp_dir in results_dir.iterdir():
        if exp_dir.is_dir() and exp_dir.name.startswith('mock_'):
            exp_name = exp_dir.name
            
            # Load summary
            summary_file = exp_dir / "mock_experiment_summary.json"
            if not summary_file.exists():
                continue
                
            with open(summary_file, 'r') as f:
                summary = json.load(f)
            
            # Load detailed results for each augmentation type
            results = {}
            for aug_type in summary.get('augmentation_types', []):
                detailed_file = exp_dir / f"mock_{aug_type}_detailed_results.json"
                if detailed_file.exists():
                    with open(detailed_file, 'r') as f:
                        results[aug_type] = json.load(f)
            
            experiments[exp_name] = {
                'config': {
                    'model_name': summary.get('model_name', ''),
                    'model_type': summary.get('model_type', ''),
                    'benchmark': summary.get('benchmark', ''),
                    'augmentation_types': summary.get('augmentation_types', [])
                },
                'summary': summary,
                'results': results
            }
    
    return experiments


def generate_html_report(results_dir: Path, output_file: Path):
    """Generate HTML report from mock test results"""
    print(f"📊 Loading mock results from {results_dir}...")
    
    # Load all experiment results
    experiments = load_mock_results(results_dir)
    
    if not experiments:
        print(f"❌ No mock experiment results found in {results_dir}")
        return
    
    print(f"✅ Loaded {len(experiments)} experiment configurations")
    
    # Generate HTML
    template = generate_html_template()
    
    # Inject data and generation time
    html_content = template.replace(
        '{mock_data}', 
        json.dumps(experiments, indent=2)
    ).replace(
        '{generation_time}', 
        datetime.now().strftime('%Y-%m-%d %H:%M:%S')
    )
    
    # Write HTML file
    output_file.parent.mkdir(parents=True, exist_ok=True)
    with open(output_file, 'w', encoding='utf-8') as f:
        f.write(html_content)
    
    print(f"🎉 HTML report generated: {output_file}")
    print(f"📁 File size: {output_file.stat().st_size / 1024 / 1024:.1f} MB")
    
    # Calculate some stats
    total_prompts = 0
    total_experiments = len(experiments)
    augmentation_types = set()
    
    for exp_data in experiments.values():
        for aug_type, prompts in exp_data['results'].items():
            augmentation_types.add(aug_type)
            total_prompts += len(prompts)
    
    print(f"📊 Report contains:")
    print(f"   • {total_experiments} experiment configurations")
    print(f"   • {total_prompts:,} total prompts")
    print(f"   • {len(augmentation_types)} augmentation types")
    print(f"   • Interactive search and filtering")
    print(f"   • Copy-to-clipboard functionality")


def main():
    """Main function"""
    parser = argparse.ArgumentParser(description='Generate HTML report from mock prompt test results')
    
    parser.add_argument('--results-dir', default='mock_prompt_test_results',
                       help='Directory containing mock test results')
    parser.add_argument('--output', default='mock_prompt_test_report.html',
                       help='Output HTML file path')
    parser.add_argument('--open-browser', action='store_true',
                       help='Open the generated report in default browser')
    
    args = parser.parse_args()
    
    results_dir = Path(args.results_dir)
    output_file = Path(args.output)
    
    if not results_dir.exists():
        print(f"❌ Results directory not found: {results_dir}")
        return
    
    # Generate HTML report
    generate_html_report(results_dir, output_file)
    
    # Optionally open in browser
    if args.open_browser:
        import webbrowser
        webbrowser.open(f'file://{output_file.absolute()}')
        print(f"🌐 Opened report in default browser")
    
    print(f"✨ To view the report, open: {output_file.absolute()}")


if __name__ == "__main__":
    main() 