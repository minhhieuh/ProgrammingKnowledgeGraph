import pandas as pd
import numpy as np

def convert_csv_to_latex(csv_file):
    # Read the CSV file
    df = pd.read_csv(csv_file)
    
    # Remove any empty rows
    df = df.dropna(how='all')
    
    # Convert percentage strings to float values for calculations
    numeric_df = df.copy()
    for col in df.columns[1:]:  # Skip the first column (Method)
        numeric_df[col] = df[col].str.rstrip('%').astype(float)
    
    # Start building the LaTeX table
    latex_output = []
    latex_output.append("\\begin{tabular}{lcccccccc}")
    latex_output.append("\\toprule")
    
    # Create header row - updated to include reranked before ideal
    header = f"\\textbf{{Model}} & \\textbf{{None}} & \\textbf{{BM25}} & \\textbf{{VoyageEmb}} & \\textbf{{Func-BM25}}  & \\textbf{{Func-PKG}} & \\textbf{{Block-PKG}} & \\textbf{{Reranked}} & \\textbf{{Ideal Reranker}} \\\\"
    latex_output.append(header)
    latex_output.append("\\midrule")
    
    # Process each row
    for idx, row in df.iterrows():
        model_name = row['Method']
        
        # Clean up model name for display
        if 'claude-3-haiku' in model_name:
            display_name = 'Claude-3-Haiku'
        elif 'claude-sonnet-4' in model_name:
            display_name = 'Claude-Sonnet-4'
        elif 'gpt-4o-mini' in model_name:
            display_name = 'GPT-4o-mini'
        elif 'gpt-4o' in model_name and 'mini' not in model_name:
            display_name = 'GPT-4o'
        else:
            display_name = model_name
        
        # Get baseline value (no_rag column)
        baseline_val = numeric_df.loc[idx, 'no_rag']
        
        # Build the row
        row_parts = [display_name]
        
        # Map columns in desired order: None, BM25, voyage func, voyage block, reranked, ideal
        column_mapping = [
            ('no_rag', 'None'),
            ('bm25', 'BM25'),
            ('voyage_emb', 'VoyageEmb'),
            ('func_bm25', 'Func-BM25'),
            ('voyage_func', 'Func-PKG'),
            ('voyage_block', 'Block-PKG'),
            ('reranked', 'Reranked'),
            ('ideal_ranker', 'Ideal Reranker')
        ]
        
        # Find the best performing method (excluding ideal_ranker and no_rag)
        best_val = -1
        best_col = None
        for col in ['no_rag', 'bm25', 'voyage_emb', 'func_bm25', 'voyage_func', 'voyage_block', 'reranked']:
            if col in df.columns and numeric_df.loc[idx, col] > best_val:
                best_val = numeric_df.loc[idx, col]
                best_col = col
        
        # Process each column in the target order
        for csv_col, display_name_col in column_mapping:
            if csv_col in df.columns:
                val = numeric_df.loc[idx, csv_col]
                val_str = f"{val:.1f}\\%"
                
                # Calculate color based on performance relative to baseline
                if csv_col == 'no_rag':
                    # Baseline gets gray background
                    cell_content = f"{val_str} \\cellcolor{{gray!0}}"
                    if csv_col == best_col:
                        cell_content = f"\\cellcolor{{gray!0}}\\textbf{{{val_str}}}"
                elif csv_col == 'ideal_ranker':
                    # Ideal ranker gets BlueViolet
                    cell_content = f"\\cellcolor{{BlueViolet!20}}{val_str}"
                elif csv_col == 'func_bm25':
                    # Reranked gets OrangeRed
                    cell_content = f"\\cellcolor{{OrangeRed!20}}?"
                else:
                    diff = val - baseline_val
                    if diff < 0:
                        # Worse than baseline - red shading
                        intensity = min(abs(diff) * 15, 90)  # Much higher multiplier and cap
                        cell_content = f"\\cellcolor{{OrangeRed!{intensity:.0f}}}{val_str}"
                    elif diff > 0:
                        # Better than baseline - green shading
                        intensity = min(diff * 15, 80)  # Much higher multiplier and cap
                        cell_content = f"\\cellcolor{{ForestGreen!{intensity:.0f}}}{val_str}"
                    else:
                        # Same as baseline
                        cell_content = val_str
                    
                    # Bold the best performing method (excluding ideal_ranker and no_rag)
                    if csv_col == best_col:
                        if diff > 0:
                            intensity = min(diff * 15, 80)  # Much higher multiplier
                            cell_content = f"\\cellcolor{{ForestGreen!{intensity:.0f}}}\\textbf{{{val_str}}}"
                        else:
                            cell_content = f"\\textbf{{{val_str}}}"
                
                row_parts.append(cell_content)
            else:
                row_parts.append("N/A")
        
        # Join the row and add to output
        latex_row = " & ".join(row_parts) + " \\\\"
        latex_output.append(latex_row)
    
    latex_output.append("\\bottomrule")
    latex_output.append("\\end{tabular}")
    
    return "\n".join(latex_output)

# Convert the CSV file
if __name__ == "__main__":
    csv_file = "humaneval_pass_rates_only.csv"
    latex_table = convert_csv_to_latex(csv_file)
    print(latex_table)
    # Also save to file
    with open("humaneval_latex_table.tex", "w") as f:
        f.write(latex_table)
    
    csv_file = "mbpp_pass_rates_only.csv"
    latex_table = convert_csv_to_latex(csv_file)
    print(latex_table)
    
    with open("mbpp_latex_table.tex", "w") as f:
        f.write(latex_table)
    
    print(f"\nLaTeX table saved to humaneval_latex_table.tex") 