#!/usr/bin/env python3
"""
Script to download MBPP dataset from Hugging Face and convert to CSV format
"""

from datasets import load_dataset
import pandas as pd
import json

def download_mbpp():
    """Download MBPP dataset and save as CSV"""
    print("Downloading MBPP dataset from Hugging Face...")
    
    # Load the full MBPP dataset
    dataset = load_dataset("google-research-datasets/mbpp", "full")
    
    # Get the test split (which contains all problems)
    test_data = dataset["test"]
    
    print(f"Downloaded {len(test_data)} MBPP problems")
    
    # Convert to pandas DataFrame
    df = pd.DataFrame(test_data)
    
    # Display basic info
    print(f"Dataset columns: {list(df.columns)}")
    print(f"Sample problem:")
    print(f"Task ID: {df.iloc[0]['task_id']}")
    print(f"Text: {df.iloc[0]['text'][:100]}...")
    print(f"Code: {df.iloc[0]['code'][:100]}...")
    
    # Save to CSV
    df.to_csv("mbpp.csv", index=False)
    print(f"Saved MBPP dataset to mbpp.csv")
    
    # save to jsonl
    df.to_json("mbpp.jsonl", orient="records", lines=True)
    print(f"Saved MBPP dataset to mbpp.jsonl")
    
    return df

if __name__ == "__main__":
    df = download_mbpp()
    print(f"\nDataset statistics:")
    print(f"- Total problems: {len(df)}")
    print(f"- Task ID range: {df['task_id'].min()} to {df['task_id'].max()}")
    print(f"- Average text length: {df['text'].str.len().mean():.1f} characters")
    print(f"- Average code length: {df['code'].str.len().mean():.1f} characters") 