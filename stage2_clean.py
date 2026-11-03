"""
Stage 2: Clean the Text & Create Train/Validation Splits

"""

import os
import time
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split

from text_cleaner import clean_text

LABEL_COLUMNS = [
    "toxic",
    "severe_toxic",
    "obscene",
    "threat",
    "insult",
    "identity_hate",
]

def main():
    print("=" * 65)
    print("STAGE 2: TEXT PREPROCESSING & DATA SPLITTING")
    print("=" * 65)
    
    # 1. Load raw data
    print("\n[Step 1] Loading raw train.csv...")
    df = pd.read_csv("train.csv")
    print(f"Loaded {len(df):,} comments.")
    
    # 2. Demonstrate before & after on illustrative test cases
    print("\n" + "=" * 65)
    print("[Step 2] DEMONSTRATING TEXT CLEANING BEFORE & AFTER")
    print("=" * 65)
    
    test_samples = [
        "== DISCUSSION TITLE ==\nHey check this link http://example.com/page?id=123 from 192.168.1.1! Won't you agree???",
        "YOU ARE A COMPLETE IDIOTTTTTTTT!!!!!! <br><p>Go away!</p> (UTC) 14 January 2008",
        "I'm don't think that's fair &amp; balanced. Ur edits r wrong.",
    ]
    
    for i, raw in enumerate(test_samples, 1):
        cleaned = clean_text(raw)
        print(f"\nExample {i}:")
        print(f"  [RAW]:     {raw.replace(chr(10), ' ')}")
        print(f"  [CLEANED]: {cleaned}")
        
    # 3. Clean the entire dataset
    print("\n" + "=" * 65)
    print("[Step 3] CLEANING ENTIRE DATASET (159,571 comments)...")
    print("=" * 65)
    
    start_time = time.time()
    df["clean_text"] = df["comment_text"].astype(str).apply(clean_text)
    elapsed = time.time() - start_time
    print(f"✓ Cleaned all {len(df):,} comments in {elapsed:.2f} seconds!")
    
    # Check for empty comments after cleaning
    empty_mask = df["clean_text"].str.strip().str.len() == 0
    num_empty = empty_mask.sum()
    if num_empty > 0:
        print(f"Found {num_empty} comments that became empty after cleaning. Replacing with placeholder 'empty_comment'.")
        df.loc[empty_mask, "clean_text"] = "empty_comment"
        
    # 4. Multi-Label Stratified Split
    print("\n" + "=" * 65)
    print("[Step 4] PERFORMING STRATIFIED TRAIN / VALIDATION SPLIT")
    print("=" * 65)
    
    # Create a composite label string (e.g., '0_0_0_0_0_0' or '1_0_1_0_1_0')
    # This guarantees that rare label combinations are evenly distributed into train & val!
    df["label_signature"] = df[LABEL_COLUMNS].astype(str).agg("_".join, axis=1)
    
    # For very rare signatures with count < 2, group them into a 'rare' bucket so train_test_split won't error
    signature_counts = df["label_signature"].value_counts()
    rare_signatures = signature_counts[signature_counts < 2].index
    df.loc[df["label_signature"].isin(rare_signatures), "label_signature"] = "rare_combination"
    
    # 80% Train, 20% Validation split with stratification
    train_df, val_df = train_test_split(
        df,
        test_size=0.20,
        random_state=42,
        stratify=df["label_signature"]
    )
    
    print(f"✓ Training set:   {len(train_df):,} comments (80.0%)")
    print(f"✓ Validation set: {len(val_df):,} comments (20.0%)")
    
    # 5. Verify label balance between Train and Validation
    print("\n" + "=" * 65)
    print("[Step 5] VERIFYING STRATIFICATION INTEGRITY")
    print("=" * 65)
    print(f"{'Category':15s} | {'Train Count':>11s} | {'Train %':>8s} | {'Val Count':>10s} | {'Val %':>8s}")
    print("-" * 65)
    for col in LABEL_COLUMNS:
        tr_cnt = train_df[col].sum()
        tr_pct = (tr_cnt / len(train_df)) * 100
        val_cnt = val_df[col].sum()
        val_pct = (val_cnt / len(val_df)) * 100
        print(f"{col:15s} | {tr_cnt:11,d} | {tr_pct:7.2f}% | {val_cnt:10,d} | {val_pct:7.2f}%")
        
    # 6. Save split files
    print("\n" + "=" * 65)
    print("[Step 6] SAVING CLEANED DATASETS")
    print("=" * 65)
    
    output_cols = ["id", "clean_text"] + LABEL_COLUMNS
    train_out = "train_clean.csv"
    val_out = "val_clean.csv"
    
    train_df[output_cols].to_csv(train_out, index=False)
    val_df[output_cols].to_csv(val_out, index=False)
    
    print(f"✓ Saved cleaned training set:   '{train_out}' ({os.path.getsize(train_out) / (1024*1024):.2f} MB)")
    print(f"✓ Saved cleaned validation set: '{val_out}' ({os.path.getsize(val_out) / (1024*1024):.2f} MB)")
    print("\n✓ STAGE 2 COMPLETE!")

if __name__ == "__main__":
    main()
