"""
Stage 1: Exploratory Data Analysis (EDA)
Content Moderation System

This script loads the Jigsaw Toxic Comment dataset, analyzes the 6 target labels,
computes class distributions and correlations, and saves a visualization.
"""

import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

# Set style for charts
sns.set_theme(style="whitegrid")

# The 6 target toxicity labels
LABEL_COLUMNS = [
    "toxic",
    "severe_toxic",
    "obscene",
    "threat",
    "insult",
    "identity_hate",
]

def load_data():
    print("=" * 60)
    print("STEP 1: LOADING DATASETS")
    print("=" * 60)
    
    train_path = "train.csv"
    test_path = "test.csv"
    test_labels_path = "test_labels.csv"
    
    train_df = pd.read_csv(train_path)
    test_df = pd.read_csv(test_path)
    test_labels_df = pd.read_csv(test_labels_path)
    
    print(f"✓ train.csv loaded: {train_df.shape[0]:,} rows, {train_df.shape[1]} columns")
    print(f"✓ test.csv loaded:  {test_df.shape[0]:,} rows, {test_df.shape[1]} columns")
    print(f"✓ test_labels.csv:  {test_labels_df.shape[0]:,} rows, {test_labels_df.shape[1]} columns")
    
    return train_df, test_df, test_labels_df

def inspect_schema_and_nulls(df):
    print("\n" + "=" * 60)
    print("STEP 2: SCHEMA & MISSING VALUE CHECK")
    print("=" * 60)
    
    print("Columns in train.csv:", list(df.columns))
    print("\nMissing values in training set:")
    missing = df.isnull().sum()
    for col, count in missing.items():
        print(f"  - {col:15s}: {count} missing")
        
    # Check comment lengths
    df["char_length"] = df["comment_text"].fillna("").astype(str).str.len()
    df["word_count"] = df["comment_text"].fillna("").astype(str).str.split().str.len()
    
    print("\nComment Length Statistics:")
    print(f"  - Average character length: {df['char_length'].mean():.1f} chars (max: {df['char_length'].max():,})")
    print(f"  - Average word count:       {df['word_count'].mean():.1f} words (max: {df['word_count'].max():,})")

def analyze_labels(df):
    print("\n" + "=" * 60)
    print("STEP 3: LABEL DISTRIBUTION & CLASS IMBALANCE")
    print("=" * 60)
    
    total_comments = len(df)
    
    # Calculate count of active labels per row
    df["total_toxic_labels"] = df[LABEL_COLUMNS].sum(axis=1)
    df["is_clean"] = (df["total_toxic_labels"] == 0).astype(int)
    
    clean_count = df["is_clean"].sum()
    toxic_count = total_comments - clean_count
    
    print(f"Clean Comments (no toxic labels): {clean_count:,} ({clean_count/total_comments*100:.2f}%)")
    print(f"Toxic Comments (>=1 toxic label): {toxic_count:,} ({toxic_count/total_comments*100:.2f}%)")
    
    print("\nPer-Category Breakdown in Training Set:")
    stats = []
    for col in LABEL_COLUMNS:
        count = df[col].sum()
        pct = (count / total_comments) * 100
        stats.append({"Label": col, "Count": count, "Percentage": pct})
        print(f"  • {col:15s}: {count:6,} comments ({pct:5.2f}%)")
        
    stats_df = pd.DataFrame(stats)
    
    print("\nMulti-label Distribution (How many labels trigger together):")
    label_counts = df["total_toxic_labels"].value_counts().sort_index()
    for num_labels, count in label_counts.items():
        pct = (count / total_comments) * 100
        print(f"  • Comments with {num_labels} labels: {count:6,} ({pct:5.2f}%)")
        
    return stats_df

def analyze_correlations(df):
    print("\n" + "=" * 60)
    print("STEP 4: LABEL CORRELATION (CO-OCCURRENCE)")
    print("=" * 60)
    
    corr = df[LABEL_COLUMNS].corr()
    print("\nCorrelation matrix between labels:")
    print(corr.round(3).to_string())
    return corr

def generate_visualizations(df, stats_df, corr):
    print("\n" + "=" * 60)
    print("STEP 5: GENERATING EDA VISUALIZATIONS")
    print("=" * 60)
    
    fig, axes = plt.subplots(1, 2, figsize=(16, 6))
    
    # Plot 1: Bar chart of label percentages
    bars = axes[0].bar(stats_df["Label"], stats_df["Percentage"], color="#3b82f6", edgecolor="black", alpha=0.85)
    axes[0].set_title("Percentage of Comments per Toxic Category", fontsize=14, fontweight="bold", pad=12)
    axes[0].set_ylabel("Percentage (%)", fontsize=12)
    axes[0].set_xticklabels(stats_df["Label"], rotation=30, ha="right", fontsize=11)
    
    # Add data labels on top of bars
    for bar in bars:
        height = bar.get_height()
        axes[0].annotate(f"{height:.2f}%",
                         xy=(bar.get_x() + bar.get_width() / 2, height),
                         xytext=(0, 4),
                         textcoords="offset points",
                         ha="center", va="bottom", fontsize=10, fontweight="bold")
    
    # Plot 2: Correlation Heatmap
    sns.heatmap(corr, annot=True, fmt=".2f", cmap="coolwarm", cbar=True, ax=axes[1], vmin=-0.1, vmax=1.0)
    axes[1].set_title("Label Correlation (Co-occurrence Heatmap)", fontsize=14, fontweight="bold", pad=12)
    axes[1].set_xticklabels(LABEL_COLUMNS, rotation=30, ha="right", fontsize=11)
    axes[1].set_yticklabels(LABEL_COLUMNS, rotation=0, fontsize=11)
    
    plt.tight_layout()
    output_image = "eda_distribution.png"
    plt.savefig(output_image, dpi=150)
    plt.close()
    print(f"✓ Saved visual plot to '{output_image}'")

def show_sample_comments(df):
    print("\n" + "=" * 60)
    print("STEP 6: REAL DATA EXAMPLES")
    print("=" * 60)
    
    print("\n--- Example 1: CLEAN COMMENT (All 0s) ---")
    clean_sample = df[df["total_toxic_labels"] == 0].iloc[0]
    preview = clean_sample["comment_text"][:200].replace("\n", " ")
    print(f"Text: \"{preview}...\"")
    print(f"Labels: {[col for col in LABEL_COLUMNS if clean_sample[col] == 1] or 'CLEAN'}")
    
    print("\n--- Example 2: MULTI-LABEL TOXIC COMMENT ---")
    multi_sample = df[df["total_toxic_labels"] >= 3].iloc[0]
    preview = multi_sample["comment_text"][:200].replace("\n", " ")
    active_labels = [col for col in LABEL_COLUMNS if multi_sample[col] == 1]
    print(f"Text: \"{preview}...\"")
    print(f"Active Labels ({len(active_labels)}): {active_labels}")

if __name__ == "__main__":
    train_df, test_df, test_labels_df = load_data()
    inspect_schema_and_nulls(train_df)
    stats_df = analyze_labels(train_df)
    corr = analyze_correlations(train_df)
    generate_visualizations(train_df, stats_df, corr)
    show_sample_comments(train_df)
    print("\n" + "=" * 60)
    print("✓ STAGE 1 EXPLORATORY ANALYSIS COMPLETE!")
    print("=" * 60)
