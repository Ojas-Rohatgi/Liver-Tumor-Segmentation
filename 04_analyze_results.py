#!/usr/bin/env python3
"""
analyze_leaderboard_csv.py

Loads the final leaderboard CSV from the 'results/' folder and
generates detailed analysis tables for a final report, focusing on:
1. Performance by lesion size (Dice, HD95, Recall, Precision).
2. "Hard Cases" (low Dice scores on scans with tumors).
3. "Healthy Cases" (false positives on scans with no tumors).
"""

import pandas as pd
import numpy as np
from pathlib import Path
import sys

# --- Configuration ---
RESULTS_DIR = Path.cwd() / "results"
CSV_PATH = RESULTS_DIR / "eval_results_2_stage_full_leaderboard.csv"
HARD_CASE_THRESHOLD = 0.25  # Scans with tumors where Dice was below this


# ---------------------

def main():
    if not CSV_PATH.exists():
        print(f"Error: {CSV_PATH} not found.")
        print("Please run the 'evaluate_all_2_stage.py' script first to generate it.")
        sys.exit(1)

    print(f"Loading final results from {CSV_PATH}...")
    df = pd.read_csv(CSV_PATH)
    df.replace([np.inf, -np.inf], np.nan, inplace=True)

    # We only care about Tumor metrics for this analysis
    tumor_df = df[df['target'] == 'Tumor'].copy()

    # Define lesion size categories based on the GT lesion count
    def categorize_lesions(row):
        total_lesions = row['num_small'] + row['num_medium'] + row['num_large']
        if total_lesions == 0:
            return "Healthy (No GT Lesions)"
        if row['num_large'] > 0:
            return "Large (>8k)"
        if row['num_medium'] > 0:
            return "Medium (1k-8k)"
        if row['num_small'] > 0:
            return "Small (<1k)"
        return "Unknown"  # Should not happen

    tumor_df['lesion_size_category'] = tumor_df.apply(categorize_lesions, axis=1)

    # Separate healthy scans from scans with lesions
    healthy_scans_df = tumor_df[tumor_df['lesion_size_category'] == 'Healthy (No GT Lesions)']
    lesion_scans_df = tumor_df[tumor_df['lesion_size_category'] != 'Healthy (No GT Lesions)']

    print(f"\nFound {len(lesion_scans_df)} scans with GT tumors.")
    print(f"Found {len(healthy_scans_df)} healthy scans (no GT tumors).")

    # ===============================================================
    # 1. Performance by Lesion Size
    # ===============================================================
    print("\n" + "=" * 60)
    print("1. PERFORMANCE BY PREDOMINANT LESION SIZE (on scans with tumors)")
    print("   (Scans are categorized by the largest lesion they contain)")
    print("=" * 60)

    # Aggregate metrics
    size_summary = lesion_scans_df.groupby('lesion_size_category')[[
        'Dice', 'HD95', 'recall_50', 'precision_50'
    ]].agg(['mean', 'std', 'count'])

    # Re-order for logical presentation
    size_summary = size_summary.reindex(['Small (<1k)', 'Medium (1k-8k)', 'Large (>8k)'])

    with pd.option_context('display.precision', 3, 'display.width', 1000):
        print(size_summary)

    # ===============================================================
    # 2. "Hard Cases" Analysis (Low Dice on scans WITH tumors)
    # ===============================================================
    print("\n" + "=" * 60)
    print(f"2. 'HARD CASES' ANALYSIS (Dice < {HARD_CASE_THRESHOLD} on scans with tumors)")
    print("=" * 60)

    hard_cases_df = lesion_scans_df[
        lesion_scans_df['Dice'] < HARD_CASE_THRESHOLD
        ].sort_values(by='Dice')

    if hard_cases_df.empty:
        print(f"No hard cases found with Dice < {HARD_CASE_THRESHOLD}!")
    else:
        print(f"Found {len(hard_cases_df)} scans with Dice < {HARD_CASE_THRESHOLD}:")
        print(hard_cases_df[[
            'scan', 'Dice', 'lesion_size_category',
            'num_small', 'num_medium', 'num_large'
        ]].to_string())

    # ===============================================================
    # 3. "Healthy Scans" (False Positive) Analysis
    # ===============================================================
    print("\n" + "=" * 60)
    print("3. 'HEALTHY SCANS' ANALYSIS (False Positives on scans with NO tumors)")
    print("=" * 60)

    # On a healthy scan, a *perfect* score (no FP) gives Dice=1
    # A *bad* score (any FP) gives Dice=0.0

    fp_scans_df = healthy_scans_df[healthy_scans_df['Dice'] < 0.5]
    num_healthy_scans = len(healthy_scans_df)
    num_fp_scans = len(fp_scans_df)

    if num_healthy_scans > 0:
        specificity = (num_healthy_scans - num_fp_scans) / num_healthy_scans
        print(f"Total healthy scans:     {num_healthy_scans}")
        print(f"Scans with False Pos.:   {num_fp_scans}")
        print(f"Per-Scan Specificity:    {specificity:.4f}")

        if num_fp_scans > 0:
            print("\nScans with False Positives:")
            print(fp_scans_df[['scan', 'Dice', 'RVD', 'HD95']].to_string())
    else:
        print("No healthy scans were found in the dataset to analyze.")

    print("\n" + "=" * 60)
    print("Analysis complete.")


if __name__ == "__main__":
    main()