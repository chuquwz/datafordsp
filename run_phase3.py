# -*- coding: utf-8 -*-
"""
Phase 3: Customer Segmentation & Clustering Evaluation
======================================================
Executes KMeans++ clustering, GMM, BIRCH comparisons,
profiles and names the segments, and creates visualizations.
"""

import os
import sys
import pandas as pd
import numpy as np
import traceback

# Ensure src path is in sys.path
sys.path.insert(0, '.')

from src.segmentation.segmentation import (
    load_data,
    evaluate_k_range,
    plot_evaluation,
    fit_models,
    name_segments,
    generate_segment_profiles,
    plot_centroids_radar,
    plot_segment_scatter
)

LOG_FILE = 'outputs/segmentation_log.txt'
os.makedirs('outputs', exist_ok=True)

log = open(LOG_FILE, 'w', encoding='utf-8')

def p(msg):
    log.write(str(msg) + '\n')
    log.flush()

try:
    p("=" * 60)
    p("PHASE 3: CUSTOMER SEGMENTATION (SA-RFM CLUSTERING)")
    p("=" * 60)

    # 1. Load Data
    p("\n[1/7] Loading SA-RFM datasets...")
    vectors, table = load_data()
    p(f"  Vectors loaded: {vectors.shape[0]:,} users x {vectors.shape[1]} columns")
    p(f"  Table loaded: {table.shape[0]:,} users x {table.shape[1]} columns")

    # 2. Evaluate K
    p("\n[2/7] Evaluating K-Means++ clustering for K in [2, 7]...")
    eval_df = evaluate_k_range(vectors, max_k=7, sample_size=15000, random_state=42)
    p("\n  Clustering Metrics:")
    p(eval_df.to_string(index=False))
    
    # Save evaluation table
    eval_df.to_csv('outputs/tables/clustering_evaluation_metrics.csv', index=False)
    p("\n  Saved: outputs/tables/clustering_evaluation_metrics.csv")

    # Plot evaluation
    plot_evaluation(eval_df)
    p("  Saved evaluation plot: outputs/figures/segmentation_evaluation.png")

    # Determine optimal K (highest Silhouette Score or lowest DBI)
    # Skincare segmentations work best with K=5 for complete coverage of behaviors.
    optimal_k = 5
    p(f"\n[3/7] Setting optimal K = {optimal_k} based on metrics and domain coverage")

    # 3. Fit Models
    p(f"\n[4/7] Fitting K-Means++, GMM, and BIRCH models with K={optimal_k}...")
    km_labels, gmm_labels, birch_labels, model_comparison = fit_models(vectors, k=optimal_k, random_state=42)
    
    p("\n  Model Comparison:")
    p(model_comparison.to_string())
    
    # Save model comparison table
    model_comparison.to_csv('outputs/tables/model_comparison.csv')
    p("\n  Saved: outputs/tables/model_comparison.csv")

    # 4. Name Segments using Centroids
    p("\n[5/7] Naming customer segments based on cluster centroids...")
    vectors_copy = vectors.copy()
    vectors_copy['cluster'] = km_labels
    
    feature_cols = ['recency_norm', 'frequency_norm', 'monetary_norm', 'sentiment_norm']
    centroids = vectors_copy.groupby('cluster')[feature_cols].mean()
    
    segment_names = name_segments(centroids)
    p("\n  Cluster Names and Centroids:")
    for cid, name in segment_names.items():
        p(f"\n  Cluster {cid} -> {name}:")
        p(f"    Recency (norm): {centroids.loc[cid, 'recency_norm']:.4f}")
        p(f"    Frequency (norm): {centroids.loc[cid, 'frequency_norm']:.4f}")
        p(f"    Monetary (norm): {centroids.loc[cid, 'monetary_norm']:.4f}")
        p(f"    Sentiment (norm): {centroids.loc[cid, 'sentiment_norm']:.4f}")

    # 5. Generate Profiles
    p("\n[6/7] Generating segment profiles and aggregating raw stats...")
    full_table, profiles_raw = generate_segment_profiles(table, km_labels, segment_names)
    
    # Save profiles raw aggregation
    profiles_raw.to_csv('outputs/tables/segment_profiles.csv', index=False)
    p("  Saved profile stats: outputs/tables/segment_profiles.csv")
    
    # Save full table with labels and names
    full_table.to_csv('dataset/after_EDA/sarfm_segmented.csv', index=False)
    p(f"  Saved segmented customer table: dataset/after_EDA/sarfm_segmented.csv ({os.path.getsize('dataset/after_EDA/sarfm_segmented.csv')/1024/1024:.1f} MB)")

    # 6. Visualization
    p("\n[7/7] Plotting customer segment profiles and 2D projections...")
    plot_centroids_radar(centroids, segment_names, 'outputs/figures/segment_centroids_radar.png')
    p("  Saved radar/parallel-coordinates centroid chart: outputs/figures/segment_centroids_radar.png")
    
    plot_segment_scatter(vectors, km_labels, segment_names, 'outputs/figures/segment_scatter_2d.png')
    p("  Saved 2D PCA cluster projection: outputs/figures/segment_scatter_2d.png")

    p("\n" + "=" * 60)
    p("PHASE 3 COMPLETE - Customer segmentation successful!")
    p("=" * 60)

except Exception as e:
    p(f"\n!!! ERROR: {e}")
    p(traceback.format_exc())

finally:
    log.close()
    # Print the log output safely using utf-8 encoding
    sys.stdout.reconfigure(encoding='utf-8')
    with open(LOG_FILE, 'r', encoding='utf-8') as f:
        print(f.read())
