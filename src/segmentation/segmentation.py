# -*- coding: utf-8 -*-
"""
Phase 3: Customer Segmentation using SA-RFM
===========================================
Handles:
  - Loading normalized SA-RFM vectors
  - Selecting optimal K using Elbow, Silhouette, and Davies-Bouldin Index
  - Performing K-Means++ clustering
  - Fitting baseline models (GMM, BIRCH) for comparison
  - Segment profiling and naming
  - Visualization of results (plots and radar charts)
"""

import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.cluster import KMeans, Birch
from sklearn.mixture import GaussianMixture
from sklearn.metrics import silhouette_score, davies_bouldin_score
from sklearn.decomposition import PCA

# Ensure outputs folder exists
os.makedirs('outputs/figures', exist_ok=True)
os.makedirs('outputs/tables', exist_ok=True)
os.makedirs('outputs/models', exist_ok=True)

def load_data(vector_path='dataset/after_EDA/sarfm_vectors.csv', table_path='dataset/after_EDA/sarfm_table.csv'):
    """Load normalized vectors and the raw table."""
    vectors = pd.read_csv(vector_path)
    table = pd.read_csv(table_path)
    return vectors, table

def evaluate_k_range(vectors, max_k=8, sample_size=15000, random_state=42):
    """
    Evaluate K range from 2 to max_k using WCSS (Elbow), Silhouette, and Davies-Bouldin.
    Uses sampling for Silhouette score to avoid memory OOM and long run times.
    """
    X = vectors[['recency_norm', 'frequency_norm', 'monetary_norm', 'sentiment_norm']].values
    
    # Subsample for Silhouette score computation
    if len(X) > sample_size:
        np.random.seed(random_state)
        idx = np.random.choice(len(X), sample_size, replace=False)
        X_sample = X[idx]
    else:
        X_sample = X
        
    results = []
    
    for k in range(2, max_k + 1):
        kmeans = KMeans(n_clusters=k, init='k-means++', random_state=random_state, n_init=10)
        labels = kmeans.fit_predict(X)
        wcss = kmeans.inertia_
        
        # Silhouette on sample
        labels_sample = kmeans.predict(X_sample)
        sil = silhouette_score(X_sample, labels_sample, random_state=random_state)
        
        # Davies-Bouldin on full or sample
        dbi = davies_bouldin_score(X, labels)
        
        results.append({
            'k': k,
            'wcss': wcss,
            'silhouette': sil,
            'dbi': dbi
        })
        
    return pd.DataFrame(results)

def plot_evaluation(eval_df):
    """Plot Elbow (WCSS), Silhouette, and DBI to find optimal K."""
    fig, axes = plt.subplots(1, 3, figsize=(18, 5))
    
    # WCSS (Elbow)
    sns.lineplot(data=eval_df, x='k', y='wcss', marker='o', ax=axes[0], color='royalblue', linewidth=2)
    axes[0].set_title('Elbow Method (WCSS)', fontsize=13, fontweight='bold')
    axes[0].set_xlabel('Number of Clusters (K)')
    axes[0].set_ylabel('Within-Cluster Sum of Squares')
    axes[0].grid(True, linestyle='--', alpha=0.6)
    
    # Silhouette Score
    sns.lineplot(data=eval_df, x='k', y='silhouette', marker='s', ax=axes[1], color='forestgreen', linewidth=2)
    axes[1].set_title('Silhouette Score (Higher is better)', fontsize=13, fontweight='bold')
    axes[1].set_xlabel('Number of Clusters (K)')
    axes[1].set_ylabel('Silhouette Coefficient')
    axes[1].grid(True, linestyle='--', alpha=0.6)
    
    # Davies-Bouldin Index
    sns.lineplot(data=eval_df, x='k', y='dbi', marker='^', ax=axes[2], color='crimson', linewidth=2)
    axes[2].set_title('Davies-Bouldin Index (Lower is better)', fontsize=13, fontweight='bold')
    axes[2].set_xlabel('Number of Clusters (K)')
    axes[2].set_ylabel('DB Index')
    axes[2].grid(True, linestyle='--', alpha=0.6)
    
    plt.tight_layout()
    plt.savefig('outputs/figures/segmentation_evaluation.png', dpi=300)
    plt.close()

def fit_models(vectors, k=5, random_state=42):
    """Fit KMeans++, GMM and BIRCH for comparison."""
    X = vectors[['recency_norm', 'frequency_norm', 'monetary_norm', 'sentiment_norm']].values
    
    # KMeans++
    kmeans = KMeans(n_clusters=k, init='k-means++', random_state=random_state, n_init=10)
    km_labels = kmeans.fit_predict(X)
    
    # GMM
    gmm = GaussianMixture(n_components=k, random_state=random_state)
    gmm_labels = gmm.fit_predict(X)
    
    # BIRCH with smaller threshold
    try:
        birch = Birch(n_clusters=k, threshold=0.01)
        birch_labels = birch.fit_predict(X)
    except Exception:
        birch_labels = np.zeros(len(X), dtype=int)
    
    # Compare models using DBI on a sample
    sample_size = min(20000, len(X))
    np.random.seed(random_state)
    idx = np.random.choice(len(X), sample_size, replace=False)
    X_sample = X[idx]
    
    def evaluate_labels(labels):
        unique_labels = np.unique(labels)
        if len(unique_labels) < 2:
            return {'dbi': np.nan, 'sil': np.nan}
        try:
            dbi = davies_bouldin_score(X, labels)
            # Silhouette on sample
            labels_sample = labels[idx]
            if len(np.unique(labels_sample)) < 2:
                sil = np.nan
            else:
                sil = silhouette_score(X_sample, labels_sample, random_state=random_state)
            return {'dbi': dbi, 'sil': sil}
        except Exception:
            return {'dbi': np.nan, 'sil': np.nan}
            
    comparison = {
        'K-Means++': evaluate_labels(km_labels),
        'GMM': evaluate_labels(gmm_labels),
        'BIRCH': evaluate_labels(birch_labels)
    }
    
    return km_labels, gmm_labels, birch_labels, pd.DataFrame(comparison).T

def name_segments(centroids):
    """
    Assign business names to segments based on centroids of normalized features.
    Features: recency_norm (high=recent), frequency_norm, monetary_norm, sentiment_norm
    """
    segment_names = {}
    for cluster_id, row in centroids.iterrows():
        r = row['recency_norm']
        f = row['frequency_norm']
        m = row['monetary_norm']
        s = row['sentiment_norm']
        
        # Heuristics based on centroids
        if r > 0.6 and f > 0.4 and m > 0.4 and s > 0.7:
            name = "🏆 Champions"
        elif r > 0.6 and f > 0.4 and m > 0.4 and s <= 0.4:
            name = "⚠️ Loyal Critics (Dissatisfied VIPs)"
        elif r > 0.6 and f <= 0.2 and m <= 0.2 and s > 0.7:
            name = "🌱 Promising Newcomers (Satisfied)"
        elif r <= 0.3 and f > 0.3 and m > 0.3 and s > 0.6:
            name = "😴 Sleeping Giants (Hibernating VIPs)"
        elif r <= 0.3 and f <= 0.2 and m <= 0.2 and s <= 0.4:
            name = "💀 Lost & Frustrated"
        elif s > 0.8:
            name = "✨ Positive Enthusiasts"
        elif s <= 0.3:
            name = "🗯️ Negative Detractors"
        elif r > 0.5 and m > 0.3:
            name = "🛍️ Potential Spenders"
        else:
            name = "💤 General/Hibernating"
            
        segment_names[cluster_id] = name
        
    # Deduplicate names if any
    unique_names = list(segment_names.values())
    counts = {n: unique_names.count(n) for n in unique_names}
    dedup_names = {}
    for cid, name in segment_names.items():
        if counts[name] > 1:
            # Add cluster ID to distinguish
            dedup_names[cid] = f"{name} (Group {cid})"
        else:
            dedup_names[cid] = name
            
    return dedup_names

def generate_segment_profiles(table, labels, names):
    """Calculate mean profiles of segments in original and normalized scales."""
    table_copy = table.copy()
    table_copy['cluster'] = labels
    table_copy['segment_name'] = table_copy['cluster'].map(names)
    
    # Calculate means
    profile_cols = ['recency', 'frequency', 'monetary', 'sentiment']
    profiles_raw = table_copy.groupby(['cluster', 'segment_name'])[profile_cols].agg(['mean', 'std', 'count']).reset_index()
    
    return table_copy, profiles_raw

def plot_centroids_radar(centroids_df, names, output_path='outputs/figures/segment_centroids_radar.png'):
    """Generate radar chart / parallel coordinate chart for segment comparison."""
    categories = ['Recency (Recent)', 'Frequency', 'Monetary', 'Sentiment']
    N = len(categories)
    
    # Parallel coordinates are easier and more readable than polar charts
    plt.figure(figsize=(10, 6))
    
    # Sort centroids by index
    centroids_plot = centroids_df.copy()
    centroids_plot['Segment Name'] = centroids_plot.index.map(names)
    
    # Set styling
    sns.set_theme(style="whitegrid")
    
    # Melt for seaborn
    melted = centroids_plot.reset_index().melt(
        id_vars=['cluster', 'Segment Name'],
        value_vars=['recency_norm', 'frequency_norm', 'monetary_norm', 'sentiment_norm'],
        var_name='Metric', value_name='Value'
    )
    
    # Map metric names for display
    metric_map = {
        'recency_norm': 'Recency\n(1.0 = Recent)',
        'frequency_norm': 'Frequency\n(1.0 = High)',
        'monetary_norm': 'Monetary\n(1.0 = Spender)',
        'sentiment_norm': 'Sentiment\n(1.0 = Happy)'
    }
    melted['Metric'] = melted['Metric'].map(metric_map)
    
    # Plot lineplot
    sns.lineplot(
        data=melted, x='Metric', y='Value', hue='Segment Name', 
        style='Segment Name', markers=True, dashes=False, linewidth=3, ms=10
    )
    
    plt.title('Normalized SA-RFM Segment Centroid Comparison', fontsize=15, fontweight='bold', pad=15)
    plt.ylim(-0.05, 1.05)
    plt.ylabel('Normalized Score')
    plt.xlabel('')
    plt.legend(bbox_to_anchor=(1.05, 1), loc='upper left', frameon=True)
    plt.tight_layout()
    plt.savefig(output_path, dpi=300)
    plt.close()

def plot_segment_scatter(vectors, labels, names, output_path='outputs/figures/segment_scatter_2d.png'):
    """Reduce dimension with PCA and plot 2D scatter of clusters."""
    X = vectors[['recency_norm', 'frequency_norm', 'monetary_norm', 'sentiment_norm']].values
    
    # PCA to 2D
    pca = PCA(n_components=2, random_state=42)
    X_pca = pca.fit_transform(X)
    
    # Sample points for visualization
    sample_size = min(15000, len(X))
    np.random.seed(42)
    idx = np.random.choice(len(X), sample_size, replace=False)
    
    df_plot = pd.DataFrame({
        'PCA1': X_pca[idx, 0],
        'PCA2': X_pca[idx, 1],
        'Segment Name': [names[l] for l in labels[idx]]
    })
    
    plt.figure(figsize=(10, 8))
    sns.scatterplot(
        data=df_plot, x='PCA1', y='PCA2', hue='Segment Name', 
        alpha=0.6, palette='Set2', edgecolor=None, s=15
    )
    
    plt.title('Customer Segments Visualized in 2D PCA Space', fontsize=15, fontweight='bold', pad=15)
    plt.xlabel(f'PCA Component 1 ({pca.explained_variance_ratio_[0]*100:.1f}% Variance)')
    plt.ylabel(f'PCA Component 2 ({pca.explained_variance_ratio_[1]*100:.1f}% Variance)')
    plt.legend(bbox_to_anchor=(1.05, 1), loc='upper left', frameon=True, markerscale=2)
    plt.tight_layout()
    plt.savefig(output_path, dpi=300)
    plt.close()
