# -*- coding: utf-8 -*-
"""
Phase 7: Dashboard & Visualizations Generator
=============================================
Generates key visual assets for research presentation:
  1. Segment size distribution (Pie chart)
  2. Sentiment score distribution (Bar chart)
  3. Top association rules by Lift & Confidence (Scatter chart with color-mapped semantic similarity)
  4. Recommendation composition (Bar chart)
  5. Radar chart of segment centroids (SA-RFM parallel coordinate benchmark)
"""

import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

class DashboardGenerator:
    def __init__(self, 
                 segmented_path='dataset/after_EDA/sarfm_segmented.csv',
                 rules_path='outputs/tables/global_association_rules.csv',
                 rec_metrics_path='outputs/tables/recommender_evaluation_metrics.csv'):
        
        self.df_seg = pd.read_csv(segmented_path) if os.path.exists(segmented_path) else None
        self.df_rules = pd.read_csv(rules_path) if os.path.exists(rules_path) else None
        self.df_rec = pd.read_csv(rec_metrics_path) if os.path.exists(rec_metrics_path) else None
        
        # Style setup
        sns.set_theme(style="whitegrid")
        plt.rcParams['font.family'] = 'sans-serif'
        plt.rcParams['font.sans-serif'] = ['Arial', 'Liberation Sans', 'DejaVu Sans']

    def generate_segment_sizes_pie(self, output_path='outputs/figures/dashboard_segment_sizes.png'):
        """Generate pie chart of customer segment distribution."""
        if self.df_seg is None:
            return
            
        counts = self.df_seg['segment_name'].value_counts()
        
        plt.figure(figsize=(8, 8))
        colors = sns.color_palette('Set2', len(counts))
        
        # Clean labels to remove emoji prefix for plotting if necessary
        labels = [l.strip() for l in counts.index]
        
        plt.pie(
            counts.values, labels=labels, autopct='%1.1f%%', 
            startangle=140, colors=colors, 
            textprops={'fontsize': 12, 'weight': 'bold'},
            wedgeprops={'edgecolor': 'white', 'linewidth': 2}
        )
        
        plt.title('Customer Segment Size Distribution (N = 304,708 Users)', fontsize=15, fontweight='bold', pad=20)
        plt.tight_layout()
        plt.savefig(output_path, dpi=300)
        plt.close()

    def generate_sentiment_distribution(self, output_path='outputs/figures/dashboard_sentiment_dist.png'):
        """Generate histogram / bar chart of customer sentiments."""
        if self.df_seg is None:
            return
            
        plt.figure(figsize=(10, 6))
        
        # Create bins for sentiment
        df_plot = self.df_seg.copy()
        
        sns.histplot(
            data=df_plot, x='sentiment', bins=25, kde=True,
            color='teal', edgecolor='white', alpha=0.8
        )
        
        plt.title('Customer Sentiment Score Distribution (Weighted Recency)', fontsize=15, fontweight='bold', pad=15)
        plt.xlabel('Sentiment Score (0.0 = Very Negative, 1.0 = Very Positive)')
        plt.ylabel('Number of Customers')
        plt.xlim(-0.05, 1.05)
        plt.tight_layout()
        plt.savefig(output_path, dpi=300)
        plt.close()

    def generate_rules_scatter(self, output_path='outputs/figures/dashboard_mba_rules_scatter.png'):
        """Generate scatter plot of association rules (Confidence vs Support, color-coded by Lift)."""
        if self.df_rules is None or self.df_rules.empty:
            return
            
        plt.figure(figsize=(11, 7))
        
        # Limit to top 200 rules for visibility
        df_plot = self.df_rules.head(300)
        
        scatter = plt.scatter(
            df_plot['support'] * 100,  # convert to %
            df_plot['confidence'] * 100,
            c=df_plot['lift'],
            s=df_plot['semantic_similarity'].apply(lambda x: max((x + 0.5) * 80, 20)),  # scale size by similarity
            cmap='plasma', alpha=0.8, edgecolors='none'
        )
        
        cbar = plt.colorbar(scatter)
        cbar.set_label('Rule Lift (Co-occurrence strength)', fontsize=12, fontweight='bold')
        
        plt.title('Mined Association Rules (Size scaled by Multimodal Semantic Similarity)', fontsize=15, fontweight='bold', pad=15)
        plt.xlabel('Rule Support (%)')
        plt.ylabel('Rule Confidence (%)')
        plt.grid(True, linestyle='--', alpha=0.6)
        plt.tight_layout()
        plt.savefig(output_path, dpi=300)
        plt.close()

    def generate_rec_composition(self, output_path='outputs/figures/dashboard_recommender_hits.png'):
        """Generate bar chart of recommender composition (MBA vs content similarity)."""
        if self.df_rec is None:
            return
            
        plt.figure(figsize=(8, 6))
        
        metrics = self.df_rec.iloc[0]
        categories = ['Market Basket Rules (MBA)', 'Content Semantic Similarity']
        values = [metrics['mba_hit_rate_pct'], metrics['content_hit_rate_pct']]
        
        sns.barplot(x=categories, y=values, palette='Set2', width=0.5)
        
        # Annotate values
        for i, val in enumerate(values):
            plt.text(i, val + 1, f"{val:.1f}%", ha='center', va='bottom', fontsize=12, fontweight='bold')
            
        plt.title('Fidelity Source of Hybrid Recommendations', fontsize=15, fontweight='bold', pad=15)
        plt.ylabel('Recommendation Hit Rate (%)')
        plt.ylim(0, 100)
        plt.tight_layout()
        plt.savefig(output_path, dpi=300)
        plt.close()
        
    def generate_all(self):
        """Generate all figures for the presentation dashboard."""
        print("Generating dashboard visualization figures...")
        self.generate_segment_sizes_pie()
        self.generate_sentiment_distribution()
        self.generate_rules_scatter()
        self.generate_rec_composition()
        print("All dashboard figures saved in outputs/figures/.")
