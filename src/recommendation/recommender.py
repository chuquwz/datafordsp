# -*- coding: utf-8 -*-
"""
Phase 6: Segment-Aware Hybrid Recommendation Engine
===================================================
Implements a hybrid recommendation engine combining:
  1. Segment-aware MBA rules (association-based)
  2. Cosine similarity on multimodal product embeddings (content-based)
  3. Product average rating and sales metrics (popularity-based)

Filter out already purchased items, weights the candidates, and outputs
recommendations with human-readable explanations.
"""

import os
import re
import numpy as np
import pandas as pd
from collections import defaultdict

class SegmentAwareHybridRecommender:
    def __init__(self, 
                 reviews_path='dataset/after_EDA/reviews_cleaned.csv',
                 segmented_path='dataset/after_EDA/sarfm_segmented.csv',
                 embeddings_path='dataset/after_EDA/product_embeddings.csv',
                 mapping_path='dataset/after_EDA/product_mapping.csv',
                 rules_folder='outputs/tables'):
        
        # 1. Load tables
        self.df_reviews = pd.read_csv(reviews_path)
        self.df_segmented = pd.read_csv(segmented_path)
        self.df_mapping = pd.read_csv(mapping_path)
        self.rules_folder = rules_folder
        
        # Set up quick lookups
        self.user_segment = dict(zip(self.df_segmented['user_id'], self.df_segmented['segment_name']))
        self.prod_name = dict(zip(self.df_mapping['product_id'], self.df_mapping['product_name']))
        self.prod_brand = dict(zip(self.df_mapping['product_id'], self.df_mapping['brand']))
        self.prod_price = dict(zip(self.df_mapping['product_id'], self.df_mapping['price']))
        self.prod_rating = dict(zip(self.df_mapping['product_id'], self.df_mapping['avg_star']))
        self.prod_sales = dict(zip(self.df_mapping['product_id'], self.df_mapping['num_sold_time']))
        
        # Max sales for normalization
        self.max_sales = self.df_mapping['num_sold_time'].max() if not self.df_mapping.empty else 1.0
        
        # Load user purchase histories
        self.user_purchases = self.df_reviews.groupby('user_id')['product_id'].apply(set).to_dict()
        
        # Load product embeddings
        df_emb = pd.read_csv(embeddings_path)
        self.embeddings = {}
        for _, row in df_emb.iterrows():
            pid = int(row['product_id'])
            vec = row.drop('product_id').values.astype(float)
            self.embeddings[pid] = vec
            
        # Cache segment rules
        self.segment_rules = {}
        self._load_segment_rules()

    def _load_segment_rules(self):
        """Pre-load segment rules from outputs folder."""
        # Find all files starting with segment_rules_
        if not os.path.exists(self.rules_folder):
            return
            
        for f in os.listdir(self.rules_folder):
            if f.startswith('segment_rules_') and f.endswith('.csv'):
                # Extract segment key from filename
                clean_name = f.replace('segment_rules_', '').replace('.csv', '')
                try:
                    df = pd.read_csv(os.path.join(self.rules_folder, f))
                    self.segment_rules[clean_name] = df
                except Exception:
                    pass

    def _get_segment_key(self, segment_name):
        """Map standard segment name to the sanitized filename key."""
        if pd.isna(segment_name):
            return "generalhibernating"
        import re
        clean_name = re.sub(r'[^\w\s-]', '', segment_name).strip()
        clean_name = re.sub(r'\s+', '_', clean_name).lower()
        return clean_name

    def _cosine_similarity(self, v1, v2):
        """Compute cosine similarity between two vectors."""
        dot = np.dot(v1, v2)
        norm1 = np.linalg.norm(v1)
        norm2 = np.linalg.norm(v2)
        if norm1 > 0 and norm2 > 0:
            return dot / (norm1 * norm2)
        return 0.0

    def recommend(self, user_id, top_k=5, w_mba=0.4, w_sim=0.4, w_pop=0.2):
        """
        Generate hybrid recommendations for a customer.
        
        Parameters:
          user_id: ID of the customer
          top_k: Number of recommendations to return
          w_mba: Weight for association-rules candidates (MBA)
          w_sim: Weight for content similarity candidates
          w_pop: Weight for popularity (sales & rating)
        """
        # Get customer segment and purchases
        segment = self.user_segment.get(user_id, "General/Hibernating")
        purchases = self.user_purchases.get(user_id, set())
        
        # If user has no history, recommend popular products in their segment (cold-start)
        if not purchases:
            return self._recommend_cold_start(segment, top_k)
            
        # Candidates storage
        candidates = defaultdict(lambda: {
            'mba_score': 0.0, 
            'sim_score': 0.0, 
            'pop_score': 0.0,
            'explanations': []
        })
        
        # 1. GENERATE MBA CANDIDATES
        seg_key = self._get_segment_key(segment)
        rules_df = self.segment_rules.get(seg_key, pd.DataFrame())
        
        if not rules_df.empty:
            # Find rules where antecedent is in user history
            matching_rules = rules_df[rules_df['antecedent_id'].isin(purchases)]
            for _, row in matching_rules.iterrows():
                conseq = int(row['consequent_id'])
                ante = int(row['antecedent_id'])
                
                # Skip if already bought
                if conseq in purchases:
                    continue
                    
                # Support/Lift score
                score = row['confidence'] * min(row['lift'] / 10.0, 1.0)
                
                # Keep maximum rule score
                if score > candidates[conseq]['mba_score']:
                    candidates[conseq]['mba_score'] = score
                    
                explanation = f"Bought by other users in your segment '{segment}' who also bought '{self.prod_name.get(ante, 'Unknown')}'"
                candidates[conseq]['explanations'].append(explanation)

        # 2. GENERATE SIMILARITY CANDIDATES (Content-based)
        for pid in purchases:
            if pid not in self.embeddings:
                continue
            v_pid = self.embeddings[pid]
            
            for target_id, v_target in self.embeddings.items():
                # Skip if already bought
                if target_id in purchases or target_id == pid:
                    continue
                    
                sim = self._cosine_similarity(v_pid, v_target)
                
                # Keep positive similarity
                if sim > 0.3:
                    # Update similarity score (keep max similarity)
                    if sim > candidates[target_id]['sim_score']:
                        candidates[target_id]['sim_score'] = sim
                        
                    # Add explanation
                    explanation = f"Similar ingredients or features to '{self.prod_name.get(pid, 'Unknown')}' ({sim*100:.0f}% match)"
                    candidates[target_id]['explanations'].append(explanation)

        # 3. ADD POPULARITY SCORE
        for target_id in list(candidates.keys()):
            sales = self.prod_sales.get(target_id, 0.0) / self.max_sales
            rating = (self.prod_rating.get(target_id, 4.0) - 1.0) / 4.0  # normalize rating 1-5 to 0-1
            pop = 0.6 * sales + 0.4 * rating
            candidates[target_id]['pop_score'] = pop

        # 4. FUSE SCORES & RANK
        results = []
        for pid, data in candidates.items():
            final_score = (w_mba * data['mba_score'] + 
                           w_sim * data['sim_score'] + 
                           w_pop * data['pop_score'])
            
            # Select the primary explanation
            primary_exp = ""
            if data['mba_score'] > 0 and data['sim_score'] > 0:
                # Combine best parts
                primary_exp = data['explanations'][0] + " (also shares similar skin type/features)"
            elif data['mba_score'] > 0:
                primary_exp = data['explanations'][0]
            elif data['sim_score'] > 0:
                # Sort explanations by similarity
                primary_exp = data['explanations'][0]
            else:
                primary_exp = "Recommended popular item in your category"
                
            results.append({
                'product_id': pid,
                'product_name': self.prod_name.get(pid, 'Unknown'),
                'brand': self.prod_brand.get(pid, 'Unknown'),
                'price': self.prod_price.get(pid, 0.0),
                'score': final_score,
                'explanation': primary_exp
            })
            
        df_results = pd.DataFrame(results)
        if not df_results.empty:
            df_results = df_results.sort_values(by='score', ascending=False).head(top_k).reset_index(drop=True)
            return df_results
        else:
            return self._recommend_cold_start(segment, top_k)

    def _recommend_cold_start(self, segment, top_k=5):
        """Recommend best-selling products for users with no transaction history."""
        # Find products popular in this segment or globally
        df_sorted = self.df_mapping.sort_values(by=['num_sold_time', 'avg_star'], ascending=False).head(top_k)
        
        results = []
        for _, row in df_sorted.iterrows():
            pid = int(row['product_id'])
            results.append({
                'product_id': pid,
                'product_name': row['product_name'],
                'brand': row['brand'],
                'price': row['price'],
                'score': 1.0,
                'explanation': f"Top best-selling product on Shopee skincare (Popular for segment: {segment})"
            })
        return pd.DataFrame(results)
