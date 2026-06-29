# -*- coding: utf-8 -*-
"""
Phase 5: Market Basket Analysis (Semantic MBA)
==============================================
Constructs user-lifetime baskets, mines association rules using an optimized
pure-Python Apriori algorithm (custom-built to run without mlxtend),
and enhances the rules with product semantic similarity.
Also supports segment-aware rule mining (rules per customer segment).
"""

import os
import numpy as np
import pandas as pd
from collections import defaultdict, Counter

def load_mba_data(reviews_path='dataset/after_EDA/reviews_cleaned.csv',
                  segmented_path='dataset/after_EDA/sarfm_segmented.csv',
                  embeddings_path='dataset/after_EDA/product_embeddings.csv',
                  mapping_path='dataset/after_EDA/product_mapping.csv'):
    """Load reviews, segments, embeddings, and product name mappings."""
    df_r = pd.read_csv(reviews_path)
    df_s = pd.read_csv(segmented_path)
    df_e = pd.read_csv(embeddings_path)
    df_m = pd.read_csv(mapping_path)
    
    # Create product mappings for quick lookups
    prod_name_map = dict(zip(df_m['product_id'], df_m['product_name']))
    prod_brand_map = dict(zip(df_m['product_id'], df_m['brand']))
    prod_price_map = dict(zip(df_m['product_id'], df_m['price']))
    
    # Load embeddings into a dictionary of vectors
    embeddings = {}
    for _, row in df_e.iterrows():
        pid = int(row['product_id'])
        vec = row.drop('product_id').values.astype(float)
        embeddings[pid] = vec
        
    return df_r, df_s, embeddings, prod_name_map, prod_brand_map, prod_price_map

def build_baskets(df_reviews, user_ids=None):
    """
    Construct user-lifetime baskets of product_ids.
    Optionally filters by a subset of user_ids (for segment-aware MBA).
    """
    df = df_reviews
    if user_ids is not None:
        df = df[df['user_id'].isin(user_ids)]
        
    # Group by user_id and aggregate product_ids as a set (to get unique products per user)
    user_prods = df.groupby('user_id')['product_id'].apply(set).to_dict()
    
    # Filter baskets with size >= 2
    baskets = {uid: prods for uid, prods in user_prods.items() if len(prods) >= 2}
    return baskets

def cosine_similarity(v1, v2):
    """Calculate cosine similarity between two vectors."""
    dot = np.dot(v1, v2)
    norm1 = np.linalg.norm(v1)
    norm2 = np.linalg.norm(v2)
    if norm1 > 0 and norm2 > 0:
        return dot / (norm1 * norm2)
    return 0.0

def mine_association_rules(baskets, min_support=0.001, min_confidence=0.1, min_lift=1.2, 
                           embeddings=None, prod_name_map=None, prod_brand_map=None, prod_price_map=None):
    """
    Optimized pure-Python association rule mining for size-2 pairs (Apriori).
    Outputs rules with support, confidence, lift, and semantic similarity.
    """
    num_baskets = len(baskets)
    if num_baskets == 0:
        return pd.DataFrame()
        
    # Step 1: Count frequent 1-itemsets
    item_counts = Counter()
    for basket in baskets.values():
        item_counts.update(basket)
        
    # Keep items that meet min_support
    min_support_count = min_support * num_baskets
    frequent_items = {item for item, cnt in item_counts.items() if cnt >= min_support_count}
    
    # Step 2: Generate and count frequent 2-itemsets (pairs)
    pair_counts = Counter()
    for basket in baskets.values():
        # Only check pairs among frequent items to save time
        freq_basket = [item for item in basket if item in frequent_items]
        n_items = len(freq_basket)
        if n_items < 2:
            continue
        # Generate combinations manually to avoid dependency
        for i in range(n_items):
            for j in range(i + 1, n_items):
                item1, item2 = freq_basket[i], freq_basket[j]
                pair = (item1, item2) if item1 < item2 else (item2, item1)
                pair_counts[pair] += 1
                
    # Filter pairs that meet min_support
    frequent_pairs = {pair: cnt for pair, cnt in pair_counts.items() if cnt >= min_support_count}
    
    # Step 3: Generate rules
    rules = []
    for (itemA, itemB), pair_count in frequent_pairs.items():
        support_pair = pair_count / num_baskets
        
        # Calculate support for A and B individually
        supportA = item_counts[itemA] / num_baskets
        supportB = item_counts[itemB] / num_baskets
        
        # Rule A -> B
        confidence_A_to_B = pair_count / item_counts[itemA]
        lift_A_to_B = confidence_A_to_B / supportB
        
        # Rule B -> A
        confidence_B_to_A = pair_count / item_counts[itemB]
        lift_B_to_A = confidence_B_to_A / supportA
        
        # Evaluate Rule A -> B
        if confidence_A_to_B >= min_confidence and lift_A_to_B >= min_lift:
            rules.append((itemA, itemB, support_pair, confidence_A_to_B, lift_A_to_B))
            
        # Evaluate Rule B -> A
        if confidence_B_to_A >= min_confidence and lift_B_to_A >= min_lift:
            rules.append((itemB, itemA, support_pair, confidence_B_to_A, lift_B_to_A))
            
    # Step 4: Build DataFrame and enhance with names and semantics
    rule_rows = []
    for ante, conseq, supp, conf, lift in rules:
        row = {
            'antecedent_id': ante,
            'consequent_id': conseq,
            'support': supp,
            'confidence': conf,
            'lift': lift
        }
        
        # Name mappings
        if prod_name_map:
            row['antecedent_name'] = prod_name_map.get(ante, 'Unknown')
            row['consequent_name'] = prod_name_map.get(conseq, 'Unknown')
        if prod_brand_map:
            row['antecedent_brand'] = prod_brand_map.get(ante, 'Unknown')
            row['consequent_brand'] = prod_brand_map.get(conseq, 'Unknown')
            # Category coherence
            row['same_brand'] = (row['antecedent_brand'] == row['consequent_brand'])
        if prod_price_map:
            row['antecedent_price'] = prod_price_map.get(ante, 0.0)
            row['consequent_price'] = prod_price_map.get(conseq, 0.0)
            
        # Semantic similarity
        if embeddings and ante in embeddings and conseq in embeddings:
            row['semantic_similarity'] = cosine_similarity(embeddings[ante], embeddings[conseq])
        else:
            row['semantic_similarity'] = 0.0
            
        rule_rows.append(row)
        
    df_rules = pd.DataFrame(rule_rows)
    if not df_rules.empty:
        df_rules = df_rules.sort_values(by='lift', ascending=False).reset_index(drop=True)
    return df_rules

def run_segment_aware_mba(df_reviews, df_segments, min_support=0.001, min_confidence=0.1, min_lift=1.2,
                          embeddings=None, prod_name_map=None, prod_brand_map=None, prod_price_map=None):
    """
    Run association rule mining separately for each customer segment.
    Returns a dictionary of DataFrames: segment_name -> rules_df
    """
    segments = df_segments['segment_name'].unique()
    segment_rules = {}
    
    for seg in segments:
        # Get users in this segment
        uids = df_segments[df_segments['segment_name'] == seg]['user_id'].tolist()
        
        # Build baskets
        baskets = build_baskets(df_reviews, user_ids=uids)
        
        # Mine rules
        rules_df = mine_association_rules(
            baskets, min_support=min_support, min_confidence=min_confidence, min_lift=min_lift,
            embeddings=embeddings, prod_name_map=prod_name_map, prod_brand_map=prod_brand_map, prod_price_map=prod_price_map
        )
        
        segment_rules[seg] = rules_df
        
    return segment_rules
