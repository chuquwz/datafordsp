# -*- coding: utf-8 -*-
"""
Phase 5: Market Basket Analysis & Semantic Rules Mining
======================================================
Loads segment designations and product embeddings,
mines global and segment-aware association rules,
and writes tabular outputs.
"""

import os
import sys
import pandas as pd
import numpy as np
import traceback

# Ensure src path is in sys.path
sys.path.insert(0, '.')

from src.mba.mba import (
    load_mba_data,
    build_baskets,
    mine_association_rules,
    run_segment_aware_mba
)

LOG_FILE = 'outputs/mba_log.txt'
os.makedirs('outputs', exist_ok=True)
os.makedirs('outputs/tables', exist_ok=True)

log = open(LOG_FILE, 'w', encoding='utf-8')

def p(msg):
    log.write(str(msg) + '\n')
    log.flush()

try:
    p("=" * 60)
    p("PHASE 5: MARKET BASKET ANALYSIS (SEMANTIC RULES MINING)")
    p("=" * 60)

    # 1. Load Data
    p("\n[1/4] Loading segmented transactions and product embeddings...")
    df_reviews, df_segments, embeddings, prod_name, prod_brand, prod_price = load_mba_data()
    p(f"  Reviews transaction entries: {df_reviews.shape[0]:,}")
    p(f"  Segmented customers: {df_segments.shape[0]:,}")
    p(f"  Product embeddings loaded: {len(embeddings)}")

    # 2. Build Baskets
    p("\n[2/4] Constructing user-lifetime transaction baskets...")
    global_baskets = build_baskets(df_reviews)
    p(f"  Total customer baskets (size >= 2): {len(global_baskets):,}")
    
    # Calculate average basket size
    avg_size = np.mean([len(b) for b in global_baskets.values()])
    p(f"  Average customer basket size: {avg_size:.2f} products")

    # 3. Mine Global Association Rules
    p("\n[3/4] Mining global association rules...")
    # Adjust support and confidence to yield a good set of rules for sparse data
    df_global_rules = mine_association_rules(
        global_baskets, min_support=0.0005, min_confidence=0.05, min_lift=1.2,
        embeddings=embeddings, prod_name_map=prod_name, prod_brand_map=prod_brand, prod_price_map=prod_price
    )
    p(f"  Mined {len(df_global_rules)} global association rules.")
    
    if not df_global_rules.empty:
        p("\n  Top 15 Global Rules (sorted by lift):")
        display_cols = ['antecedent_name', 'consequent_name', 'support', 'confidence', 'lift', 'semantic_similarity', 'same_brand']
        p(df_global_rules[display_cols].head(15).to_string(index=False))
        
        # Save global rules
        global_rules_path = 'outputs/tables/global_association_rules.csv'
        df_global_rules.to_csv(global_rules_path, index=False)
        p(f"\n  Saved: {global_rules_path}")

    # 4. Mine Segment-Aware Association Rules
    p("\n[4/4] Mining segment-aware association rules (rules per customer segment)...")
    segment_rules = run_segment_aware_mba(
        df_reviews, df_segments, min_support=0.0005, min_confidence=0.05, min_lift=1.2,
        embeddings=embeddings, prod_name_map=prod_name, prod_brand_map=prod_brand, prod_price_map=prod_price
    )
    
    # Save and display rules per segment
    for seg, df_seg_rules in segment_rules.items():
        p(f"\n  Segment: {seg}")
        p(f"    Mined rules: {len(df_seg_rules)}")
        if not df_seg_rules.empty:
            p(df_seg_rules[['antecedent_name', 'consequent_name', 'support', 'confidence', 'lift', 'semantic_similarity']].head(5).to_string(index=False))
            # Save segment rules
            import re
            clean_name = re.sub(r'[^\w\s-]', '', seg).strip()
            clean_name = re.sub(r'\s+', '_', clean_name).lower()
            seg_rules_path = f'outputs/tables/segment_rules_{clean_name}.csv'
            df_seg_rules.to_csv(seg_rules_path, index=False)
            p(f"    Saved: {seg_rules_path}")

    p("\n" + "=" * 60)
    p("PHASE 5 COMPLETE - Market basket analysis successful!")
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
