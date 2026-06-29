# -*- coding: utf-8 -*-
"""
Phase 6: Segment-Aware Hybrid Recommendation Engine Evaluation
=============================================================
Instantiates the hybrid recommendation engine, generates sample
recommendations for users in each segment, evaluates list diversity,
coverage, and saves tabular outputs.
"""

import os
import sys
import pandas as pd
import numpy as np
import traceback

# Ensure src path is in sys.path
sys.path.insert(0, '.')

from src.recommendation.recommender import SegmentAwareHybridRecommender

LOG_FILE = 'outputs/recommendation_log.txt'
os.makedirs('outputs', exist_ok=True)
os.makedirs('outputs/tables', exist_ok=True)

log = open(LOG_FILE, 'w', encoding='utf-8')

def p(msg):
    log.write(str(msg) + '\n')
    log.flush()

try:
    p("=" * 60)
    p("PHASE 6: HYBRID RECOMMENDATION ENGINE EVALUATION")
    p("=" * 60)

    # 1. Initialize Recommender
    p("\n[1/3] Initializing Segment-Aware Recommender...")
    recommender = SegmentAwareHybridRecommender()
    p("  Recommender initialized successfully.")
    p(f"  Loaded transaction histories for {len(recommender.user_purchases):,} users.")
    p(f"  Loaded category metadata for {len(recommender.prod_name):,} products.")
    p(f"  Segment rules pre-loaded: {list(recommender.segment_rules.keys())}")

    # 2. Test Recommendation for representative users of each segment
    p("\n[2/3] Generating sample recommendations for each customer segment...")
    
    # Select sample users from each segment
    df_segmented = recommender.df_segmented
    segments = df_segmented['segment_name'].unique()
    
    sample_users = []
    for seg in segments:
        # Find users with at least 2 purchases in this segment for interesting rules
        seg_users = df_segmented[df_segmented['segment_name'] == seg]['user_id'].tolist()
        active_seg_users = [uid for uid in seg_users if uid in recommender.user_purchases and len(recommender.user_purchases[uid]) >= 2]
        
        if active_seg_users:
            uid = active_seg_users[0]
        else:
            uid = seg_users[0]  # fallback to first user
            
        sample_users.append((seg, uid))
        
    p(f"  Selected {len(sample_users)} sample users for evaluation.")
    
    all_sample_recs = []
    
    for seg, uid in sample_users:
        p(f"\n  -------------------------------------------------------------")
        p(f"  User ID: {uid} | Segment: {seg}")
        history = list(recommender.user_purchases.get(uid, set()))
        history_names = [recommender.prod_name.get(pid, 'Unknown') for pid in history]
        p(f"  Purchase History ({len(history)} products):")
        for idx, h_name in enumerate(history_names[:5]):
            p(f"    - {h_name}")
        if len(history_names) > 5:
            p(f"    - ... and {len(history_names)-5} more")
            
        # Get recommendations
        recs = recommender.recommend(user_id=uid, top_k=5)
        p(f"  Recommended Products:")
        for idx, row in recs.iterrows():
            p(f"    {idx+1}. [{row['brand']}] {row['product_name']} (Score: {row['score']:.4f})")
            p(f"       💡 Explanation: {row['explanation']}")
            
            # Save row for CSV
            all_sample_recs.append({
                'segment': seg,
                'user_id': uid,
                'rank': idx + 1,
                'product_id': row['product_id'],
                'product_name': row['product_name'],
                'brand': row['brand'],
                'price': row['price'],
                'score': row['score'],
                'explanation': row['explanation']
            })
            
    # Save sample recommendations sheet
    df_sample_recs = pd.DataFrame(all_sample_recs)
    sample_recs_path = 'outputs/tables/recommender_samples.csv'
    df_sample_recs.to_csv(sample_recs_path, index=False)
    p(f"\n  Saved sample recommendation runs: {sample_recs_path}")

    # 3. Calculate Engine Evaluation Metrics (Coverage & Diversity)
    p("\n[3/3] Evaluating recommendation engine quality...")
    
    # Select random users for batch evaluation
    np.random.seed(42)
    eval_uids = np.random.choice(list(recommender.user_purchases.keys()), size=200, replace=False)
    
    all_recs_pids = set()
    rule_hits = 0
    sim_hits = 0
    list_similarities = []
    
    p(f"  Running batch evaluation on 200 random users...")
    for uid in eval_uids:
        recs = recommender.recommend(user_id=int(uid), top_k=5)
        if recs.empty:
            continue
        pids = recs['product_id'].tolist()
        all_recs_pids.update(pids)
        
        # Check explanation types
        exps = recs['explanation'].tolist()
        for e in exps:
            if "Bought by other users" in e:
                rule_hits += 1
            if "Similar ingredients" in e:
                sim_hits += 1
                
        # Calculate list diversity (average cosine similarity of recommended items)
        if len(pids) >= 2:
            sims = []
            for i in range(len(pids)):
                for j in range(i+1, len(pids)):
                    v1 = recommender.embeddings.get(pids[i], None)
                    v2 = recommender.embeddings.get(pids[j], None)
                    if v1 is not None and v2 is not None:
                        sims.append(recommender._cosine_similarity(v1, v2))
            if sims:
                list_similarities.append(np.mean(sims))
                
    # Metrics computation
    catalog_coverage = len(all_recs_pids) / len(recommender.prod_name) * 100
    list_similarity = np.mean(list_similarities) if list_similarities else 0.0
    list_diversity = 1.0 - list_similarity  # 1 - avg_similarity
    
    p(f"\n  Recommendation Engine Performance Metrics:")
    p(f"    1. Catalog Coverage: {catalog_coverage:.2f}% ({len(all_recs_pids)} out of {len(recommender.prod_name)} products recommended)")
    p(f"    2. Recommendation Diversity (1 - Avg Similarity): {list_diversity:.4f} (higher means recommended items are less redundant)")
    p(f"    3. MBA Rule Hit Rate: {rule_hits / (200 * 5) * 100:.1f}% (percentage of recommendations driven by association rules)")
    p(f"    4. Content Similarity Hit Rate: {sim_hits / (200 * 5) * 100:.1f}% (percentage of recommendations driven by content similarity)")

    # Save metrics to CSV
    metrics_df = pd.DataFrame([{
        'catalog_coverage_pct': catalog_coverage,
        'list_diversity': list_diversity,
        'mba_hit_rate_pct': rule_hits / (200 * 5) * 100,
        'content_hit_rate_pct': sim_hits / (200 * 5) * 100
    }])
    metrics_df.to_csv('outputs/tables/recommender_evaluation_metrics.csv', index=False)
    p("  Saved: outputs/tables/recommender_evaluation_metrics.csv")

    p("\n" + "=" * 60)
    p("PHASE 6 COMPLETE - Hybrid recommendation engine evaluation successful!")
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
