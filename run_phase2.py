# -*- coding: utf-8 -*-
"""
Phase 2: Sentiment Analysis & SA-RFM Computation
=================================================
Two modes:
  - MODE 1 (default): Rating-based sentiment (fast, no GPU needed)
  - MODE 2 (enhanced): PhoBERT-based sentiment (requires GPU, transformers)

The SA-RFM module adds the Sentiment (S) dimension to the RFM model,
creating the novel 4-dimensional RFMS customer representation.
"""

import os
import csv
import math
import traceback
from collections import defaultdict
from datetime import datetime

LOG_FILE = 'outputs/sentiment_log.txt'
os.makedirs('outputs', exist_ok=True)

try:
    log = open(LOG_FILE, 'w', encoding='utf-8')
except PermissionError:
    try:
        if os.path.exists(LOG_FILE):
            os.remove(LOG_FILE)
        log = open(LOG_FILE, 'w', encoding='utf-8')
    except Exception:
        LOG_FILE = 'sentiment_log.txt'
        log = open(LOG_FILE, 'w', encoding='utf-8')

def p(msg):
    log.write(str(msg) + '\n')
    log.flush()


def compute_rating_based_sentiment():
    """
    MODE 1: Derive sentiment from the rating column.
    
    Mapping: 1→0.0, 2→0.25, 3→0.5, 4→0.75, 5→1.0
    
    For reviews with empty comments, use the rating directly.
    This provides a reliable baseline sentiment score.
    """
    p("\n--- Rating-Based Sentiment ---")
    
    rating_to_sentiment = {
        '1': 0.0, '1.0': 0.0,
        '2': 0.25, '2.0': 0.25,
        '3': 0.5, '3.0': 0.5,
        '4': 0.75, '4.0': 0.75,
        '5': 1.0, '5.0': 1.0,
    }
    
    reviews_path = 'dataset/after_EDA/reviews_cleaned.csv'
    
    # Collect per-user sentiment with recency weighting
    user_sentiments = defaultdict(lambda: {'weighted_sum': 0.0, 'weight_total': 0.0, 'count': 0})
    
    # We need the max date for recency weighting
    max_date = datetime(2023, 1, 7)  # From Phase 1 analysis
    
    total = 0
    with open(reviews_path, 'r', encoding='utf-8') as f:
        reader = csv.DictReader(f)
        for row in reader:
            total += 1
            uid = row['user_id']
            rating = row['rating'].strip()
            sentiment = rating_to_sentiment.get(rating, 0.5)
            
            # Recency weight: more recent reviews count more
            try:
                dt = datetime.strptime(row['cmt_date'][:10], '%Y-%m-%d')
                days_ago = (max_date - dt).days
                # Exponential decay: weight = exp(-days_ago / 365)
                weight = math.exp(-days_ago / 365.0)
            except (ValueError, KeyError):
                weight = 0.5  # default weight for unparseable dates
            
            ud = user_sentiments[uid]
            ud['weighted_sum'] += sentiment * weight
            ud['weight_total'] += weight
            ud['count'] += 1
            
            if total % 100000 == 0:
                p(f"  Processed {total:,} reviews...")
    
    p(f"  Total reviews processed: {total:,}")
    p(f"  Unique users: {len(user_sentiments):,}")
    
    return user_sentiments


def try_phobert_sentiment():
    """
    MODE 2: PhoBERT-based sentiment analysis.
    
    Requires: pip install transformers torch underthesea
    
    Uses vinai/phobert-base with a sentiment classification head.
    Falls back to MODE 1 if dependencies are not available.
    """
    try:
        import torch
        from transformers import AutoTokenizer, AutoModelForSequenceClassification
        
        p("\n--- PhoBERT Sentiment Analysis ---")
        p("  Loading PhoBERT model...")
        
        # Use a Vietnamese sentiment model
        # Options:
        #   1. wonrax/phobert-base-vietnamese-sentiment (3-class: NEG, NEU, POS)
        #   2. cardiffnlp/twitter-xlm-roberta-base-sentiment (multilingual)
        model_name = "wonrax/phobert-base-vietnamese-sentiment"
        tokenizer = AutoTokenizer.from_pretrained(model_name)
        model = AutoModelForSequenceClassification.from_pretrained(model_name)
        model.eval()
        
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        model = model.to(device)
        p(f"  Model loaded on {device}")
        
        # Process reviews in batches
        reviews_path = 'dataset/after_EDA/reviews_cleaned.csv'
        max_date = datetime(2023, 1, 7)
        
        user_sentiments = defaultdict(lambda: {'weighted_sum': 0.0, 'weight_total': 0.0, 'count': 0})
        
        batch_texts = []
        batch_meta = []  # (uid, weight)
        batch_size = 32
        total = 0
        
        def process_batch(texts, metas):
            if not texts:
                return
            inputs = tokenizer(texts, return_tensors="pt", truncation=True,
                             padding=True, max_length=256).to(device)
            with torch.no_grad():
                outputs = model(**inputs)
                probs = torch.softmax(outputs.logits, dim=1)
            
            # Model output: [NEG, NEU, POS] probabilities
            # Map to sentiment score: NEG→0.0, NEU→0.5, POS→1.0
            for i, (uid, weight) in enumerate(metas):
                neg, neu, pos = probs[i].cpu().tolist()
                sentiment = neg * 0.0 + neu * 0.5 + pos * 1.0
                ud = user_sentiments[uid]
                ud['weighted_sum'] += sentiment * weight
                ud['weight_total'] += weight
                ud['count'] += 1
        
        with open(reviews_path, 'r', encoding='utf-8') as f:
            reader = csv.DictReader(f)
            for row in reader:
                total += 1
                comment = row.get('processed_comment', '').strip()
                
                if not comment:
                    # No comment: use rating as fallback
                    rating_map = {'1': 0.0, '2': 0.25, '3': 0.5, '4': 0.75, '5': 1.0}
                    sentiment = rating_map.get(row['rating'].strip(), 0.5)
                    try:
                        dt = datetime.strptime(row['cmt_date'][:10], '%Y-%m-%d')
                        weight = math.exp(-(max_date - dt).days / 365.0)
                    except ValueError:
                        weight = 0.5
                    uid = row['user_id']
                    ud = user_sentiments[uid]
                    ud['weighted_sum'] += sentiment * weight
                    ud['weight_total'] += weight
                    ud['count'] += 1
                else:
                    try:
                        dt = datetime.strptime(row['cmt_date'][:10], '%Y-%m-%d')
                        weight = math.exp(-(max_date - dt).days / 365.0)
                    except ValueError:
                        weight = 0.5
                    
                    batch_texts.append(comment[:256])  # truncate long comments
                    batch_meta.append((row['user_id'], weight))
                    
                    if len(batch_texts) >= batch_size:
                        process_batch(batch_texts, batch_meta)
                        batch_texts = []
                        batch_meta = []
                
                if total % 10000 == 0:
                    p(f"  Processed {total:,} reviews...")
        
        # Process remaining batch
        process_batch(batch_texts, batch_meta)
        
        p(f"  PhoBERT sentiment complete: {total:,} reviews, {len(user_sentiments):,} users")
        return user_sentiments
        
    except ImportError as e:
        p(f"  PhoBERT not available ({e}), falling back to rating-based sentiment")
        return None
    except Exception as e:
        p(f"  PhoBERT error: {e}, falling back to rating-based sentiment")
        return None


def build_sarfm(user_sentiments):
    """
    Build the SA-RFM table by merging RFM with Sentiment scores.
    
    Output columns: user_id, recency, frequency, monetary, sentiment,
                    recency_norm, frequency_norm, monetary_norm, sentiment_norm
    """
    import pandas as pd  # safe for small DataFrames
    
    p("\n--- Building SA-RFM Table ---")
    
    # Load RFM table from Phase 1
    rfm = pd.read_csv('dataset/after_EDA/rfm_table.csv')
    p(f"  RFM table loaded: {len(rfm):,} users")
    
    # Compute sentiment score per user
    sentiment_rows = []
    for uid, ud in user_sentiments.items():
        if ud['weight_total'] > 0:
            s = ud['weighted_sum'] / ud['weight_total']
        else:
            s = 0.5  # neutral default
        sentiment_rows.append({'user_id': int(uid), 'sentiment': s})
    
    sentiment_df = pd.DataFrame(sentiment_rows)
    p(f"  Sentiment scores computed: {len(sentiment_df):,} users")
    
    # Merge with RFM
    sarfm = rfm.merge(sentiment_df, on='user_id', how='left')
    sarfm['sentiment'] = sarfm['sentiment'].fillna(0.5)  # neutral for users without sentiment
    
    # Normalize sentiment (already in [0, 1] range, but apply Min-Max for consistency)
    s_min, s_max = sarfm['sentiment'].min(), sarfm['sentiment'].max()
    if s_max > s_min:
        sarfm['sentiment_norm'] = (sarfm['sentiment'] - s_min) / (s_max - s_min)
    else:
        sarfm['sentiment_norm'] = 0.5
    
    # Summary statistics
    p(f"\n  SA-RFM Table: {len(sarfm):,} users x {sarfm.shape[1]} columns")
    p(f"\n  Sentiment statistics:")
    p(f"    mean={sarfm['sentiment'].mean():.4f}")
    p(f"    std={sarfm['sentiment'].std():.4f}")
    p(f"    min={sarfm['sentiment'].min():.4f}")
    p(f"    25%={sarfm['sentiment'].quantile(0.25):.4f}")
    p(f"    50%={sarfm['sentiment'].quantile(0.5):.4f}")
    p(f"    75%={sarfm['sentiment'].quantile(0.75):.4f}")
    p(f"    max={sarfm['sentiment'].max():.4f}")
    
    # Distribution buckets
    p(f"\n  Sentiment distribution:")
    p(f"    Very Negative (0.0-0.2): {(sarfm['sentiment'] <= 0.2).sum():,}")
    p(f"    Negative (0.2-0.4): {((sarfm['sentiment'] > 0.2) & (sarfm['sentiment'] <= 0.4)).sum():,}")
    p(f"    Neutral (0.4-0.6): {((sarfm['sentiment'] > 0.4) & (sarfm['sentiment'] <= 0.6)).sum():,}")
    p(f"    Positive (0.6-0.8): {((sarfm['sentiment'] > 0.6) & (sarfm['sentiment'] <= 0.8)).sum():,}")
    p(f"    Very Positive (0.8-1.0): {(sarfm['sentiment'] > 0.8).sum():,}")
    
    # Save
    sarfm.to_csv('dataset/after_EDA/sarfm_table.csv', index=False)
    size = os.path.getsize('dataset/after_EDA/sarfm_table.csv')
    p(f"\n  Saved: sarfm_table.csv ({size/1024/1024:.1f} MB)")
    
    # Also save the normalized 4D vectors for clustering
    sarfm_vectors = sarfm[['user_id', 'recency_norm', 'frequency_norm', 'monetary_norm', 'sentiment_norm']]
    sarfm_vectors.to_csv('dataset/after_EDA/sarfm_vectors.csv', index=False)
    size2 = os.path.getsize('dataset/after_EDA/sarfm_vectors.csv')
    p(f"  Saved: sarfm_vectors.csv ({size2/1024/1024:.1f} MB)")
    
    return sarfm


# ===========================================================================
# MAIN
# ===========================================================================

try:
    p("=" * 60)
    p("PHASE 2: SENTIMENT ANALYSIS & SA-RFM")
    p("=" * 60)
    
    # Try PhoBERT first, fall back to rating-based
    user_sentiments = try_phobert_sentiment()
    
    if user_sentiments is None:
        p("\n  Using rating-based sentiment (MODE 1)")
        user_sentiments = compute_rating_based_sentiment()
    else:
        p("\n  Using PhoBERT sentiment (MODE 2)")
    
    # Build SA-RFM table
    sarfm = build_sarfm(user_sentiments)
    
    p("\n" + "=" * 60)
    p("PHASE 2 COMPLETE")
    p("=" * 60)

except Exception as e:
    p(f"\n!!! ERROR: {e}")
    p(traceback.format_exc())

finally:
    log.close()
