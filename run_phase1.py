# -*- coding: utf-8 -*-
"""
Phase 1: Data Preprocessing & RFM Engineering
Uses chunked/pure-python approach for the groupby to avoid pandas segfault.
"""

import pandas as pd
import numpy as np
import re
import os
import csv
import traceback
from datetime import datetime
from collections import defaultdict

LOG_FILE = 'outputs/preprocessing_log.txt'
os.makedirs('outputs', exist_ok=True)
os.makedirs('dataset/after_EDA', exist_ok=True)

try:
    log = open(LOG_FILE, 'w', encoding='utf-8')
except PermissionError:
    try:
        if os.path.exists(LOG_FILE):
            os.remove(LOG_FILE)
        log = open(LOG_FILE, 'w', encoding='utf-8')
    except Exception:
        LOG_FILE = 'preprocessing_log.txt'
        log = open(LOG_FILE, 'w', encoding='utf-8')

def p(msg):
    log.write(str(msg) + '\n')
    log.flush()

try:
    p("=" * 60)
    p("PHASE 1: DATA PREPROCESSING & RFM ENGINEERING")
    p("=" * 60)

    # ===================================================================
    # STEP 1: LOAD RAW DATA (small tables with pandas, large table later)
    # ===================================================================
    p("\n[1/7] Loading raw data...")
    
    products = pd.read_csv('dataset/before_EDA/data_product.csv')
    attributes = pd.read_csv('dataset/before_EDA/data_product_attribute.csv')
    shops = pd.read_csv('dataset/before_EDA/data_shop.csv')
    
    p(f"  products: {products.shape[0]:,} rows x {products.shape[1]} cols")
    p(f"  attributes: {attributes.shape[0]:,} rows x {attributes.shape[1]} cols")
    p(f"  shops: {shops.shape[0]:,} rows x {shops.shape[1]} cols")

    # ===================================================================
    # STEP 2: CLEAN PRODUCTS & ATTRIBUTES (small tables - safe with pandas)
    # ===================================================================
    p("\n[2/7] Cleaning products & attributes...")
    
    def clean_text(text):
        if pd.isna(text) or not isinstance(text, str):
            return ''
        text = text.strip().lower()
        text = re.sub(r'http\S+|www\.\S+', '', text)
        text = re.sub(r'\s+', ' ', text)
        return text.strip()
    
    products['processed_description'] = products['processed_description'].apply(clean_text)
    
    for col in ['ingredient', 'feature', 'skin_type']:
        attributes[col] = attributes[col].apply(clean_text)
    
    # Handle missing values in attributes
    attributes['ingredient'] = attributes['ingredient'].replace('', 'unknown')
    attributes['skin_type'] = attributes['skin_type'].replace('', 'all_skin')
    attributes['capacity'] = attributes['capacity'].fillna('unknown')
    attributes['design'] = attributes['design'].fillna('unknown')
    attributes['brand'] = attributes['brand'].fillna('no_brand')
    attributes['expiry'] = attributes['expiry'].fillna('unknown')
    attributes['origin'] = attributes['origin'].fillna('unknown')
    
    p("  Products & attributes cleaned and imputed")

    # ===================================================================
    # STEP 3: BUILD PRICE LOOKUP
    # ===================================================================
    p("\n[3/7] Building price lookup...")
    price_lookup = dict(zip(products['product_id'].astype(int), products['price'].astype(float)))
    p(f"  Price lookup: {len(price_lookup)} products")

    # ===================================================================
    # STEP 4: PROCESS REVIEWS LINE-BY-LINE (avoid pandas segfault)
    # ===================================================================
    p("\n[4/7] Processing reviews (line-by-line for stability)...")
    
    reviews_input = 'dataset/before_EDA/data_reviews_purchase.csv'
    reviews_output = 'dataset/after_EDA/reviews_cleaned.csv'
    
    # First pass: clean & save reviews, collect RFM data
    user_data = defaultdict(lambda: {'max_date': None, 'count': 0, 'total': 0.0})
    max_date_global = None
    total_rows = 0
    empty_comments = 0
    
    with open(reviews_input, 'r', encoding='utf-8') as fin, \
         open(reviews_output, 'w', encoding='utf-8', newline='') as fout:
        
        reader = csv.DictReader(fin)
        
        # Output columns (drop Unnamed: 0, add date features)
        out_cols = ['user_id', 'product_id', 'rating', 'product_name_x',
                    'cmt_date', 'shop_id', 'variation_x', 'product_quality',
                    'processed_comment',
                    'purchase_year', 'purchase_month', 'purchase_day_of_week', 'purchase_hour']
        
        writer = csv.DictWriter(fout, fieldnames=out_cols)
        writer.writeheader()
        
        for row in reader:
            total_rows += 1
            
            # Clean comment
            comment = row.get('processed_comment', '')
            if not comment or comment.strip() == '':
                comment = ''
                empty_comments += 1
            else:
                comment = clean_text(comment)
            
            # Parse date
            dt_str = row['cmt_date']
            try:
                dt = datetime.strptime(dt_str, '%Y-%m-%d %H:%M:%S')
            except ValueError:
                dt = datetime.strptime(dt_str[:19], '%Y-%m-%d %H:%M:%S')
            
            # Date features
            year = dt.year
            month = dt.month
            dow = dt.weekday()
            hour = dt.hour
            
            # RFM aggregation
            uid = row['user_id']
            pid = int(row['product_id'])
            price = price_lookup.get(pid, 0.0)
            
            ud = user_data[uid]
            if ud['max_date'] is None or dt > ud['max_date']:
                ud['max_date'] = dt
            ud['count'] += 1
            ud['total'] += price
            
            if max_date_global is None or dt > max_date_global:
                max_date_global = dt
            
            # Write cleaned row
            out_row = {
                'user_id': row['user_id'],
                'product_id': row['product_id'],
                'rating': row['rating'],
                'product_name_x': row.get('product_name_x', ''),
                'cmt_date': dt_str,
                'shop_id': row['shop_id'],
                'variation_x': row.get('variation_x', ''),
                'product_quality': row.get('product_quality', ''),
                'processed_comment': comment,
                'purchase_year': year,
                'purchase_month': month,
                'purchase_day_of_week': dow,
                'purchase_hour': hour,
            }
            writer.writerow(out_row)
            
            if total_rows % 100000 == 0:
                p(f"    Processed {total_rows:,} rows...")
    
    p(f"  Total reviews processed: {total_rows:,}")
    p(f"  Empty comments: {empty_comments:,}")
    p(f"  Date range: ... to {max_date_global}")
    p(f"  Unique users: {len(user_data):,}")

    # ===================================================================
    # STEP 5: COMPUTE RFM TABLE
    # ===================================================================
    p("\n[5/7] Computing RFM table...")
    
    snapshot_date = max_date_global
    
    rfm_rows = []
    for uid, ud in user_data.items():
        recency = (snapshot_date - ud['max_date']).days
        rfm_rows.append({
            'user_id': uid,
            'recency': recency,
            'frequency': ud['count'],
            'monetary': ud['total'],
        })
    
    # Build RFM DataFrame (small: 304K rows x 4 cols)
    rfm = pd.DataFrame(rfm_rows)
    
    p(f"  RFM table: {len(rfm):,} users")
    p(f"  Snapshot date: {snapshot_date.date()}")
    
    for col in ['recency', 'frequency', 'monetary']:
        vals = rfm[col]
        p(f"\n  {col}:")
        p(f"    mean={vals.mean():.2f}, std={vals.std():.2f}")
        p(f"    min={vals.min()}, 25%={vals.quantile(0.25):.0f}, "
          f"50%={vals.quantile(0.5):.0f}, 75%={vals.quantile(0.75):.0f}, max={vals.max()}")
    
    # ===================================================================
    # STEP 6: NORMALIZE RFM
    # ===================================================================
    p("\n[6/7] Normalizing RFM (Min-Max)...")
    
    for col in ['recency', 'frequency', 'monetary']:
        min_v, max_v = rfm[col].min(), rfm[col].max()
        if max_v > min_v:
            rfm[f'{col}_norm'] = (rfm[col] - min_v) / (max_v - min_v)
        else:
            rfm[f'{col}_norm'] = 0.0
    
    # Invert recency: low recency (recent) = high score
    rfm['recency_norm'] = 1.0 - rfm['recency_norm']
    
    p("  recency_norm: INVERTED (1.0 = most recent)")
    p("  frequency_norm: 0.0 = 1 purchase, 1.0 = 20 purchases")
    p("  monetary_norm: 0.0 = 1,000 VND, 1.0 = 4,016,000 VND")

    # ===================================================================
    # STEP 7: SAVE ALL CLEANED DATA
    # ===================================================================
    p("\n[7/7] Saving cleaned data...")
    
    out_path = 'dataset/after_EDA'
    
    # Reviews already saved in step 4
    rev_size = os.path.getsize(os.path.join(out_path, 'reviews_cleaned.csv'))
    p(f"  reviews_cleaned.csv: {rev_size/1024/1024:.1f} MB")
    
    products.to_csv(os.path.join(out_path, 'products_cleaned.csv'), index=False)
    p(f"  products_cleaned.csv: {os.path.getsize(os.path.join(out_path, 'products_cleaned.csv'))/1024/1024:.1f} MB")
    
    attributes.to_csv(os.path.join(out_path, 'attributes_cleaned.csv'), index=False)
    p(f"  attributes_cleaned.csv: {os.path.getsize(os.path.join(out_path, 'attributes_cleaned.csv'))/1024/1024:.1f} MB")
    
    shops.to_csv(os.path.join(out_path, 'shops_cleaned.csv'), index=False)
    p(f"  shops_cleaned.csv: {os.path.getsize(os.path.join(out_path, 'shops_cleaned.csv'))/1024/1024:.1f} MB")
    
    rfm.to_csv(os.path.join(out_path, 'rfm_table.csv'), index=False)
    p(f"  rfm_table.csv: {os.path.getsize(os.path.join(out_path, 'rfm_table.csv'))/1024/1024:.1f} MB")
    
    p("\n" + "=" * 60)
    p("PHASE 1 COMPLETE - All data cleaned and saved!")
    p("=" * 60)

except Exception as e:
    p(f"\n!!! ERROR: {e}")
    p(traceback.format_exc())

finally:
    log.close()
