# -*- coding: utf-8 -*-
"""
Phase 4: Multimodal Product Embedding
=====================================
Processes text, image, and metadata for products, combines them,
and projects them into a 128-dimensional unified semantic space.
"""

import os
import sys
import pandas as pd
import numpy as np
import traceback

# Ensure src path is in sys.path
sys.path.insert(0, '.')

from src.embedding.product_embedding import (
    load_product_data,
    build_concatenated_text,
    extract_text_embeddings_fallback,
    try_extract_text_embeddings_dl,
    extract_image_embeddings_fallback,
    try_extract_image_embeddings_dl,
    encode_metadata,
    fuse_features
)

LOG_FILE = 'outputs/embedding_log.txt'
os.makedirs('outputs', exist_ok=True)

try:
    log = open(LOG_FILE, 'w', encoding='utf-8')
except PermissionError:
    try:
        if os.path.exists(LOG_FILE):
            os.remove(LOG_FILE)
        log = open(LOG_FILE, 'w', encoding='utf-8')
    except Exception:
        LOG_FILE = 'embedding_log.txt'
        log = open(LOG_FILE, 'w', encoding='utf-8')

def p(msg):
    log.write(str(msg) + '\n')
    log.flush()

try:
    p("=" * 60)
    p("PHASE 4: MULTIMODAL PRODUCT EMBEDDING")
    p("=" * 60)

    # 1. Load Data
    p("\n[1/5] Loading products and attributes data...")
    df_product = load_product_data()
    p(f"  Merged catalog size: {df_product.shape[0]:,} products x {df_product.shape[1]} columns")

    # 2. Text Representation
    p("\n[2/5] Extracting product text representations...")
    texts = build_concatenated_text(df_product)
    p(f"  Built concatenated texts. Average character length: {np.mean([len(t) for t in texts]):.1f}")
    
    # Try DL first, fall back to TF-IDF
    p("  Attempting deep learning sentence embeddings (MiniLM)...")
    text_emb = try_extract_text_embeddings_dl(texts, n_components=128)
    if text_emb is None:
        p("  SentenceTransformers not available. Using TF-IDF + PCA fallback.")
        text_emb, svd_txt = extract_text_embeddings_fallback(texts, n_components=128)
    else:
        p("  Successfully extracted SentenceTransformers text embeddings.")
    p(f"  Text representation shape: {text_emb.shape}")

    # 3. Image Representation
    p("\n[3/5] Extracting product image representations...")
    image_paths = df_product['image_path'].tolist()
    p(f"  Checking image filenames in catalog. Found {len(image_paths)} image entries.")
    
    # Try DL first (will fail because actual images are not provided)
    img_emb = try_extract_image_embeddings_dl(image_paths, n_components=128)
    if img_emb is None:
        p("  Image files not present on disk. Using hash-based mock visual features.")
        img_emb = extract_image_embeddings_fallback(image_paths, n_components=128)
    else:
        p("  Successfully extracted deep learning visual embeddings.")
    p(f"  Image representation shape: {img_emb.shape}")

    # 4. Metadata Encoding
    p("\n[4/5] Encoding categorical and numerical metadata...")
    meta_features = encode_metadata(df_product)
    p(f"  Metadata feature shape: {meta_features.shape}")

    # 5. Multimodal Feature Fusion
    p("\n[5/5] Performing multimodal feature fusion and dimensionality reduction...")
    fused_emb, pca_fuse = fuse_features(text_emb, img_emb, meta_features, out_dim=128)
    p(f"  Final fused product embedding matrix shape: {fused_emb.shape}")
    
    # Save product embeddings to CSV
    embedding_cols = [f'dim_{i}' for i in range(128)]
    df_emb = pd.DataFrame(fused_emb, columns=embedding_cols)
    df_emb.insert(0, 'product_id', df_product['product_id'])
    
    emb_output_path = 'dataset/after_EDA/product_embeddings.csv'
    df_emb.to_csv(emb_output_path, index=False)
    p(f"\n  Saved product embeddings: {emb_output_path} ({os.path.getsize(emb_output_path)/1024/1024:.1f} MB)")

    # Save mapping file of product_id, product_name, brand, and type for downstreams
    df_mapping = df_product[['product_id', 'product_name', 'brand', 'type', 'price', 'avg_star', 'num_sold_time']]
    mapping_output_path = 'dataset/after_EDA/product_mapping.csv'
    df_mapping.to_csv(mapping_output_path, index=False)
    p(f"  Saved product mapping metadata: {mapping_output_path}")

    p("\n" + "=" * 60)
    p("PHASE 4 COMPLETE - Multimodal product embedding successful!")
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
