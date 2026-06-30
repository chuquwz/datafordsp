# -*- coding: utf-8 -*-
"""
kaggle_run_all.py
=================
Single-file Kaggle script that runs ALL 7 phases of the DSP pipeline.

HOW TO USE ON KAGGLE
--------------------
1. Create a new Kaggle Notebook (Code type).
2. Add your datasets:
      - "before_EDA" raw data  -> /kaggle/input/<SLUG_RAW>/
      - "after_EDA"  processed -> /kaggle/input/<SLUG_EDA>/  (if Phase 1 already done)
3. Add your code repo as a Dataset (zip the whole DSP folder and upload it):
      - /kaggle/input/<SLUG_CODE>/
4. Enable GPU (T4 x1 or P100) and Internet in Notebook settings.
5. In the first cell run:
      !pip install transformers torch tqdm sentence-transformers mlxtend underthesea -q
6. In the second cell run:
      import subprocess, sys
      subprocess.run([sys.executable, "/kaggle/input/<SLUG_CODE>/kaggle_run_all.py"])
   OR just paste this entire file into a cell.

CONFIGURATION  (edit the block below)
"""

import os, sys, csv, math, re, time, traceback
import pandas as pd
import numpy as np
from collections import defaultdict
from datetime import datetime

# ================================================================
# >>>  EDIT THESE THREE SLUGS TO MATCH YOUR KAGGLE DATASETS  <<<
# ================================================================
SLUG_RAW  = "your-username/tiki-before-eda"   # dataset with before_EDA/ folder
SLUG_EDA  = "your-username/tiki-after-eda"    # dataset with after_EDA/ folder (optional)
SLUG_CODE = "your-username/dsp-code"          # dataset containing src/ folder

# PhoBERT config
USE_PHOBERT        = True
PHOBERT_BATCH_SIZE = 64        # T4 -> 64, P100 -> 96
CHECKPOINT_EVERY   = 50_000

# Kaggle session = 9h -> process everything
MAX_PHOBERT_REVIEWS = None     # set an integer (e.g. 300_000) to cap
# ================================================================

# -------- Paths --------
_IN_RAW  = f"/kaggle/input/{SLUG_RAW}"
_IN_EDA  = f"/kaggle/input/{SLUG_EDA}"
_CODE    = f"/kaggle/input/{SLUG_CODE}"
_WORK    = "/kaggle/working"

# Add src to Python path
if os.path.exists(os.path.join(_CODE, "src")):
    sys.path.insert(0, _CODE)
elif os.path.exists("src"):          # running locally
    sys.path.insert(0, ".")

# Input paths
RAW_REVIEWS   = os.path.join(_IN_RAW, "before_EDA/data_reviews_purchase.csv")
RAW_PRODUCTS  = os.path.join(_IN_RAW, "before_EDA/data_product.csv")
RAW_ATTRS     = os.path.join(_IN_RAW, "before_EDA/data_product_attribute.csv")
RAW_SHOPS     = os.path.join(_IN_RAW, "before_EDA/data_shop.csv")

# If after_EDA is already uploaded (skipping Phase 1)
EDA_REVIEWS   = os.path.join(_IN_EDA, "after_EDA/reviews_cleaned.csv")
EDA_RFM       = os.path.join(_IN_EDA, "after_EDA/rfm_table.csv")

# Output paths (all go to /kaggle/working so they can be downloaded)
OUT_EDA   = os.path.join(_WORK, "after_EDA")
OUT_LOGS  = os.path.join(_WORK, "logs")
OUT_FIGS  = os.path.join(_WORK, "figures")
OUT_TABS  = os.path.join(_WORK, "tables")

for d in [OUT_EDA, OUT_LOGS, OUT_FIGS, OUT_TABS]:
    os.makedirs(d, exist_ok=True)

# -------- Logger --------
_log_file = open(os.path.join(OUT_LOGS, "pipeline.log"), "w", encoding="utf-8")

def p(msg):
    print(msg)
    _log_file.write(str(msg) + "\n")
    _log_file.flush()

def section(title):
    p("\n" + "=" * 65)
    p(f"  {title}")
    p("=" * 65)

# ================================================================
#  HELPERS
# ================================================================

RATING_MAP = {"1": 0.0, "1.0": 0.0, "2": 0.25, "2.0": 0.25,
              "3": 0.5,  "3.0": 0.5,  "4": 0.75, "4.0": 0.75,
              "5": 1.0,  "5.0": 1.0}
MAX_DATE = datetime(2023, 1, 7)

def recency_weight(date_str):
    try:
        dt = datetime.strptime(str(date_str)[:10], "%Y-%m-%d")
        return math.exp(-max((MAX_DATE - dt).days, 0) / 365.0)
    except Exception:
        return 0.5

def clean_text(text):
    if not isinstance(text, str) or not text.strip():
        return ""
    text = text.strip().lower()
    text = re.sub(r"http\S+|www\.\S+", "", text)
    text = re.sub(r"\s+", " ", text)
    return text.strip()

def count_rows(path):
    with open(path, "r", encoding="utf-8") as f:
        return sum(1 for _ in f) - 1


# ================================================================
#  PHASE 1 — Data Preprocessing & RFM Engineering
# ================================================================

def run_phase1():
    section("PHASE 1: DATA PREPROCESSING & RFM ENGINEERING")
    t0 = time.time()

    REVIEWS_OUT = os.path.join(OUT_EDA, "reviews_cleaned.csv")
    RFM_OUT     = os.path.join(OUT_EDA, "rfm_table.csv")

    # Check if Phase 1 outputs already exist (from uploaded after_EDA dataset)
    if os.path.exists(EDA_REVIEWS) and os.path.exists(EDA_RFM):
        p(f"  [SKIP] Found existing cleaned files in {_IN_EDA}")
        p(f"         Copying to working directory...")
        import shutil
        shutil.copy2(EDA_REVIEWS, REVIEWS_OUT)
        shutil.copy2(EDA_RFM,     RFM_OUT)
        # Copy other EDA files if they exist
        for fname in ["products_cleaned.csv", "attributes_cleaned.csv",
                      "shops_cleaned.csv", "product_embeddings.csv",
                      "product_mapping.csv", "sarfm_table.csv",
                      "sarfm_vectors.csv", "sarfm_segmented.csv"]:
            src = os.path.join(_IN_EDA, "after_EDA", fname)
            if os.path.exists(src):
                shutil.copy2(src, os.path.join(OUT_EDA, fname))
                p(f"         Copied: {fname}")
        p(f"  Phase 1 skipped (pre-processed data used)  ({time.time()-t0:.1f}s)")
        return REVIEWS_OUT, RFM_OUT

    p("  Loading raw data...")
    products   = pd.read_csv(RAW_PRODUCTS)
    attributes = pd.read_csv(RAW_ATTRS)
    shops      = pd.read_csv(RAW_SHOPS)
    p(f"  products: {products.shape}, attributes: {attributes.shape}, shops: {shops.shape}")

    # Clean products
    products["processed_description"] = products["processed_description"].apply(clean_text)
    for col in ["ingredient", "feature", "skin_type"]:
        if col in attributes.columns:
            attributes[col] = attributes[col].apply(clean_text)
    attributes["ingredient"] = attributes.get("ingredient", pd.Series()).replace("", "unknown")
    attributes["skin_type"]  = attributes.get("skin_type",  pd.Series()).replace("", "all_skin")
    for col in ["capacity", "design", "brand", "expiry", "origin"]:
        if col in attributes.columns:
            attributes[col] = attributes[col].fillna("unknown")

    price_lookup = dict(zip(products["product_id"].astype(int),
                            products["price"].astype(float)))
    p(f"  Price lookup: {len(price_lookup):,} products")

    p("  Processing reviews (line-by-line)...")
    out_cols = ["user_id", "product_id", "rating", "product_name_x",
                "cmt_date", "shop_id", "variation_x", "product_quality",
                "processed_comment",
                "purchase_year", "purchase_month", "purchase_day_of_week", "purchase_hour"]

    user_data       = defaultdict(lambda: {"max_date": None, "count": 0, "total": 0.0})
    max_date_global = None
    total_rows      = 0

    with open(RAW_REVIEWS, "r", encoding="utf-8") as fin, \
         open(REVIEWS_OUT, "w", encoding="utf-8", newline="") as fout:
        reader = csv.DictReader(fin)
        writer = csv.DictWriter(fout, fieldnames=out_cols)
        writer.writeheader()
        for row in reader:
            total_rows += 1
            comment = clean_text(row.get("processed_comment", ""))
            try:
                dt = datetime.strptime(row["cmt_date"][:19], "%Y-%m-%d %H:%M:%S")
            except Exception:
                dt = MAX_DATE
            uid = row["user_id"]
            pid = int(row["product_id"])
            price = price_lookup.get(pid, 0.0)
            ud = user_data[uid]
            if ud["max_date"] is None or dt > ud["max_date"]:
                ud["max_date"] = dt
            ud["count"] += 1
            ud["total"] += price
            if max_date_global is None or dt > max_date_global:
                max_date_global = dt
            writer.writerow({
                "user_id": uid, "product_id": pid,
                "rating": row.get("rating", ""),
                "product_name_x": row.get("product_name_x", ""),
                "cmt_date": row["cmt_date"], "shop_id": row.get("shop_id", ""),
                "variation_x": row.get("variation_x", ""),
                "product_quality": row.get("product_quality", ""),
                "processed_comment": comment,
                "purchase_year": dt.year, "purchase_month": dt.month,
                "purchase_day_of_week": dt.weekday(), "purchase_hour": dt.hour,
            })
            if total_rows % 200_000 == 0:
                p(f"    {total_rows:,} rows processed...")

    p(f"  Reviews: {total_rows:,} rows | {len(user_data):,} users | max_date={max_date_global}")

    # RFM
    snap = max_date_global or MAX_DATE
    rfm_rows = [{"user_id": uid,
                 "recency":   (snap - ud["max_date"]).days,
                 "frequency": ud["count"],
                 "monetary":  ud["total"]}
                for uid, ud in user_data.items()]
    rfm = pd.DataFrame(rfm_rows)
    for col in ["recency", "frequency", "monetary"]:
        mn, mx = rfm[col].min(), rfm[col].max()
        rfm[f"{col}_norm"] = (rfm[col] - mn) / (mx - mn) if mx > mn else 0.0
    rfm["recency_norm"] = 1.0 - rfm["recency_norm"]   # invert: recent = high

    rfm.to_csv(RFM_OUT, index=False)
    products.to_csv(os.path.join(OUT_EDA, "products_cleaned.csv"), index=False)
    attributes.to_csv(os.path.join(OUT_EDA, "attributes_cleaned.csv"), index=False)
    shops.to_csv(os.path.join(OUT_EDA, "shops_cleaned.csv"), index=False)
    p(f"  Saved: rfm_table.csv ({len(rfm):,} users)")
    p(f"  Phase 1 done  ({(time.time()-t0)/60:.1f} min)")
    return REVIEWS_OUT, RFM_OUT


# ================================================================
#  PHASE 2 — Sentiment Analysis & SA-RFM
# ================================================================

def run_phase2(reviews_path, rfm_path):
    section("PHASE 2: SENTIMENT ANALYSIS & SA-RFM")
    t0 = time.time()

    SARFM_OUT   = os.path.join(OUT_EDA, "sarfm_table.csv")
    VECTORS_OUT = os.path.join(OUT_EDA, "sarfm_vectors.csv")
    CKPT        = os.path.join(OUT_LOGS, "phobert_checkpoint.csv")

    # ------ Try PhoBERT ------
    user_sentiments = None
    if USE_PHOBERT:
        try:
            import torch
            from transformers import AutoTokenizer, AutoModelForSequenceClassification
            try:
                from tqdm.auto import tqdm as tbar
                has_tqdm = True
            except ImportError:
                has_tqdm = False

            p("  Loading wonrax/phobert-base-vietnamese-sentiment ...")
            model_name = "wonrax/phobert-base-vietnamese-sentiment"
            tok   = AutoTokenizer.from_pretrained(model_name)
            model = AutoModelForSequenceClassification.from_pretrained(model_name)
            model.eval()
            device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
            model  = model.to(device)
            p(f"  Device: {device}")

            # Column check
            with open(reviews_path, "r", encoding="utf-8") as f:
                header = next(csv.reader(f))
            has_cmt = "processed_comment" in header
            if not has_cmt:
                p("  WARNING: processed_comment column missing -> rating fallback")

            total_rows = count_rows(reviews_path)
            limit = min(total_rows, MAX_PHOBERT_REVIEWS) if MAX_PHOBERT_REVIEWS else total_rows
            p(f"  Rows: {total_rows:,} | Processing: {limit:,}")

            # Resume checkpoint
            user_sentiments = defaultdict(lambda: {"ws": 0.0, "wt": 0.0, "n": 0})
            resume_from = 0
            if os.path.exists(CKPT):
                with open(CKPT, "r", encoding="utf-8") as cf:
                    for r in csv.DictReader(cf):
                        u = r["user_id"]
                        user_sentiments[u]["ws"] = float(r["ws"])
                        user_sentiments[u]["wt"] = float(r["wt"])
                        user_sentiments[u]["n"]  = int(r["n"])
                        resume_from = max(resume_from, int(r.get("last_row", 0)))
                p(f"  Resuming from row {resume_from:,}")

            btexts, bmeta = [], []

            def flush_batch(texts, metas):
                if not texts:
                    return
                inp = tok(texts, return_tensors="pt", truncation=True,
                          padding=True, max_length=256).to(device)
                with torch.no_grad():
                    probs = torch.softmax(model(**inp).logits, dim=1)
                for i, (uid, w) in enumerate(metas):
                    neg, neu, pos = probs[i].cpu().tolist()
                    s = neg*0.0 + neu*0.5 + pos*1.0
                    d = user_sentiments[uid]
                    d["ws"] += s*w; d["wt"] += w; d["n"] += 1

            def save_ckpt(last_row):
                with open(CKPT, "w", encoding="utf-8", newline="") as cf:
                    w = csv.writer(cf)
                    w.writerow(["user_id", "ws", "wt", "n", "last_row"])
                    for uid, d in user_sentiments.items():
                        w.writerow([uid, d["ws"], d["wt"], d["n"], last_row])
                p(f"  [ckpt saved @ row {last_row:,}]")

            t1 = time.time()
            total_read = processed = 0
            fh = open(reviews_path, "r", encoding="utf-8")
            try:
                reader   = csv.DictReader(fh)
                row_iter = tbar(reader, total=limit, desc="PhoBERT", unit="rev") if has_tqdm else reader
                for row in row_iter:
                    total_read += 1
                    if total_read <= resume_from:
                        continue
                    if MAX_PHOBERT_REVIEWS and processed >= MAX_PHOBERT_REVIEWS:
                        p(f"  Reached MAX={MAX_PHOBERT_REVIEWS:,}")
                        break
                    uid    = row["user_id"]
                    weight = recency_weight(row.get("cmt_date", ""))
                    cmt    = row.get("processed_comment", "").strip() if has_cmt else ""
                    if not cmt:
                        s = RATING_MAP.get(row.get("rating", "").strip(), 0.5)
                        d = user_sentiments[uid]
                        d["ws"] += s*weight; d["wt"] += weight; d["n"] += 1
                    else:
                        btexts.append(cmt[:512]); bmeta.append((uid, weight))
                        if len(btexts) >= PHOBERT_BATCH_SIZE:
                            flush_batch(btexts, bmeta)
                            btexts.clear(); bmeta.clear()
                    processed += 1
                    if not has_tqdm and processed % 10_000 == 0:
                        el = time.time()-t1
                        rate = processed/el if el>0 else 1
                        eta  = (limit-processed)/rate
                        p(f"  {processed:,}/{limit:,} | {rate:.0f}/s | ETA {eta/60:.1f}m")
                    if processed % CHECKPOINT_EVERY == 0:
                        flush_batch(btexts, bmeta); btexts.clear(); bmeta.clear()
                        save_ckpt(total_read)
            finally:
                fh.close()

            flush_batch(btexts, bmeta)
            elapsed = time.time()-t1
            p(f"  PhoBERT done: {processed:,} reviews in {elapsed/60:.1f} min")
            if os.path.exists(CKPT):
                os.remove(CKPT)

        except ImportError as e:
            p(f"  PhoBERT unavailable ({e}) -> rating fallback")
            user_sentiments = None
        except Exception as e:
            p(f"  PhoBERT error ({e}) -> rating fallback")
            p(traceback.format_exc())
            user_sentiments = None

    # ------ Rating-based fallback ------
    if user_sentiments is None:
        p("  Using MODE 1: rating-based sentiment")
        user_sentiments = defaultdict(lambda: {"ws": 0.0, "wt": 0.0, "n": 0})
        total = 0
        t1 = time.time()
        with open(reviews_path, "r", encoding="utf-8") as f:
            reader = csv.DictReader(f)
            for row in reader:
                total += 1
                uid = row["user_id"]
                s   = RATING_MAP.get(row.get("rating", "").strip(), 0.5)
                w   = recency_weight(row.get("cmt_date", ""))
                d   = user_sentiments[uid]
                d["ws"] += s*w; d["wt"] += w; d["n"] += 1
                if total % 200_000 == 0:
                    p(f"  {total:,} reviews  ({time.time()-t1:.1f}s)")
        p(f"  Rating sentiment done: {total:,} reviews | {len(user_sentiments):,} users")

    # ------ Build SA-RFM ------
    rfm = pd.read_csv(rfm_path)
    sent_rows = []
    for uid, d in user_sentiments.items():
        s = d["ws"]/d["wt"] if d["wt"] > 0 else 0.5
        sent_rows.append({"user_id": int(uid), "sentiment": s})
    sdf = pd.DataFrame(sent_rows)
    sarfm = rfm.merge(sdf, on="user_id", how="left")
    sarfm["sentiment"] = sarfm["sentiment"].fillna(0.5)
    smin, smax = sarfm["sentiment"].min(), sarfm["sentiment"].max()
    sarfm["sentiment_norm"] = (sarfm["sentiment"]-smin)/(smax-smin) if smax>smin else 0.5

    sarfm.to_csv(SARFM_OUT, index=False)
    vec_cols = [c for c in ["user_id","recency_norm","frequency_norm","monetary_norm","sentiment_norm"]
                if c in sarfm.columns]
    sarfm[vec_cols].to_csv(VECTORS_OUT, index=False)
    p(f"  Saved sarfm_table.csv ({len(sarfm):,} users)")
    p(f"  Saved sarfm_vectors.csv ({len(vec_cols)} cols)")
    p(f"  Phase 2 done  ({(time.time()-t0)/60:.1f} min)")
    return SARFM_OUT, VECTORS_OUT


# ================================================================
#  PHASE 3 — Customer Segmentation
# ================================================================

def run_phase3(vectors_path, table_path):
    section("PHASE 3: CUSTOMER SEGMENTATION (SA-RFM CLUSTERING)")
    t0 = time.time()

    try:
        from src.segmentation.segmentation import (
            load_data, evaluate_k_range, plot_evaluation,
            fit_models, name_segments, generate_segment_profiles,
            plot_centroids_radar, plot_segment_scatter
        )
    except ImportError as e:
        p(f"  ERROR importing segmentation module: {e}")
        raise

    # Monkey-patch load_data to use our paths
    import src.segmentation.segmentation as seg_mod
    _orig_load = seg_mod.load_data
    def _load_data_patched():
        vectors = pd.read_csv(vectors_path)
        table   = pd.read_csv(table_path)
        return vectors, table
    seg_mod.load_data = _load_data_patched

    vectors, table = seg_mod.load_data()
    p(f"  Vectors: {vectors.shape}, Table: {table.shape}")

    p("  Evaluating K in [2,7]...")
    os.makedirs(os.path.join(_WORK, "outputs/tables"), exist_ok=True)
    os.makedirs(os.path.join(_WORK, "outputs/figures"), exist_ok=True)

    eval_df = evaluate_k_range(vectors, max_k=7, sample_size=15000, random_state=42)
    p(eval_df.to_string(index=False))
    eval_df.to_csv(os.path.join(OUT_TABS, "clustering_evaluation_metrics.csv"), index=False)

    plot_evaluation(eval_df)

    optimal_k = 5
    p(f"  Optimal K = {optimal_k}")

    km_labels, gmm_labels, birch_labels, model_cmp = fit_models(vectors, k=optimal_k, random_state=42)
    p(model_cmp.to_string())
    model_cmp.to_csv(os.path.join(OUT_TABS, "model_comparison.csv"))

    feature_cols = ["recency_norm","frequency_norm","monetary_norm","sentiment_norm"]
    vc = vectors.copy(); vc["cluster"] = km_labels
    centroids = vc.groupby("cluster")[feature_cols].mean()
    segment_names = name_segments(centroids)
    p("  Segment names: " + str(segment_names))

    full_table, profiles_raw = generate_segment_profiles(table, km_labels, segment_names)
    profiles_raw.to_csv(os.path.join(OUT_TABS, "segment_profiles.csv"), index=False)
    full_table.to_csv(os.path.join(OUT_EDA, "sarfm_segmented.csv"), index=False)
    p(f"  Saved sarfm_segmented.csv ({len(full_table):,} rows)")

    seg_mod.load_data = _orig_load
    p(f"  Phase 3 done  ({(time.time()-t0)/60:.1f} min)")
    return full_table


# ================================================================
#  PHASE 4 — Multimodal Product Embedding
# ================================================================

def run_phase4():
    section("PHASE 4: MULTIMODAL PRODUCT EMBEDDING")
    t0 = time.time()

    EMB_OUT = os.path.join(OUT_EDA, "product_embeddings.csv")
    MAP_OUT = os.path.join(OUT_EDA, "product_mapping.csv")

    # Check if already exists from uploaded after_EDA
    if os.path.exists(EMB_OUT) and os.path.exists(MAP_OUT):
        p(f"  [SKIP] product_embeddings.csv already exists")
        return EMB_OUT

    try:
        from src.embedding.product_embedding import (
            load_product_data, build_concatenated_text,
            extract_text_embeddings_fallback, try_extract_text_embeddings_dl,
            extract_image_embeddings_fallback, try_extract_image_embeddings_dl,
            encode_metadata, fuse_features
        )
    except ImportError as e:
        p(f"  ERROR importing embedding module: {e}")
        raise

    df_product = load_product_data()
    texts      = build_concatenated_text(df_product)
    p(f"  Products: {df_product.shape[0]:,}")

    text_emb = try_extract_text_embeddings_dl(texts, n_components=128)
    if text_emb is None:
        text_emb, _ = extract_text_embeddings_fallback(texts, n_components=128)
        p("  Text: TF-IDF + PCA fallback")
    else:
        p("  Text: SentenceTransformers")

    img_emb = try_extract_image_embeddings_dl(df_product["image_path"].tolist(), n_components=128)
    if img_emb is None:
        img_emb = extract_image_embeddings_fallback(df_product["image_path"].tolist(), n_components=128)
        p("  Image: hash-based fallback")

    meta = encode_metadata(df_product)
    fused, _ = fuse_features(text_emb, img_emb, meta, out_dim=128)
    p(f"  Fused embedding shape: {fused.shape}")

    df_emb = pd.DataFrame(fused, columns=[f"dim_{i}" for i in range(128)])
    df_emb.insert(0, "product_id", df_product["product_id"])
    df_emb.to_csv(EMB_OUT, index=False)

    mapping_cols = [c for c in ["product_id","product_name","brand","type","price","avg_star","num_sold_time"]
                    if c in df_product.columns]
    df_product[mapping_cols].to_csv(MAP_OUT, index=False)
    p(f"  Saved product_embeddings.csv")
    p(f"  Phase 4 done  ({(time.time()-t0)/60:.1f} min)")
    return EMB_OUT


# ================================================================
#  PHASE 5 — Market Basket Analysis
# ================================================================

def run_phase5():
    section("PHASE 5: MARKET BASKET ANALYSIS")
    t0 = time.time()

    try:
        from src.mba.mba import (
            load_mba_data, build_baskets, mine_association_rules,
            run_segment_aware_mba
        )
    except ImportError as e:
        p(f"  ERROR importing MBA module: {e}")
        raise

    df_reviews, df_segments, embeddings, prod_name, prod_brand, prod_price = load_mba_data()
    p(f"  Reviews: {df_reviews.shape[0]:,} | Segments: {df_segments.shape[0]:,}")

    global_baskets = build_baskets(df_reviews)
    p(f"  Baskets (>=2 products): {len(global_baskets):,}")

    df_rules = mine_association_rules(
        global_baskets, min_support=0.0005, min_confidence=0.05, min_lift=1.2,
        embeddings=embeddings, prod_name_map=prod_name,
        prod_brand_map=prod_brand, prod_price_map=prod_price
    )
    p(f"  Global rules mined: {len(df_rules)}")
    if not df_rules.empty:
        df_rules.to_csv(os.path.join(OUT_TABS, "global_association_rules.csv"), index=False)

    seg_rules = run_segment_aware_mba(
        df_reviews, df_segments,
        min_support=0.0005, min_confidence=0.05, min_lift=1.2,
        embeddings=embeddings, prod_name_map=prod_name,
        prod_brand_map=prod_brand, prod_price_map=prod_price
    )
    for seg, df_seg in seg_rules.items():
        if not df_seg.empty:
            cn = re.sub(r"\s+", "_", re.sub(r"[^\w\s-]", "", seg).strip()).lower()
            df_seg.to_csv(os.path.join(OUT_TABS, f"segment_rules_{cn}.csv"), index=False)
            p(f"  Segment {seg}: {len(df_seg)} rules saved")

    p(f"  Phase 5 done  ({(time.time()-t0)/60:.1f} min)")


# ================================================================
#  PHASE 6 — Recommendation Engine Evaluation
# ================================================================

def run_phase6():
    section("PHASE 6: HYBRID RECOMMENDATION ENGINE EVALUATION")
    t0 = time.time()

    try:
        from src.recommendation.recommender import SegmentAwareHybridRecommender
    except ImportError as e:
        p(f"  ERROR importing recommender module: {e}")
        raise

    rec = SegmentAwareHybridRecommender()
    p(f"  Users: {len(rec.user_purchases):,} | Products: {len(rec.prod_name):,}")
    p(f"  Segments: {list(rec.segment_rules.keys())}")

    df_seg  = rec.df_segmented
    segments = df_seg["segment_name"].unique()
    samples  = []
    all_recs = []

    for seg in segments:
        seg_users = df_seg[df_seg["segment_name"]==seg]["user_id"].tolist()
        active = [u for u in seg_users if u in rec.user_purchases and len(rec.user_purchases[u])>=2]
        uid = active[0] if active else seg_users[0]
        samples.append((seg, uid))

    for seg, uid in samples:
        recs = rec.recommend(user_id=uid, top_k=5)
        p(f"\n  [{seg}] User {uid}:")
        for idx, row in recs.iterrows():
            p(f"    {idx+1}. {row['product_name']} (score={row['score']:.4f})")
            all_recs.append({"segment": seg, "user_id": uid, "rank": idx+1,
                             "product_id": row["product_id"],
                             "product_name": row["product_name"],
                             "score": row["score"],
                             "explanation": row["explanation"]})

    pd.DataFrame(all_recs).to_csv(os.path.join(OUT_TABS, "recommender_samples.csv"), index=False)

    # Batch evaluation
    np.random.seed(42)
    eval_uids = np.random.choice(list(rec.user_purchases.keys()), size=200, replace=False)
    all_pids  = set(); rule_hits = 0; sims_list = []
    for uid in eval_uids:
        r = rec.recommend(user_id=int(uid), top_k=5)
        if r.empty: continue
        pids = r["product_id"].tolist()
        all_pids.update(pids)
        rule_hits += r["explanation"].str.contains("Bought by other users").sum()
        if len(pids) >= 2:
            vs = [rec.embeddings.get(p) for p in pids]
            s  = [rec._cosine_similarity(vs[i], vs[j])
                  for i in range(len(vs)) for j in range(i+1, len(vs))
                  if vs[i] is not None and vs[j] is not None]
            if s: sims_list.append(np.mean(s))

    cov = len(all_pids)/len(rec.prod_name)*100
    div = 1.0 - (np.mean(sims_list) if sims_list else 0.0)
    p(f"\n  Coverage: {cov:.2f}% | Diversity: {div:.4f}")
    pd.DataFrame([{"catalog_coverage_pct": cov, "list_diversity": div,
                   "mba_hit_rate_pct": rule_hits/(200*5)*100}]
                ).to_csv(os.path.join(OUT_TABS, "recommender_evaluation_metrics.csv"), index=False)
    p(f"  Phase 6 done  ({(time.time()-t0)/60:.1f} min)")


# ================================================================
#  PHASE 7 — Dashboard & Visualization
# ================================================================

def run_phase7():
    section("PHASE 7: DASHBOARD GENERATION & VISUALIZATION")
    t0 = time.time()
    try:
        from src.visualization.dashboard import DashboardGenerator
        gen = DashboardGenerator()
        gen.generate_all()
        figs = [f for f in os.listdir(OUT_FIGS) if f.endswith(".png")]
        p(f"  Generated {len(figs)} figure(s):")
        for f in figs:
            sz = os.path.getsize(os.path.join(OUT_FIGS, f))
            p(f"    - {f}  ({sz/1024:.1f} KB)")
    except Exception as e:
        p(f"  Phase 7 error (non-fatal): {e}")
        p(traceback.format_exc())
    p(f"  Phase 7 done  ({(time.time()-t0)/60:.1f} min)")


# ================================================================
#  MAIN PIPELINE
# ================================================================

if __name__ == "__main__":
    T_PIPELINE = time.time()

    p("=" * 65)
    p("  DSP FULL PIPELINE — Kaggle Edition")
    p(f"  Started: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    p(f"  SLUG_RAW = {SLUG_RAW}")
    p(f"  SLUG_EDA = {SLUG_EDA}")
    p(f"  OUT_DIR  = {_WORK}")
    p("=" * 65)

    try:
        reviews_path, rfm_path = run_phase1()
        sarfm_path, vectors_path = run_phase2(reviews_path, rfm_path)
        run_phase3(vectors_path, sarfm_path)
        run_phase4()
        run_phase5()
        run_phase6()
        run_phase7()
    except Exception as e:
        p(f"\n!!! PIPELINE ERROR: {e}")
        p(traceback.format_exc())
    finally:
        total = time.time() - T_PIPELINE
        p("\n" + "=" * 65)
        p(f"  PIPELINE FINISHED  (total: {total/3600:.2f} h)")
        p(f"  Output directory: {_WORK}")
        p("=" * 65)
        _log_file.close()

        # List all outputs
        for root, dirs, files in os.walk(_WORK):
            dirs[:] = [d for d in dirs if d != "__pycache__"]
            for fname in files:
                fp = os.path.join(root, fname)
                sz = os.path.getsize(fp)
                rel = os.path.relpath(fp, _WORK)
                print(f"  {rel:60s}  {sz/1e6:7.2f} MB")
