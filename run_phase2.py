# -*- coding: utf-8 -*-
"""
Phase 2: Sentiment Analysis & SA-RFM Computation
=================================================
Two modes:
  - MODE 1 (default): Rating-based sentiment (fast, no GPU needed)
  - MODE 2 (enhanced): PhoBERT-based sentiment (requires GPU, transformers)

The SA-RFM module adds the Sentiment (S) dimension to the RFM model,
creating the novel 4-dimensional RFMS customer representation.

FIX LOG (2026-06-30):
  - Added USE_PHOBERT flag at top to easily force MODE 1
  - Added MAX_PHOBERT_REVIEWS cap so PhoBERT never runs forever
  - Added tqdm progress bar with ETA estimate
  - Added column validation before processing (catches missing columns early)
  - Added checkpoint saving every CHECKPOINT_EVERY reviews (resume on crash)
  - Increased batch_size to 64 for better GPU throughput
  - Added print() alongside log so output is visible in Colab cells
  - Added total elapsed time report at completion
  - Auto-detects Kaggle / Colab / Local environment and sets paths accordingly
"""

import os
import csv
import math
import time
import traceback
from collections import defaultdict
from datetime import datetime

# ============================================================
# CONFIGURATION
# ============================================================

# Set False to skip PhoBERT entirely and use fast rating-based MODE 1
USE_PHOBERT = True

# PhoBERT batch size
#   T4  (Colab / Kaggle free) -> 64
#   P100 (Kaggle)             -> 96
#   A100 (Colab Pro+)         -> 128
PHOBERT_BATCH_SIZE = 64

# Save checkpoint every N reviews (resume on crash/timeout)
CHECKPOINT_EVERY = 50_000

# ---- Kaggle-specific: name of your uploaded dataset ----
# Go to kaggle.com -> Datasets -> New Dataset, upload the after_EDA folder,
# then paste the slug here (shown in the dataset URL).
KAGGLE_DATASET_SLUG = "your-username/tiki-after-eda"   # <-- CHANGE THIS

# End of data collection period (for recency weighting)
MAX_DATE = datetime(2023, 1, 7)

# ============================================================
# ENVIRONMENT DETECTION  (Kaggle / Colab / Local)
# ============================================================

_ON_KAGGLE = os.path.exists("/kaggle/working")
_ON_COLAB  = not _ON_KAGGLE and os.path.exists("/content")

if _ON_KAGGLE:
    # --------------- KAGGLE ---------------
    # Input: /kaggle/input/<dataset-slug>/
    # Output: /kaggle/working/  (persisted, downloadable)
    _INPUT   = f"/kaggle/input/{KAGGLE_DATASET_SLUG}"
    _OUTPUT  = "/kaggle/working"
    # Kaggle gives 9h sessions on GPU -> safe to process everything
    MAX_PHOBERT_REVIEWS = None   # unlimited
    print(f"[ENV] Kaggle detected | input={_INPUT} | output={_OUTPUT}")

elif _ON_COLAB:
    # --------------- COLAB ---------------
    # Mount Drive first:  from google.colab import drive; drive.mount('/content/drive')
    # Then set _DRIVE to your project folder.
    _DRIVE   = "/content/drive/MyDrive/DSP"   # <-- change if your folder differs
    _INPUT   = _DRIVE
    _OUTPUT  = _DRIVE
    # Colab free tier disconnects after ~90 min of idle -> cap at 500k
    MAX_PHOBERT_REVIEWS = 500_000
    print(f"[ENV] Colab detected  | drive={_DRIVE}")

else:
    # --------------- LOCAL ---------------
    _INPUT   = "."
    _OUTPUT  = "."
    MAX_PHOBERT_REVIEWS = 500_000
    print("[ENV] Local machine")

# Build final paths
REVIEWS_PATH    = os.path.join(_INPUT,  "dataset/after_EDA/reviews_cleaned.csv")
RFM_PATH        = os.path.join(_INPUT,  "dataset/after_EDA/rfm_table.csv")
OUT_DIR         = os.path.join(_OUTPUT, "dataset/after_EDA")
OUT_SARFM       = os.path.join(OUT_DIR, "sarfm_table.csv")
OUT_VECTORS     = os.path.join(OUT_DIR, "sarfm_vectors.csv")
CHECKPOINT_PATH = os.path.join(_OUTPUT, "outputs/phobert_checkpoint.csv")
LOG_FILE        = os.path.join(_OUTPUT, "outputs/sentiment_log.txt")

os.makedirs(os.path.join(_OUTPUT, "outputs"),      exist_ok=True)
os.makedirs(OUT_DIR,                               exist_ok=True)

print(f"[ENV] REVIEWS_PATH    = {REVIEWS_PATH}")
print(f"[ENV] RFM_PATH        = {RFM_PATH}")
print(f"[ENV] OUT_SARFM       = {OUT_SARFM}")
print(f"[ENV] MAX_PHOBERT_REVIEWS = {MAX_PHOBERT_REVIEWS}")

try:
    log = open(LOG_FILE, "w", encoding="utf-8")
except PermissionError:
    try:
        if os.path.exists(LOG_FILE):
            os.remove(LOG_FILE)
        log = open(LOG_FILE, "w", encoding="utf-8")
    except Exception:
        LOG_FILE = "sentiment_log.txt"
        log = open(LOG_FILE, "w", encoding="utf-8")


def p(msg):
    print(msg)              # visible in Colab cell output
    log.write(str(msg) + "\n")
    log.flush()


# ============================================================
# HELPERS
# ============================================================

RATING_MAP = {
    "1": 0.0,  "1.0": 0.0,
    "2": 0.25, "2.0": 0.25,
    "3": 0.5,  "3.0": 0.5,
    "4": 0.75, "4.0": 0.75,
    "5": 1.0,  "5.0": 1.0,
}


def recency_weight(date_str):
    """Exponential decay weight: more recent reviews count more."""
    try:
        dt = datetime.strptime(str(date_str)[:10], "%Y-%m-%d")
        days_ago = max((MAX_DATE - dt).days, 0)
        return math.exp(-days_ago / 365.0)
    except (ValueError, KeyError, TypeError):
        return 0.5


def validate_columns(path, required_cols):
    """Abort early with a clear error if required columns are missing."""
    with open(path, "r", encoding="utf-8") as f:
        header = next(csv.reader(f))
    missing = [c for c in required_cols if c not in header]
    if missing:
        raise ValueError(f"Missing columns in {path}: {missing}\nFound: {header}")
    p(f"  column check OK: {header}")
    return header


def count_rows(path):
    """Fast line count (subtract 1 for header row)."""
    with open(path, "r", encoding="utf-8") as f:
        return sum(1 for _ in f) - 1


# ============================================================
# MODE 1: Rating-based sentiment
# ============================================================

def compute_rating_based_sentiment():
    """
    MODE 1: Derive sentiment from the rating column.

    Mapping: 1->0.0, 2->0.25, 3->0.5, 4->0.75, 5->1.0

    Uses exponential recency weighting so recent reviews matter more.
    Fast and requires no GPU.
    """
    p("\n--- MODE 1: Rating-Based Sentiment ---")

    validate_columns(REVIEWS_PATH, ["user_id", "rating", "cmt_date"])

    user_sentiments = defaultdict(lambda: {"weighted_sum": 0.0, "weight_total": 0.0, "count": 0})

    t0 = time.time()
    total = 0
    with open(REVIEWS_PATH, "r", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        for row in reader:
            total += 1
            uid       = row["user_id"]
            rating    = row.get("rating", "").strip()
            sentiment = RATING_MAP.get(rating, 0.5)
            weight    = recency_weight(row.get("cmt_date", ""))

            ud = user_sentiments[uid]
            ud["weighted_sum"]  += sentiment * weight
            ud["weight_total"]  += weight
            ud["count"]         += 1

            if total % 100_000 == 0:
                elapsed = time.time() - t0
                p(f"  {total:,} reviews processed  ({elapsed:.1f}s elapsed)")

    p(f"  Total: {total:,} reviews  ({time.time()-t0:.1f}s)")
    p(f"  Unique users: {len(user_sentiments):,}")
    return user_sentiments


# ============================================================
# MODE 2: PhoBERT-based sentiment
# ============================================================

def try_phobert_sentiment():
    """
    MODE 2: PhoBERT sentiment analysis (Vietnamese).

    Requires: pip install transformers torch tqdm
    Falls back to MODE 1 if USE_PHOBERT=False or dependencies missing.

    Key fixes vs original:
      - MAX_PHOBERT_REVIEWS prevents infinite processing
      - tqdm shows progress + ETA
      - Checkpoint every CHECKPOINT_EVERY rows to survive Colab disconnects
      - Column validation prevents silent errors
    """
    if not USE_PHOBERT:
        p("  USE_PHOBERT=False -> skipping PhoBERT, using MODE 1")
        return None

    try:
        import torch
        from transformers import AutoTokenizer, AutoModelForSequenceClassification

        try:
            from tqdm.auto import tqdm as tqdm_bar
            has_tqdm = True
        except ImportError:
            has_tqdm = False
            p("  (tip: pip install tqdm for nicer progress bars)")

        p("\n--- MODE 2: PhoBERT Sentiment Analysis ---")
        p("  Loading: wonrax/phobert-base-vietnamese-sentiment ...")

        model_name = "wonrax/phobert-base-vietnamese-sentiment"
        tokenizer = AutoTokenizer.from_pretrained(model_name)
        model = AutoModelForSequenceClassification.from_pretrained(model_name)
        model.eval()

        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        model = model.to(device)
        p(f"  Device: {device}")

        # Validate columns; check if processed_comment exists
        header = validate_columns(REVIEWS_PATH, ["user_id", "rating", "cmt_date"])
        has_comment_col = "processed_comment" in header
        if not has_comment_col:
            p("  WARNING: 'processed_comment' column not found -> using rating fallback for all rows")

        # Count rows for progress display
        p("  Counting rows in file...")
        total_rows = count_rows(REVIEWS_PATH)
        limit = min(total_rows, MAX_PHOBERT_REVIEWS) if MAX_PHOBERT_REVIEWS else total_rows
        p(f"  File total rows : {total_rows:,}")
        p(f"  Will process    : {limit:,}  (MAX_PHOBERT_REVIEWS={MAX_PHOBERT_REVIEWS})")
        if MAX_PHOBERT_REVIEWS and total_rows > MAX_PHOBERT_REVIEWS:
            p(f"  NOTE: {total_rows - limit:,} rows will be skipped (set MAX_PHOBERT_REVIEWS=None to process all)")

        # Resume from checkpoint if available
        user_sentiments = defaultdict(lambda: {"weighted_sum": 0.0, "weight_total": 0.0, "count": 0})
        resume_from = 0
        if os.path.exists(CHECKPOINT_PATH):
            p(f"  Checkpoint found: {CHECKPOINT_PATH} -- resuming...")
            with open(CHECKPOINT_PATH, "r", encoding="utf-8") as cf:
                for ckr in csv.DictReader(cf):
                    uid = ckr["user_id"]
                    user_sentiments[uid]["weighted_sum"]  = float(ckr["weighted_sum"])
                    user_sentiments[uid]["weight_total"]  = float(ckr["weight_total"])
                    user_sentiments[uid]["count"]         = int(ckr["count"])
                    resume_from = max(resume_from, int(ckr.get("last_row", 0)))
            p(f"  Resuming from row {resume_from:,}")

        batch_texts: list = []
        batch_meta:  list = []

        def process_batch(texts, metas):
            if not texts:
                return
            inputs = tokenizer(
                texts, return_tensors="pt",
                truncation=True, padding=True, max_length=256,
            ).to(device)
            with torch.no_grad():
                probs = torch.softmax(model(**inputs).logits, dim=1)
            # Labels order: [NEG=0, NEU=1, POS=2]
            for i, (uid, weight) in enumerate(metas):
                neg, neu, pos = probs[i].cpu().tolist()
                s = neg * 0.0 + neu * 0.5 + pos * 1.0
                ud = user_sentiments[uid]
                ud["weighted_sum"]  += s * weight
                ud["weight_total"]  += weight
                ud["count"]         += 1

        def save_checkpoint(last_row):
            with open(CHECKPOINT_PATH, "w", encoding="utf-8", newline="") as cf:
                w = csv.writer(cf)
                w.writerow(["user_id", "weighted_sum", "weight_total", "count", "last_row"])
                for uid, ud in user_sentiments.items():
                    w.writerow([uid, ud["weighted_sum"], ud["weight_total"], ud["count"], last_row])
            p(f"  [checkpoint saved at row {last_row:,}]")

        t0 = time.time()
        total_read = 0
        processed  = 0

        fh = open(REVIEWS_PATH, "r", encoding="utf-8")
        try:
            reader   = csv.DictReader(fh)
            row_iter = tqdm_bar(reader, total=limit, desc="PhoBERT", unit="rev") if has_tqdm else reader

            for row in row_iter:
                total_read += 1

                # Skip already-checkpointed rows
                if total_read <= resume_from:
                    continue

                # Hard stop at limit
                if MAX_PHOBERT_REVIEWS and processed >= MAX_PHOBERT_REVIEWS:
                    p(f"  Reached MAX_PHOBERT_REVIEWS={MAX_PHOBERT_REVIEWS:,} -- stopping early")
                    break

                uid     = row["user_id"]
                weight  = recency_weight(row.get("cmt_date", ""))
                comment = row.get("processed_comment", "").strip() if has_comment_col else ""

                if not comment:
                    # Fallback: use star rating
                    rating = row.get("rating", "").strip()
                    s = RATING_MAP.get(rating, 0.5)
                    ud = user_sentiments[uid]
                    ud["weighted_sum"]  += s * weight
                    ud["weight_total"]  += weight
                    ud["count"]         += 1
                else:
                    batch_texts.append(comment[:512])
                    batch_meta.append((uid, weight))
                    if len(batch_texts) >= PHOBERT_BATCH_SIZE:
                        process_batch(batch_texts, batch_meta)
                        batch_texts.clear()
                        batch_meta.clear()

                processed += 1

                # Manual progress if no tqdm
                if not has_tqdm and processed % 10_000 == 0:
                    elapsed = time.time() - t0
                    rate = processed / elapsed if elapsed > 0 else 1
                    eta  = (limit - processed) / rate
                    p(f"  {processed:,}/{limit:,} | {rate:.0f} rev/s | ETA {eta/60:.1f} min")

                # Periodic checkpoint
                if processed % CHECKPOINT_EVERY == 0:
                    process_batch(batch_texts, batch_meta)
                    batch_texts.clear()
                    batch_meta.clear()
                    save_checkpoint(total_read)

        finally:
            fh.close()

        # Flush remaining batch
        process_batch(batch_texts, batch_meta)

        elapsed = time.time() - t0
        p(f"\n  Done: {processed:,} reviews in {elapsed/60:.1f} min")
        p(f"  Unique users: {len(user_sentiments):,}")

        # Remove checkpoint after full success
        if os.path.exists(CHECKPOINT_PATH):
            os.remove(CHECKPOINT_PATH)
            p("  Checkpoint removed (run complete)")

        return user_sentiments

    except ImportError as e:
        p(f"  PhoBERT unavailable ({e}) -> falling back to MODE 1")
        return None
    except Exception as e:
        p(f"  PhoBERT error: {e} -> falling back to MODE 1")
        p(traceback.format_exc())
        return None


# ============================================================
# SA-RFM Builder
# ============================================================

def build_sarfm(user_sentiments):
    """
    Merge RFM table with per-user sentiment scores to create SA-RFM.

    Outputs:
      sarfm_table.csv   -- full details (recency, frequency, monetary, sentiment + norms)
      sarfm_vectors.csv -- 4D normalized vectors ready for clustering
    """
    import pandas as pd

    p("\n--- Building SA-RFM Table ---")

    rfm = pd.read_csv(RFM_PATH)
    p(f"  RFM loaded: {len(rfm):,} users")

    # Convert dict -> DataFrame
    sentiment_rows = []
    for uid, ud in user_sentiments.items():
        s = ud["weighted_sum"] / ud["weight_total"] if ud["weight_total"] > 0 else 0.5
        sentiment_rows.append({"user_id": int(uid), "sentiment": s})

    sentiment_df = pd.DataFrame(sentiment_rows)
    p(f"  Sentiment computed: {len(sentiment_df):,} users")

    # Merge (left join keeps all RFM users)
    sarfm = rfm.merge(sentiment_df, on="user_id", how="left")
    sarfm["sentiment"] = sarfm["sentiment"].fillna(0.5)  # neutral for users with no reviews

    # Min-Max normalize to [0, 1]
    s_min, s_max = sarfm["sentiment"].min(), sarfm["sentiment"].max()
    sarfm["sentiment_norm"] = (
        (sarfm["sentiment"] - s_min) / (s_max - s_min) if s_max > s_min else 0.5
    )

    p(f"\n  SA-RFM: {len(sarfm):,} users x {sarfm.shape[1]} columns")
    p("\n  Sentiment statistics:")
    for name, val in [
        ("mean", sarfm["sentiment"].mean()),
        ("std",  sarfm["sentiment"].std()),
        ("min",  sarfm["sentiment"].min()),
        ("25%",  sarfm["sentiment"].quantile(0.25)),
        ("50%",  sarfm["sentiment"].quantile(0.50)),
        ("75%",  sarfm["sentiment"].quantile(0.75)),
        ("max",  sarfm["sentiment"].max()),
    ]:
        p(f"    {name:4s} = {val:.4f}")

    p("\n  Sentiment distribution:")
    for label, mask in [
        ("Very Negative (0.0-0.2)", sarfm["sentiment"] <= 0.2),
        ("Negative      (0.2-0.4)", (sarfm["sentiment"] > 0.2) & (sarfm["sentiment"] <= 0.4)),
        ("Neutral       (0.4-0.6)", (sarfm["sentiment"] > 0.4) & (sarfm["sentiment"] <= 0.6)),
        ("Positive      (0.6-0.8)", (sarfm["sentiment"] > 0.6) & (sarfm["sentiment"] <= 0.8)),
        ("Very Positive (0.8-1.0)", sarfm["sentiment"] > 0.8),
    ]:
        p(f"    {label}: {mask.sum():,}")

    # Save full table
    sarfm.to_csv(OUT_SARFM, index=False)
    p(f"\n  Saved: {OUT_SARFM}  ({os.path.getsize(OUT_SARFM)/1e6:.1f} MB)")

    # Save 4D vectors for clustering (only keep columns that exist)
    vec_cols = [c for c in ["user_id", "recency_norm", "frequency_norm", "monetary_norm", "sentiment_norm"]
                if c in sarfm.columns]
    sarfm[vec_cols].to_csv(OUT_VECTORS, index=False)
    p(f"  Saved: {OUT_VECTORS}  ({os.path.getsize(OUT_VECTORS)/1e6:.1f} MB)")

    return sarfm


# ============================================================
# MAIN
# ============================================================

try:
    p("=" * 60)
    p("PHASE 2: SENTIMENT ANALYSIS & SA-RFM")
    p("=" * 60)
    p(f"  USE_PHOBERT         = {USE_PHOBERT}")
    p(f"  MAX_PHOBERT_REVIEWS = {MAX_PHOBERT_REVIEWS}")
    p(f"  PHOBERT_BATCH_SIZE  = {PHOBERT_BATCH_SIZE}")
    p(f"  CHECKPOINT_EVERY    = {CHECKPOINT_EVERY}")

    t_start = time.time()

    user_sentiments = try_phobert_sentiment()

    if user_sentiments is None:
        p("\n  -> Using MODE 1: rating-based sentiment (fast)")
        user_sentiments = compute_rating_based_sentiment()
    else:
        p("\n  -> Using MODE 2: PhoBERT sentiment")

    sarfm = build_sarfm(user_sentiments)

    total_time = time.time() - t_start
    p("\n" + "=" * 60)
    p(f"PHASE 2 COMPLETE  (total time: {total_time/60:.1f} min)")
    p("=" * 60)

except Exception as e:
    p(f"\n!!! ERROR: {e}")
    p(traceback.format_exc())

finally:
    log.close()
