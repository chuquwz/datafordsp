# -*- coding: utf-8 -*-
"""
Interactive Testing Tool for Multimodal Data Mining Recommendation System
========================================================================
Designed for showcasing the input/output of the customer segmentation 
and recommendation engine to teachers/evaluators.
"""

import os
import sys
import pandas as pd
import numpy as np

# Ensure root path is in sys.path
sys.path.insert(0, '.')

try:
    from src.recommendation.recommender import SegmentAwareHybridRecommender
except ImportError:
    print("Error: Could not import SegmentAwareHybridRecommender. Please run this script from the workspace root.")
    sys.exit(1)

def safe_print(text=""):
    """Print text safely, replacing characters that cannot be encoded in the current stdout encoding."""
    try:
        print(text)
    except UnicodeEncodeError:
        try:
            enc = sys.stdout.encoding or 'utf-8'
            print(str(text).encode(enc, errors='replace').decode(enc))
        except Exception:
            # Absolute fallback
            print(str(text).encode('ascii', errors='replace').decode('ascii'))

def format_vnd(price):
    try:
        return f"{int(price):,} VND"
    except (ValueError, TypeError):
        return f"{price} VND"

def clear_screen():
    os.system('cls' if os.name == 'nt' else 'clear')

class PredictDemoApp:
    def __init__(self):
        safe_print("Initializing Recommendation Engine... (Loading datasets & embeddings)")
        self.recommender = SegmentAwareHybridRecommender()
        self.df_segmented = self.recommender.df_segmented
        
        # Load raw attributes for product similarity description
        self.attributes_cleaned_path = 'dataset/after_EDA/attributes_cleaned.csv'
        if os.path.exists(self.attributes_cleaned_path):
            self.df_attr = pd.read_csv(self.attributes_cleaned_path)
            self.prod_desc = dict(zip(self.df_attr['product_id'], self.df_attr['feature'].fillna('No description available')))
        else:
            self.prod_desc = {}

    def display_customer_profile(self, user_id):
        """Display customer profile stats and purchase history."""
        # Find customer stats in segmented table
        user_row = self.df_segmented[self.df_segmented['user_id'] == user_id]
        
        safe_print("\n" + "=" * 65)
        safe_print(f" CUSTOMER PROFILE (User ID: {user_id}) ".center(65, "="))
        safe_print("=" * 65)
        
        if not user_row.empty:
            row = user_row.iloc[0]
            safe_print(f" * Segment Label:      {row['segment_name']}")
            safe_print(f" * Recency (R):        {int(row['recency'])} days (Time since last purchase)")
            safe_print(f" * Frequency (F):      {int(row['frequency'])} purchases")
            safe_print(f" * Monetary (M):       {format_vnd(row['monetary'])}")
            safe_print(f" * Sentiment Score(S): {row['sentiment']:.4f} (Derived from review text via PhoBERT)")
            safe_print(f"   - Norm Vector:      [R_norm={row['recency_norm']:.4f}, F_norm={row['frequency_norm']:.4f}, M_norm={row['monetary_norm']:.4f}, S_norm={row['sentiment_norm']:.4f}]")
        else:
            safe_print(" * Customer Status:    [New Customer / Cold-Start Case]")
            safe_print(" * Segment Label:      General/Hibernating (Cold Start)")
            
        safe_print("-" * 65)
        
        history = list(self.recommender.user_purchases.get(user_id, set()))
        if history:
            safe_print(f" Purchase History ({len(history)} items):")
            for idx, pid in enumerate(history):
                name = self.recommender.prod_name.get(pid, 'Unknown Product')
                brand = self.recommender.prod_brand.get(pid, 'Unknown')
                price = self.recommender.prod_price.get(pid, 0)
                safe_print(f"   {idx+1}. [ID: {pid}] [{brand.upper()}] {name} ({format_vnd(price)})")
        else:
            safe_print(" Purchase History:     Empty (No previous transaction records)")
        safe_print("=" * 65 + "\n")

    def show_recommendations(self, user_id):
        """Generate and display recommendations with explanations."""
        self.display_customer_profile(user_id)
        
        safe_print(" FUSING HYBRID SCORES & GENERATING PERSONALIZED SUGGESTIONS... ".center(65, "~"))
        
        recs = self.recommender.recommend(user_id=user_id, top_k=5)
        
        safe_print("\n" + "=" * 65)
        safe_print(f" RECOMMENDATION RESULTS FOR USER {user_id} ".center(65, "="))
        safe_print("=" * 65)
        
        if recs.empty:
            safe_print(" No recommendations could be generated.")
        else:
            for idx, row in recs.iterrows():
                safe_print(f" {idx+1}. [{row['brand'].upper()}] {row['product_name']}")
                safe_print(f"    - Product ID:  {row['product_id']}")
                safe_print(f"    - Price:       {format_vnd(row['price'])}")
                safe_print(f"    - Engine Score:{row['score']:.4f}")
                safe_print(f"    - 💡 Rationale: {row['explanation']}")
                safe_print("-" * 65)
        safe_print("=" * 65 + "\n")

    def show_product_similarity(self):
        """Search products and show content-based similarity using embeddings."""
        safe_print("\n" + "=" * 65)
        safe_print(" SEMANTIC PRODUCT SIMILARITY TESTER ".center(65, "="))
        safe_print("=" * 65)
        keyword = input("Enter product keyword or brand to search: ").strip()
        if not keyword:
            return
            
        # Search in mapping
        df_map = self.recommender.df_mapping
        matches = df_map[
            df_map['product_name'].str.contains(keyword, case=False, na=False) |
            df_map['brand'].str.contains(keyword, case=False, na=False)
        ].head(10)
        
        if matches.empty:
            safe_print(" No products matched your keyword.")
            return
            
        safe_print(f"\nFound {len(matches)} matching products:")
        for _, row in matches.iterrows():
            safe_print(f"  [ID: {row['product_id']}] [{row['brand'].upper()}] {row['product_name'][:70]}... ({format_vnd(row['price'])})")
            
        try:
            target_pid = int(input("\nEnter Product ID to find similar items: ").strip())
        except ValueError:
            safe_print("Invalid Product ID.")
            return
            
        if target_pid not in self.recommender.embeddings:
            safe_print(f"Product ID {target_pid} not found or has no embedding vector.")
            return
            
        target_vec = self.recommender.embeddings[target_pid]
        target_name = self.recommender.prod_name.get(target_pid, 'Unknown')
        target_brand = self.recommender.prod_brand.get(target_pid, 'Unknown')
        
        safe_print("\n" + f" Selected Target Product: [{target_brand.upper()}] {target_name} ".center(65, "-"))
        desc = self.prod_desc.get(target_pid, "")
        if desc:
            safe_print(f" Description Summary:\n   {desc[:140]}...")
            
        # Calculate similarity with all other products
        sim_scores = []
        for pid, vec in self.recommender.embeddings.items():
            if pid == target_pid:
                continue
            sim = self.recommender._cosine_similarity(target_vec, vec)
            sim_scores.append((pid, sim))
            
        # Sort and take top 5
        sim_scores.sort(key=lambda x: x[1], reverse=True)
        top_sims = sim_scores[:5]
        
        safe_print("\n" + "=" * 65)
        safe_print(" TOP 5 MOST SEMANTICALLY SIMILAR PRODUCTS ".center(65, "="))
        safe_print(" (Computed via 128-dimensional fused PCA embeddings) ")
        safe_print("=" * 65)
        
        for idx, (pid, sim) in enumerate(top_sims):
            name = self.recommender.prod_name.get(pid, 'Unknown')
            brand = self.recommender.prod_brand.get(pid, 'Unknown')
            price = self.recommender.prod_price.get(pid, 0)
            
            safe_print(f" {idx+1}. [{brand.upper()}] {name}")
            safe_print(f"    - Product ID:  {pid}")
            safe_print(f"    - Price:       {format_vnd(price)}")
            safe_print(f"    - Cosine Sim:  {sim * 100:.2f}% Match")
            desc_sub = self.prod_desc.get(pid, "")
            if desc_sub:
                safe_print(f"    - Description: {desc_sub[:100]}...")
            safe_print("-" * 65)
        safe_print("=" * 65 + "\n")

    def run_cli(self):
        """Run the interactive command-line interface."""
        while True:
            safe_print("*" * 65)
            safe_print(" MULTIMODAL E-COMMERCE RECOMMENDATION SYSTEM TESTER ".center(65, " "))
            safe_print("*" * 65)
            safe_print(" 1. Test Customer ID 142  (🌱 Promising Newcomer - Highly Satisfied)")
            safe_print(" 2. Test Customer ID 32   (🗯️ Negative Detractor - Highly Dissatisfied)")
            safe_print(" 3. Test Customer ID 3    (💤 General/Hibernating - Segment Group 1)")
            safe_print(" 4. Test Customer ID 228  (💤 General/Hibernating - Segment Group 4)")
            safe_print(" 5. Test Customer ID 3144 (💤 General/Hibernating - Segment Group 0)")
            safe_print(" 6. Input Custom Customer ID (Test custom history / Cold start)")
            safe_print(" 7. Find Semantically Similar Products (Explore Multimodal Embeddings)")
            safe_print(" 8. Exit")
            safe_print("*" * 65)
            
            choice = input("Select an option (1-8): ").strip()
            
            if choice == '1':
                self.show_recommendations(142)
            elif choice == '2':
                self.show_recommendations(32)
            elif choice == '3':
                self.show_recommendations(3)
            elif choice == '4':
                self.show_recommendations(228)
            elif choice == '5':
                self.show_recommendations(3144)
            elif choice == '6':
                try:
                    user_id = int(input("Enter Customer ID: ").strip())
                    self.show_recommendations(user_id)
                except ValueError:
                    safe_print("Invalid Customer ID. Please enter an integer.")
            elif choice == '7':
                self.show_product_similarity()
            elif choice == '8':
                safe_print("Exiting. Thank you for testing!")
                break
            else:
                safe_print("Invalid selection. Please choose from 1 to 8.")
                
            input("\nPress Enter to return to Menu...")
            clear_screen()

if __name__ == '__main__':
    # Force output encoding to UTF-8 to handle Vietnamese text in terminal properly
    try:
        sys.stdout.reconfigure(encoding='utf-8')
    except Exception:
        pass
    app = PredictDemoApp()
    clear_screen()
    app.run_cli()
