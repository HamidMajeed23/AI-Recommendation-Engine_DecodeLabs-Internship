import os
import pandas as pd
import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

# =====================================================================
# 1. DATASET INITIALIZATION
# =====================================================================
def get_ecommerce_metadata(excel_path="Dataset for Data Analytics.xlsx"):
    """
    Ingests the provided Excel dataset and builds rich item metadata profiles.
    """
    if os.path.exists(excel_path):
        df_raw = pd.read_excel(excel_path)
        
        # Product-level profile build using associated channels & tags
        grouped = df_raw.groupby('Product').agg({
            'ReferralSource': lambda x: " ".join(set(x)),
            'PaymentMethod': lambda x: " ".join(set(x)),
            'UnitPrice': 'mean',
            'Quantity': 'sum'
        }).reset_index()
        
        # Build contextual metadata tag string
        grouped['metadata'] = (
            grouped['Product'] + " " +
            grouped['ReferralSource'] + " " +
            grouped['PaymentMethod'] + " " +
            grouped['Product'].map({
                'Monitor': 'Display Screen HighResolution Office Electronics Hardware',
                'Phone': 'Mobile Smartphone Communication iOS Android Device',
                'Tablet': 'Touchscreen Portable iPad Media Android Device',
                'Chair': 'Ergonomic Office Furniture Seating Comfort Workstation',
                'Printer': 'Office Scanning Printing Hardware Document Peripheral',
                'Laptop': 'Portable Computer Performance Workstation Hardware Coding',
                'Desk': 'Standing Office Furniture Setup Table Workstation'
            })
        )
        grouped['popularity_rank'] = grouped['Quantity'].rank(ascending=False).astype(int)
        return grouped[['Product', 'metadata', 'popularity_rank']].rename(
            columns={'Product': 'item_name', 'metadata': 'features'}
        )
    else:
        # Fallback to standard Tech Stack dataset
        tech_data = {
            "item_name": [
                "Data Scientist", "DevOps Engineer", "Backend Developer",
                "Cloud Architect", "Frontend Developer", "AI/ML Engineer", "System Administrator"
            ],
            "features": [
                "Python SQL Machine Learning Statistics Pandas Scikit-Learn Data Analysis",
                "AWS Docker Kubernetes CI/CD Terraform Linux Cloud Automation Bash",
                "Python Java SQL APIs PostgreSQL FastAPI Django Node.js Redis",
                "AWS Azure GCP Cloud Networking Security Terraform Docker Kubernetes",
                "JavaScript React CSS HTML TypeScript Redux UI/UX Tailwind",
                "Python PyTorch TensorFlow Deep Learning Machine Learning NLP Docker CUDA",
                "Linux Bash Networking Shell Scripting SysAdmin Security Troubleshooting"
            ],
            "popularity_rank": [1, 2, 3, 4, 5, 6, 7]
        }
        return pd.DataFrame(tech_data)


# =====================================================================
# 2. DIGITAL MATCHMAKER RECOMMENDER ENGINE
# =====================================================================
class DigitalMatchmaker:
    def __init__(self, items_df: pd.DataFrame):
        self.df = items_df.reset_index(drop=True)
        self.vectorizer = TfidfVectorizer(token_pattern=r'(?u)\b\w+\b')
        self.tfidf_matrix = self.vectorizer.fit_transform(self.df["features"])
        self.item_similarity_matrix = cosine_similarity(self.tfidf_matrix)

    # --- Cold Start Bypass Strategy ---
    def get_trending_fallback(self, top_n: int = 3):
        """Bypasses cold start using global popularity/trending metrics."""
        fallback_items = self.df.sort_values(by="popularity_rank").head(top_n)
        return [
            {
                "item_name": row["item_name"],
                "features": row["features"],
                "similarity_score": 0.0,
                "is_fallback": True
            }
            for _, row in fallback_items.iterrows()
        ]

    # --- 4-Step Ranking Pipeline with User Preference Rating Support ---
    def recommend(self, user_preferences: dict, top_n: int = 3):
        """
        user_preferences: Dict with { 'term/skill': rating_weight (e.g. 1 to 5) }
        """
        # Step 1: Ingestion
        if not user_preferences:
            print("\n[!] Cold Start: No input provided. Using Trending Fallback.")
            return self.get_trending_fallback(top_n)

        # Advanced Enhancement: Weighted User Profile using preference ratings
        weighted_terms = []
        for term, weight in user_preferences.items():
            clean_term = term.strip()
            if clean_term:
                # Repeat term based on user rating weight (1-5)
                multiplier = max(1, int(round(weight)))
                weighted_terms.extend([clean_term] * multiplier)

        user_query = " ".join(weighted_terms)
        user_vec = self.vectorizer.transform([user_query])

        if user_vec.nnz == 0:
            print("\n[!] Cold Start: Vocabulary mismatch. Using Trending Fallback.")
            return self.get_trending_fallback(top_n)

        # Step 2: Scoring (Cosine Similarity)
        scores = cosine_similarity(user_vec, self.tfidf_matrix).flatten()

        # Step 3: Sorting (Descending Order)
        ranked_indices = np.argsort(scores)[::-1]

        # Step 4: Filtering (Top-N Truncation)
        results = []
        for idx in ranked_indices[:top_n]:
            score = float(scores[idx])
            if score > 0.0:
                results.append({
                    "item_name": self.df.iloc[idx]["item_name"],
                    "features": self.df.iloc[idx]["features"],
                    "similarity_score": round(score, 4),
                    "is_fallback": False
                })

        if not results:
            return self.get_trending_fallback(top_n)

        return results

    # --- Advance Enhancement: Item-to-Item Similarity Lookup ---
    def get_similar_items(self, item_name: str, top_n: int = 2):
        """Calculates inter-item similarity from the Item Similarity Matrix."""
        match_idx = self.df.index[self.df['item_name'].str.lower() == item_name.lower()].tolist()
        if not match_idx:
            return []
        
        idx = match_idx[0]
        sim_scores = list(enumerate(self.item_similarity_matrix[idx]))
        sim_scores = sorted(sim_scores, key=lambda x: x[1], reverse=True)
        
        # Exclude self-match
        similar_items = []
        for i, score in sim_scores[1:top_n+1]:
            similar_items.append((self.df.iloc[i]['item_name'], round(score, 4)))
        return similar_items


# =====================================================================
# 3. INTERACTIVE DEMO / TESTING
# =====================================================================
def main():
    print("="*65)
    print("      DECODELABS AI RECOMMENDATION ENGINE (PROJECT 3)     ")
    print("="*65)

    df_items = get_ecommerce_metadata()
    engine = DigitalMatchmaker(df_items)

    print(f"\n[+] Loaded {len(df_items)} Items into Vector Space.")
    print("Available Items in System:", ", ".join(df_items['item_name'].tolist()))

    # Sample Input with Ratings (Advance Enhancement)
    print("\n" + "-"*65)
    print("TEST CASE 1: Ingesting >= 3 Choices with Preference Ratings")
    print("-" * 65)
    sample_profile = {
        "Laptop": 5.0,        # Highly rated
        "HighResolution": 4.0,
        "Coding": 5.0,
        "Office": 2.0
    }
    print(f"User Input Profile: {sample_profile}")
    
    recommendations = engine.recommend(sample_profile, top_n=3)
    
    print("\n--- TOP-N RECOMMENDATIONS ---")
    for rank, item in enumerate(recommendations, 1):
        status = "(Trending Fallback)" if item["is_fallback"] else f"({item['similarity_score'] * 100:.1f}% Match)"
        print(f"#{rank} {item['item_name']} - Score: {item['similarity_score']} {status}")
        print(f"   Attributes: {item['features'][:70]}...")

    # Advance Enhancement Demo: Item-to-Item Similarity Matrix Lookup
    print("\n" + "-"*65)
    print("TEST CASE 2: Item-to-Item Similarity Matrix (Advance Enhancement)")
    print("-" * 65)
    test_item = recommendations[0]['item_name']
    related = engine.get_similar_items(test_item, top_n=2)
    print(f"Users interested in '{test_item}' might also like:")
    for sim_item, score in related:
        print(f" -> {sim_item} (Correlation Score: {score})")

    # Cold-Start Test Case
    print("\n" + "-"*65)
    print("TEST CASE 3: Cold Start Resilience (Empty / Unknown tags)")
    print("-" * 65)
    cold_results = engine.recommend({}, top_n=3)
    for rank, item in enumerate(cold_results, 1):
        print(f"#{rank} {item['item_name']} (Fallback Popularity Rank #{rank})")


if __name__ == "__main__":
    main() 