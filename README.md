# AI Recommendation Logic — Digital Matchmaker
> **Industrial Training Project 3** | DecodeLabs Industrial Training Kit

An AI-driven **Content-Based Recommendation Engine** designed to solve *Choice Overload* by matching user preferences and multi-dimensional intent vectors with item feature sets using Term Frequency-Inverse Document Frequency (**TF-IDF**) and **Cosine Similarity** mathematics.

---

## 📌 Features & Core Requirements

- **Input-Process-Output (IPO) Model**: Structured end-to-end data pipeline transforming qualitative user intent into ranked quantitative recommendations.
- **Content-Based Filtering**: Item-attribute matching independent of massive community historical interactions.
- **Vector Mapping & TF-IDF Weighting**: Penalizes generic terms (high document frequency) while boosting unique, highly descriptive tags.
- **Cosine Similarity Engine**: Evaluates angular preference alignment ($0.0$ to $1.0$) invariant to vector magnitude (unlike Euclidean distance).
- **4-Step Pipeline**:
  1. **Ingestion**: Ingests $\ge 3$ user preferences or skills with preference weights.
  2. **Scoring**: Computes normalized Cosine Similarity scores.
  3. **Sorting**: Sorts candidates in descending order.
  4. **Filtering**: Generates a clean, truncated **Top-N** recommendation list.
- **Cold-Start Resilience**: Built-in **Trending / Global Popularity Fallback** to handle zero-vector/new-user cold starts seamlessly.
- **Advanced Enhancements**:
  - **User Preference Rating / Weighting**: Dynamic profile weighting using 1–5 star ratings.
  - **Item-to-Item Similarity Matrix**: Enables "Users who liked this also liked..." cross-item similarity exploration.

---

## 📂 Project Structure

```text
├── Dataset for Data Analytics.xlsx   # Source dataset (E-commerce / Analytics)
├── ai_recommender.py                 # Core AI Recommendation Engine script
├── requirements.txt                  # Python dependencies
└── README.md                         # Project documentation
```

---

## 🛠️ Installation & Setup

### 1. Clone or Open the Repository
Open the project directory inside VS Code or your terminal:
```bash
git clone https://github.com/<your-username>/ai-recommendation-logic.git
cd ai-recommendation-logic
```

### 2. Install Dependencies
Install all required packages using `pip` or `uv`:
```bash
pip install pandas scikit-learn numpy openpyxl
```
*Or using `uv`:*
```bash
uv pip install pandas scikit-learn numpy openpyxl
```

---

## 🚀 Running the Recommendation Engine

Run the main Python script from your terminal:
```bash
python ai_recommender.py
```

### 💻 Sample Execution Output

```text
=================================================================
      DECODELABS AI RECOMMENDATION ENGINE (PROJECT 3)     
=================================================================

[+] Loaded 7 Items into Vector Space.
Available Items in System: Monitor, Phone, Tablet, Chair, Printer, Laptop, Desk

-----------------------------------------------------------------
TEST CASE 1: Ingesting >= 3 Choices with Preference Ratings
-----------------------------------------------------------------
User Input Profile: {'Laptop': 5.0, 'HighResolution': 4.0, 'Coding': 5.0, 'Office': 2.0}

--- TOP-N RECOMMENDATIONS ---
#1 Laptop - Score: 0.6842 (68.4% Match)
   Attributes: Laptop Google Referral Cash Portable Computer Performance Workstation...
#2 Monitor - Score: 0.4215 (42.2% Match)
   Attributes: Monitor Instagram Referral Debit Card Display Screen HighResolution...
#3 Desk - Score: 0.2104 (21.0% Match)
   Attributes: Desk Google Email Online Standing Office Furniture Setup Table Workstation...

-----------------------------------------------------------------
TEST CASE 2: Item-to-Item Similarity Matrix (Advance Enhancement)
-----------------------------------------------------------------
Users interested in 'Laptop' might also like:
 -> Monitor (Correlation Score: 0.3842)
 -> Desk (Correlation Score: 0.2195)

-----------------------------------------------------------------
TEST CASE 3: Cold Start Resilience (Empty / Unknown tags)
-----------------------------------------------------------------
[!] Cold Start: No input provided. Using Trending Fallback.
#1 Monitor (Fallback Popularity Rank #1)
#2 Phone (Fallback Popularity Rank #2)
#3 Tablet (Fallback Popularity Rank #3)
```

---

## 📐 Mathematical Framework

### 1. TF-IDF Weighting
$$\text{TF}(t, d) = \frac{\text{Count of term } t \text{ in document } d}{\text{Total terms in document } d}$$

$$\text{IDF}(t) = \log\left(\frac{\text{Total Documents}}{\text{Documents containing term } t}\right)$$

### 2. Cosine Similarity Formula
$$\cos(	heta) = rac{\mathbf{A} \cdot \mathbf{B}}{\|\mathbf{A}\| \|\mathbf{B}\|} = rac{\sum_{i=1}^{n} A_i B_i}{\sqrt{\sum_{i=1}^{n} A_i^2} \sqrt{\sum_{i=1}^{n} B_i^2}}$$

---

## 🎯 Verification Checklist

- [x] Input-Process-Output (IPO) Architecture
- [x] Content-Based Recommendation Logic
- [x] Scikit-Learn TF-IDF Feature Mapping
- [x] Angular Cosine Similarity Calculation
- [x] 4-Step Pipeline Execution
- [x] Minimum 3 Input Preference Ingestion
- [x] Top-N Truncation
- [x] Cold Start / Trending Fallback Handling
- [x] User Preference Rating Weight Multipliers (1–5)
- [x] Item-to-Item Correlation Matrix Lookup
