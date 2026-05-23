# Recommendation System Pipeline (BPR + BGE Reranker)

This project implements a modern two-stage recommendation system architecture for an ecommerce book platform.

The system combines:

1. BPR (Bayesian Personalized Ranking) for candidate retrieval
2. BGE Reranker for semantic reranking using:
   - user profile
   - item description

The recommendation system is designed for scalable personalized recommendation using behavioral signals, temporal dynamics, and semantic understanding.

---

# Pipeline Overview

```text
Gold Layer Database
        ↓
Build Time-Aware Interaction Table
        ↓
Train BPR Model
        ↓
Generate Top-50 Candidate Books
        ↓
Build Weighted User Profile
        ↓
Build Item Description
        ↓
BGE Semantic Reranking
        ↓
Top-5 Final Recommendations
        ↓
Cache Recommendations
```

---

# Gold Layer Database Schema

## dim_books

Book dimension table containing metadata for each book.

| Column | Type |
|---|---|
| author | string |
| book_id | int |
| category_name | string |
| price | double |
| purchase_count | int |
| rating_avg | double |
| seller_username | string |
| title | string |

---

## dim_date

Date dimension table.

| Column | Type |
|---|---|
| date | date |
| year | int |
| month | int |
| day | int |
| day_of_week | string |
| quarter | int |
| is_weekend | boolean |

---

## dim_users

User dimension table.

| Column | Type |
|---|---|
| user_id | int |
| username | string |
| full_name | string |
| email | string |
| role | string |
| is_active | boolean |

---

## fact_cart

Cart interaction fact table.

| Column | Type |
|---|---|
| cart_id | int |
| buyer_id | int |
| book_id | int |
| quantity | int |
| added_at | timestamp |

---

## fact_reviews

Book review snapshot table.

| Column | Type |
|---|---|
| book_id | int |
| buyer_id | int |
| score | double |
| total_reviews_at_snapshot | int |
| snapshot_date | timestamp |

---

## fact_sales

Sales transaction fact table.

| Column | Type |
|---|---|
| order_item_id | int |
| order_id | int |
| buyer_id | int |
| seller_id | int |
| book_id | int |
| order_date | timestamp |
| quantity | int |
| unit_price | double |
| line_total | double |
| order_overall_status | string |
| item_status | string |
| payment_method | string |
| payment_status | string |

---

# Step 1 — Build Time-Aware Interaction Table

## Purpose

Transform user behavioral data into implicit feedback scores for collaborative filtering.

The interaction table is constructed from:
- purchases
- cart interactions
- positive reviews
- temporal interaction weighting

This table is used to train the BPR recommendation model.

---

# Interaction Score Design

The recommendation system uses implicit feedback scoring with time decay.

Higher interaction scores indicate stronger and more recent user preference.

---

## Base Interaction Weights

| User Action | Base Score |
|---|---|
| Add to cart | 3 |
| Purchase | 10 |
| Review score >= 4 | 5 |

---

# Time Decay Strategy

Recent interactions should contribute more strongly than old interactions.

This assumption is based on:
- temporal recommendation systems (Wu et al., 2010)
- dynamic user preference modeling (DDCF, 2021)
- collaborative filtering with temporal dynamics (Koren, 2009)

The system applies **exponential time decay per interaction**, using months as the time unit to preserve long-term preference signals — appropriate for a book platform where user tastes evolve slowly.

## Time Decay Formula

```text
decay_weight = exp(-λ × months_ago)

final_interaction_score = SUM(base_score × decay_weight)
                          per (user_id, book_id)
```

Where:
- λ is the decay rate hyperparameter (recommended starting value: 0.05)
- `months_ago = days_ago / 30.0`
- each interaction is decayed individually before aggregation
- recent interactions receive higher weight
- older interactions gradually lose influence but are not zeroed out

## Decay Rate Reference

| λ | 1 month ago | 6 months ago | 12 months ago | 24 months ago |
|---|---|---|---|---|
| 0.01 | 0.99 | 0.94 | 0.89 | 0.79 |
| 0.05 | 0.95 | 0.74 | 0.55 | 0.30 |
| 0.10 | 0.90 | 0.55 | 0.30 | 0.09 |

λ = 0.05 is recommended as a starting point.
Tune based on evaluation metrics (NDCG@5, MAP@5) on the validation set.

---

## Interaction Score SQL

> **Design note:** Time decay is applied at the individual interaction level
> before aggregation. This ensures each interaction contributes its own
> recency-adjusted weight, rather than applying a single decay to a merged score.

```sql
WITH interaction_source AS (

    -- Purchases
    SELECT
        buyer_id,
        book_id,
        DATEDIFF(CURRENT_DATE, order_date) / 30.0 AS months_ago,
        10 AS base_score
    FROM fact_sales
    WHERE item_status = 'completed'

    UNION ALL

    -- Cart interactions
    SELECT
        buyer_id,
        book_id,
        DATEDIFF(CURRENT_DATE, added_at) / 30.0 AS months_ago,
        3 AS base_score
    FROM fact_cart

    UNION ALL

    -- Positive reviews
    SELECT
        buyer_id,
        book_id,
        DATEDIFF(CURRENT_DATE, snapshot_date) / 30.0 AS months_ago,
        CASE WHEN score >= 4 THEN 5 ELSE 0 END AS base_score
    FROM fact_reviews
    WHERE score >= 4

),

decayed_interactions AS (

    SELECT
        buyer_id AS user_id,
        book_id,
        base_score * EXP(-0.05 * months_ago) AS decayed_score
    FROM interaction_source
    WHERE base_score > 0

)

SELECT
    user_id,
    book_id,
    SUM(decayed_score) AS interaction_score
FROM decayed_interactions
GROUP BY user_id, book_id

```

---

## Example Interaction Table

| user_id | book_id | interaction_score |
|---|---|---|
| 101 | 5001 | 17.4 |
| 101 | 5012 | 8.9 |
| 205 | 7003 | 3.7 |

---

# Step 2 — Train BPR Model (Bayesian Personalized Ranking)

## Purpose

Train a pairwise ranking model using implicit feedback.

BPR learns:
- user latent embeddings
- book latent embeddings

These embeddings represent:
- user preferences
- book similarity patterns

---

## Input Tables

### fact_sales
Used to identify purchases.

### fact_cart
Used to identify cart behavior.

### fact_reviews
Used to identify positive feedback.

---

## BPR Input Dataset

| user_id | book_id | interaction_score |
|---|---|---|
| 101 | 5001 | 17.4 |
| 205 | 7003 | 3.7 |

---

## Output

Learned embeddings:
- user embedding matrix
- item embedding matrix

---

# Step 3 — Generate Top-50 Candidate Books

## Purpose

Retrieve the most relevant candidate books using BPR scoring.

Instead of ranking all books in the catalog, ALS retrieves only:
- Top-50 candidate books per user

This improves:
- scalability
- inference speed
- reranking efficiency

---

## Input

- BPR model
- user embeddings
- book embeddings

---

## Output

| user_id | candidate_book_id | mf_score |
|---|---|---|
| 101 | 7001 | 0.95 |
| 101 | 8020 | 0.91 |

---

# Step 4 — Build Weighted User Profile

## Purpose

Construct semantic user preference profiles for semantic reranking.

The user profile summarizes:
- favorite categories
- preferred authors
- recently purchased books
- highly rated books
- purchasing patterns

---

# User Profile Construction

The profile is built using **weighted aggregation** of the time-decayed interaction scores from Step 1.

## Profile Score Formula

```text
profile_score(user, attribute) =
    SUM(interaction_score × time_decay)
    for all books linked to that attribute
```

---

## Top Categories SQL

```sql
SELECT
    i.user_id,
    b.category_name,
    SUM(i.interaction_score) AS category_weight
FROM interaction_table i
JOIN dim_books b ON i.book_id = b.book_id
GROUP BY i.user_id, b.category_name
ORDER BY i.user_id, category_weight DESC
```

Top 3 categories per user are extracted and used in the profile text.

---

## Top Authors SQL

```sql
SELECT
    i.user_id,
    b.author,
    SUM(i.interaction_score) AS author_weight
FROM interaction_table i
JOIN dim_books b ON i.book_id = b.book_id
GROUP BY i.user_id, b.author
ORDER BY i.user_id, author_weight DESC
```

Top 3 authors per user are extracted.

---

## Profile Text Construction

```text
User prefers:
- [Top category 1], [Top category 2], [Top category 3]
- books by [Top author 1], [Top author 2]
- highly rated books (rating >= 4.0)
- recently purchased: [Title 1], [Title 2]
```

---

## Example User Profile

```text
User strongly prefers:
- Fantasy novels, Japanese fiction, self-help books
- Books by Haruki Murakami, James Clear
- Highly rated psychological fiction
- Recently purchased: Norwegian Wood, Atomic Habits
```

---

## Data Sources

### dim_books
- category_name
- author
- title

### fact_sales
- purchased books

### fact_reviews
- positively reviewed books

---

# Step 5 — Build Item Description

## Purpose

Construct semantic descriptions for each candidate book from the Top-50 ALS output.

The item description is used as the item-side input to the BGE reranker.

---

## Example Item Description

```text
Title: Atomic Habits
Author: James Clear
Category: Self-help
Average Rating: 4.8
Popular productivity and habit-building book.
```

---

## Item Description SQL

```sql
SELECT
    book_id,
    CONCAT(
        'Title: ', title, '\n',
        'Author: ', author, '\n',
        'Category: ', category_name, '\n',
        'Average Rating: ', ROUND(rating_avg, 1), '\n',
        'Popularity rank: ', purchase_count
    ) AS item_description
FROM dim_books
WHERE book_id IN (/* Top-50 candidate book_ids */)
```

---

## Data Sources

### dim_books
- title
- author
- category_name
- rating_avg
- purchase_count

---

# Step 6 — BGE Semantic Reranking

## Purpose

Rerank ALS candidate books using semantic relevance between user profile and item description.

---

# Reranking Strategy

Input pair per candidate:

```text
(user_profile, item_description)
```

Example:

```text
Query:    "User likes fantasy novels and Japanese fiction"
Document: "Murakami fantasy novel with psychological themes"
```

The BGE cross-encoder scores each (query, document) pair and ranks candidates by semantic relevance.

---

## Model

BGE Cross-Encoder Reranker from Hugging Face:

- **BAAI/bge-reranker-base**

Selected because:
- lower inference latency than `-large`
- suitable for near real-time recommendation serving
- lower GPU memory requirement
- sufficient precision for Top-50 → Top-5 reranking

---

# Inference Optimization Strategy

Cross-encoder reranking is computationally expensive compared to bi-encoders.
The following strategies reduce latency:

## Candidate Reduction
Only rerank Top-50 ALS candidates — not the full catalog.

## Batch Inference
Rerank multiple users in parallel during batch processing.

## Recommendation Caching
Store precomputed Top-5 recommendations per user.

### Cache Invalidation Triggers
Cache is invalidated and refreshed when:
- user makes a new purchase
- user submits a new review
- ALS model is retrained (scheduled refresh)
- time-based expiry (maximum 24 hours)

## Scheduled Refresh
ALS model and recommendations are retrained/refreshed:
- daily batch (default)
- or triggered by significant new interaction volume

---

## Input

Top-50 ALS candidate books per user.

---

## Output

Top-5 final recommendations.

| Rank | Book | Semantic Score |
|---|---|---|
| 1 | Book A | 0.98 |
| 2 | Book B | 0.95 |
| 3 | Book C | 0.91 |
| 4 | Book D | 0.87 |
| 5 | Book E | 0.83 |

---

# Step 7 — Cold Start Strategy

## Problem

ALS cannot generate embeddings for:
- new users with no interaction history
- newly added books with insufficient interactions

---

# New User Fallback

Use popularity-based recommendation.

## Trigger Condition

```text
IF user has 0 interactions:
    use popularity-based fallback
ELIF user has < 5 interactions:
    use hybrid (ALS weighted low + popularity weighted high)
ELSE:
    use full ALS retrieval
```

## Popularity Ranking Query

```sql
SELECT
    book_id,
    title,
    author,
    category_name,
    rating_avg,
    purchase_count,
    (0.6 * PERCENT_RANK() OVER (ORDER BY purchase_count)
     + 0.4 * PERCENT_RANK() OVER (ORDER BY rating_avg)) AS popularity_score
FROM dim_books
WHERE rating_avg >= 3.5
ORDER BY popularity_score DESC
LIMIT 10
```

A random sample of Top-10 is returned to add diversity and avoid showing the same list to every new user.

---

# New Book Fallback

Use content-based semantic similarity.

## Trigger Condition

```text
IF book has < 10 interactions:
    exclude from ALS training
    use content-based similarity for retrieval
ELSE:
    include in ALS training
```

Book similarity is estimated using:
- category_name
- author
- title
- rating_avg

---

# Step 8 — Evaluation

## Purpose

Evaluate retrieval quality (ALS) and semantic ranking quality (BGE) separately.

---

# Train / Validation / Test Split

The system uses **temporal split** instead of random split to prevent data leakage.

| Period | Usage |
|---|---|
| Jan – Oct | Train |
| Nov | Validation (hyperparameter tuning) |
| Dec | Test (final evaluation) |

This avoids:
- future data leakage
- unrealistically optimistic evaluation results

---

# Ground Truth Definition

Relevant books are defined as:
- books **purchased** by the user in the evaluation period
- books with a **review score >= 4** in the evaluation period

Cart interactions are **excluded** from ground truth — they are weak signals used only for training, not evaluation.

---

# Retrieval Metrics (ALS)

## Recall@50

Measures whether relevant books appear in the Top-50 retrieved candidates.

```text
Target: Recall@50 ≥ 0.75
```

## HitRate@50

Measures whether at least one relevant book appears in the Top-50 candidates.

```text
Target: HitRate@50 ≥ 0.85
```

---

# Reranking Metrics (BGE)

## NDCG@5

Measures ranking quality considering recommendation order.
Higher scores for relevant items ranked near the top.

```text
Target: NDCG@5 ≥ 0.35
```

## MAP@5

Measures average precision across all relevant items in Top-5.

```text
Target: MAP@5 ≥ 0.25
```

## MRR

Measures how early the first relevant recommendation appears.

```text
Target: MRR ≥ 0.40
```

---

# Recommended Architecture

```text
Gold Layer Database
        ↓
Time-Aware Interaction Table
(decay applied per interaction before aggregation)
        ↓
ALS Collaborative Filtering
        ↓
Top-50 Candidate Books
        ↓
Weighted User Profile Builder     Item Description Builder
(top categories + authors,              (title, author,
 recency-weighted)                    category, rating)
        ↓                                    ↓
        └─────────────┬───────────────────────┘
                      ↓
            BGE Semantic Reranker
            (BAAI/bge-reranker-base)
                      ↓
          Top-5 Personalized Recommendations
                      ↓
            Recommendation Cache
            (invalidated on new interaction
             or ALS retrain, max 24h TTL)
```

---

# Technologies

| Component | Technology |
|---|---|
| Distributed processing | Apache Spark / Databricks |
| Collaborative filtering | PySpark ALS |
| Semantic reranking | Hugging Face Transformers — BAAI/bge-reranker-base |
| Storage | Delta Lake |
| Feature engineering | PySpark feature pipeline |

---

# Design Notes

- BPR is used for **high-recall candidate retrieval** — speed and coverage over precision.
- BGE reranker is used for **semantic precision reranking** — quality over speed.
- Time decay is applied **per interaction before aggregation**, not to the aggregated score.
- λ = 0.05 with months as the time unit is the recommended starting point; tune on validation set.
- Use **temporal split** (not random split) to avoid data leakage.
- Recommendation quality depends heavily on **interaction score design** and **user profile quality**.
- Cross-encoder reranking is computationally expensive — limit candidates to Top-50.
- Cache recommendations with clear invalidation triggers to balance freshness and performance.
- Popularity-based fallback is required for cold-start users.
- New books with fewer than 10 interactions should be excluded from ALS training and served via content-based similarity.

---

# References

- Koren, Y. (2009). Collaborative Filtering with Temporal Dynamics. *KDD 2009*.
- Wu, P. et al. (2010). Time-aware Collaborative Filtering with the Piecewise Decay Function. *arXiv:1010.3988*.
- Ghiye, A. et al. (2023). Adaptive Collaborative Filtering with Personalized Time Decay Functions. *RecSys 2023 / arXiv:2308.01208*.
- BAAI/bge-reranker-base. Hugging Face Model Hub.