"""
Generate seller profiles (sellers.csv).

BASE FEATURES generated here (saved to sellers.csv):
  - seller_id
  - seller_age_days
  - seller_rating         (age-consistent: new sellers -> lower rating)
  - seller_return_rate
  - seller_customer_frequency
  - seller_type           (archetype: fashion_lifestyle, electronics_tech, etc.)

INTERNAL columns (prefixed with '_', dropped before CSV save):
  - seller_order_weight   (popularity weight for order assignment)
  - _seller_categories    (list of categories this seller stocks; drives gen_orders.py)

SELLER TYPE LOGIC
-----------------
Each seller is assigned a type from SELLER_TYPES in config.py.
The type determines:
  - Which categories the seller stocks (primary + secondary)
  - Order assignment in gen_orders.py: a furniture seller never sells electronics.

This creates realistic seller profiles — a fashion seller on Flipkart carries
Clothing + Footwear + Beauty, while a big general marketplace seller carries many.
"""

import numpy as np
import pandas as pd

from config import RANDOM_SEED, N_SELLERS, SELLER, SELLER_POPULARITY_ALPHA, SELLER_TYPES


def _assign_seller_types(rng: np.random.Generator, n: int):
    """
    Assign seller archetypes and build their category portfolios.

    Returns
    -------
    seller_type_names  : list of str (one per seller)
    seller_categories  : list of list[str] (categories each seller carries)
    """
    types  = list(SELLER_TYPES.keys())
    probs  = np.array([SELLER_TYPES[t]["prob"] for t in types], dtype=float)
    probs /= probs.sum()

    assigned_types = rng.choice(types, size=n, p=probs)

    seller_categories = []
    for st in assigned_types:
        cfg = SELLER_TYPES[st]
        seller_categories.append(list(cfg["categories"]))

    return list(assigned_types), seller_categories


def generate_sellers(seed: int = RANDOM_SEED) -> pd.DataFrame:
    """
    Returns sellers_df with all columns.
    run_pipeline.py calls drop_internal() to remove '_*' columns before saving.

    Note: 'seller_type' IS saved to sellers.csv — it is a real base feature.
          '_seller_categories' is internal only (list-typed, can't serialise well).
    """
    rng = np.random.default_rng(seed)

    # ── Seller age ────────────────────────────────────────────────────────────
    p = SELLER["age_days_lognormal"]
    age_days = rng.lognormal(mean=p["mu"], sigma=p["sigma"], size=N_SELLERS)
    age_days = np.clip(age_days, p["min_val"], p["max_val"]).astype(int)

    # ── Seller type + category portfolio ──────────────────────────────────────
    seller_type_names, seller_categories = _assign_seller_types(rng, N_SELLERS)

    # ── Seller rating (0–5, consistent with age AND type) ─────────────────────
    # New sellers have fewer reviews → lower, noisier ratings.
    # General marketplaces tend to have slightly higher ratings (established brand).
    threshold = SELLER["new_seller_threshold"]
    ratings   = np.zeros(N_SELLERS, dtype=float)

    for i, (age, stype) in enumerate(zip(age_days, seller_type_names)):
        if age <= threshold:
            a = SELLER["rating_new_beta"]["a"]
            b = SELLER["rating_new_beta"]["b"]
        else:
            a = SELLER["rating_veteran_beta"]["a"]
            b = SELLER["rating_veteran_beta"]["b"]
            # General marketplace sellers are slightly more established
            if stype == "general_marketplace":
                a += 1.0

        ratings[i] = rng.beta(a, b) * SELLER["rating_max"]

    ratings = np.clip(ratings, SELLER["rating_min"], SELLER["rating_max"])
    ratings = np.round(ratings, 1)

    # ── Seller return rate ────────────────────────────────────────────────────
    # Beta(1.5, 12): most sellers 5–15%, a few outliers approach 40%.
    # Fashion and Clothing sellers tend to have higher return rates (sizing issues).
    a, b = SELLER["return_rate_beta"]["a"], SELLER["return_rate_beta"]["b"]
    return_rate = np.zeros(N_SELLERS, dtype=float)
    for i, stype in enumerate(seller_type_names):
        ai, bi = a, b
        if stype in ("fashion_lifestyle", "clothing_specialist"):
            ai = a * 1.5    # fashion return rates are higher
            bi = b * 0.85
        elif stype in ("grocery_health", "books_stationery"):
            ai = a * 0.7    # grocery/books are rarely returned
            bi = b * 1.2
        return_rate[i] = rng.beta(ai, bi)

    return_rate = np.clip(return_rate, 0.005, 0.55).round(4)

    # ── Customer frequency (fraction of repeat buyers) ─────────────────────────
    a, b = SELLER["customer_freq_beta"]["a"], SELLER["customer_freq_beta"]["b"]
    customer_freq = rng.beta(a, b, size=N_SELLERS)
    # Specialist sellers tend to have higher repeat customer rates
    for i, stype in enumerate(seller_type_names):
        if stype in ("books_stationery", "electronics_specialist", "clothing_specialist"):
            customer_freq[i] = min(customer_freq[i] * 1.3, 0.95)
    customer_freq = np.clip(customer_freq, 0.01, 0.95).round(3)

    # ── Seller popularity weights for order assignment ─────────────────────────
    # Moderate power-law: a few sellers dominate, but not extremely.
    # General marketplace sellers get a slight popularity boost.
    ranks = np.arange(1, N_SELLERS + 1, dtype=float)
    raw_weights = 1.0 / np.power(ranks, SELLER_POPULARITY_ALPHA - 1)
    raw_weights /= raw_weights.sum()
    rng.shuffle(raw_weights)  # decouple rank from seller_id order

    # Boost general marketplace sellers (they handle more volume)
    for i, stype in enumerate(seller_type_names):
        if stype == "general_marketplace":
            raw_weights[i] *= 1.4

    raw_weights /= raw_weights.sum()  # re-normalise after boost
    seller_order_weight = np.round(raw_weights, 8)

    # ── Assemble DataFrame ─────────────────────────────────────────────────────
    df = pd.DataFrame(
        {
            "seller_id":                 [f"SELL_{i + 1:04d}" for i in range(N_SELLERS)],
            "seller_age_days":           age_days,
            "seller_rating":             ratings,
            "seller_return_rate":        return_rate,
            "seller_customer_frequency": customer_freq,
            "seller_type":               seller_type_names,   # base feature, saved to CSV
            # Internal columns (dropped before CSV save)
            "_seller_categories":        seller_categories,   # list-typed, in-memory only
            "seller_order_weight":       seller_order_weight, # used for order sampling
        }
    )

    return df


if __name__ == "__main__":
    df = generate_sellers()
    print("Seller type distribution:")
    print(df["seller_type"].value_counts())
    print("\nSeller rating by age group:")
    df["age_group"] = pd.cut(df["seller_age_days"], bins=[0, 180, 730, 9999],
                             labels=["new", "mid", "veteran"])
    print(df.groupby("age_group")["seller_rating"].describe().round(2))
    print("\nReturn rate by seller type:")
    print(df.groupby("seller_type")["seller_return_rate"].mean().round(3).sort_values(ascending=False))
    print("\nSample category portfolios:")
    for st in list(df["seller_type"].unique())[:4]:
        row = df[df["seller_type"] == st].iloc[0]
        print(f"  {st}: {row['_seller_categories']}")
