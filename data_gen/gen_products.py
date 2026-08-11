"""
Generate product catalogue (products.csv).

BASE FEATURES generated here:
  - category
  - price               (log-normal per category)
  - is_non_returnable   (category-driven probability)
  - review_count        (log-normal, right-skewed)
  - sku                 (stock keeping unit = units in stock, log-normal)
  - category_base_defect_rate  (fixed per category)
  - return_window_days  (policy config per category)

NOTE: category one-hot encoding is done in the feature-engineering pipeline,
      not here. We store the raw category string.
"""

import numpy as np
import pandas as pd

from config import (
    RANDOM_SEED, N_PRODUCTS,
    CATEGORIES, CATEGORY_WEIGHTS,
    CATEGORY_PRICE_PARAMS, PRODUCT,
    NON_RETURNABLE_PROB, RETURN_WINDOW_DAYS, CATEGORY_DEFECT_RATE,
)


def generate_products(seed: int = RANDOM_SEED) -> pd.DataFrame:
    rng = np.random.default_rng(seed + 1)        # offset seed to differ from sellers

    # ── Category assignment ────────────────────────────────────────────────────
    cat_weights = np.array(CATEGORY_WEIGHTS, dtype=float)
    cat_weights /= cat_weights.sum()              # normalise (safety)
    categories = rng.choice(CATEGORIES, size=N_PRODUCTS, p=cat_weights)

    # ── Price per product (log-normal per category) ───────────────────────────
    prices = np.zeros(N_PRODUCTS, dtype=float)
    for i, cat in enumerate(categories):
        p = CATEGORY_PRICE_PARAMS[cat]
        prices[i] = rng.lognormal(p["mu"], p["sigma"])
    prices = np.clip(prices, 10, 5_00_000).round(2)

    # ── Is non-returnable (category-driven) ───────────────────────────────────
    is_non_returnable = np.array(
        [rng.random() < NON_RETURNABLE_PROB[cat] for cat in categories],
        dtype=int,
    )

    # ── Review count (log-normal, heavy right tail) ────────────────────────────
    p = PRODUCT["review_count_lognormal"]
    review_count = rng.lognormal(p["mu"], p["sigma"], size=N_PRODUCTS)
    review_count = np.clip(review_count, p["min_val"], p["max_val"]).astype(int)

    # ── SKU = stock keeping unit (units in stock) ──────────────────────────────
    p = PRODUCT["sku_lognormal"]
    sku = rng.lognormal(p["mu"], p["sigma"], size=N_PRODUCTS)
    sku = np.clip(sku, p["min_val"], p["max_val"]).astype(int)

    # ── Category-level features (same value for all products in a category) ────
    defect_rate    = np.array([CATEGORY_DEFECT_RATE[cat]    for cat in categories])
    return_window  = np.array([RETURN_WINDOW_DAYS[cat]      for cat in categories])

    # ── Assemble ───────────────────────────────────────────────────────────────
    df = pd.DataFrame(
        {
            "product_id":                  [f"PROD_{i + 1:05d}" for i in range(N_PRODUCTS)],
            "category":                    categories,
            "price":                       prices,
            "is_non_returnable":           is_non_returnable,
            "review_count":                review_count,
            "sku":                         sku,
            "category_base_defect_rate":   defect_rate,
            "return_window_days":          return_window,
        }
    )

    return df


if __name__ == "__main__":
    df = generate_products()
    print("Products per category:")
    print(df["category"].value_counts())
    print("\nPrice statistics by category:")
    print(df.groupby("category")["price"].describe()[["min", "50%", "max"]].round(0))
    print("\nNon-returnable fraction by category:")
    print(df.groupby("category")["is_non_returnable"].mean().round(2))
