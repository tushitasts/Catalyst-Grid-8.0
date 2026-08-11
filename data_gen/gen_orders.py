"""
Generate orders.csv — the full order transaction log.

BASE FEATURES generated here:
  - order_id, user_id, product_id, seller_id
  - order_value     (product_price x (1 - discount_pct/100))
  - is_prepaid
  - discount_pct
  - order_date
  - delivery_date
  - is_returned     (flag; only returnable products can be True)

GENERATION LOGIC:
  1. Build a per-category seller index so each order is fulfilled by a seller
     that actually stocks the ordered category (realistic marketplace behaviour).
  2. For each user, generate N orders chronologically (dates increment).
  3. Each order samples category from user's affinity, product from that category,
     then seller filtered to those who carry that category.
  4. All orders globally sorted by order_date (interleaves users naturally).

NOTE: Derived features (avg_order_value, orders_last_30d, order_value_percentile)
      are NOT stored here — computed in the feature-engineering pipeline.
"""

import numpy as np
import pandas as pd
from datetime import datetime, timedelta

from config import (
    RANDOM_SEED,
    CATEGORIES, CATEGORY_WEIGHTS,
    BUYER_PRICE_SEGMENTS, CATEGORY_EXPLORATION_PROB,
    ORDER,
)


# ── Helpers ────────────────────────────────────────────────────────────────────

def _build_category_product_index(products_df: pd.DataFrame) -> dict:
    """Pre-index products by category for O(1) category-based lookup."""
    return {
        cat: products_df[products_df["category"] == cat].reset_index(drop=True)
        for cat in CATEGORIES
    }


def _build_category_seller_index(sellers_df: pd.DataFrame) -> dict:
    """
    Build a per-category seller index: {category -> {ids, weights}}.
    Uses '_seller_categories' (internal list column from gen_sellers.py).

    Sellers that stock a category are eligible to fulfil orders in that category.
    Their popularity weights are preserved and re-normalised per category.
    """
    cat_seller_ids     = {cat: [] for cat in CATEGORIES}
    cat_seller_weights = {cat: [] for cat in CATEGORIES}

    for _, row in sellers_df.iterrows():
        cats = row.get("_seller_categories", [])
        w    = float(row["seller_order_weight"])
        sid  = row["seller_id"]
        for cat in cats:
            if cat in cat_seller_ids:
                cat_seller_ids[cat].append(sid)
                cat_seller_weights[cat].append(w)

    # Normalise weights per category; convert to arrays for fast sampling
    cat_seller_index = {}
    for cat in CATEGORIES:
        ids = cat_seller_ids[cat]
        if ids:
            w = np.array(cat_seller_weights[cat], dtype=float)
            w /= w.sum()
            cat_seller_index[cat] = {"ids": np.array(ids), "weights": w}
        else:
            cat_seller_index[cat] = None   # fallback: sample from all sellers

    return cat_seller_index


def _sample_product(rng, cat_products: pd.DataFrame, price_lo: float, price_hi: float):
    """Sample a product within price range; falls back if no match."""
    eligible = cat_products[
        (cat_products["price"] >= price_lo) & (cat_products["price"] <= price_hi)
    ]
    if len(eligible) == 0:
        eligible = cat_products           # relax price constraint
    idx = rng.integers(0, len(eligible))
    return eligible.iloc[idx]


def _sample_category(rng, cat_prefs: list) -> str:
    """
    Pick category from user's preferences (82% of time) or explore randomly (18%).
    """
    if rng.random() > CATEGORY_EXPLORATION_PROB:
        return str(rng.choice(cat_prefs))
    else:
        cw = np.array(CATEGORY_WEIGHTS, dtype=float)
        return str(rng.choice(CATEGORIES, p=cw / cw.sum()))


def _sample_seller(rng, category: str, cat_seller_index: dict,
                   fallback_ids: np.ndarray, fallback_weights: np.ndarray) -> str:
    """
    Sample a seller that stocks 'category'.
    Falls back to all sellers if no specialist found for this category.
    """
    entry = cat_seller_index.get(category)
    if entry is not None:
        return str(rng.choice(entry["ids"], p=entry["weights"]))
    return str(rng.choice(fallback_ids, p=fallback_weights))


# ── Main function ──────────────────────────────────────────────────────────────

def generate_orders(
    users_df:    pd.DataFrame,
    profiles_df: pd.DataFrame,
    products_df: pd.DataFrame,
    sellers_df:  pd.DataFrame,
    seed: int = RANDOM_SEED,
) -> pd.DataFrame:
    """
    Returns orders_df with all columns including internal '_behavior_type'
    (prefixed '_'). run_pipeline.py drops internal columns before saving.
    """
    rng = np.random.default_rng(seed + 3)

    date_start = datetime.strptime("2022-01-01", "%Y-%m-%d")
    date_end   = datetime.strptime("2025-07-01", "%Y-%m-%d")

    # ── Build indices ──────────────────────────────────────────────────────────
    cat_product_index = _build_category_product_index(products_df)
    cat_seller_index  = _build_category_seller_index(sellers_df)

    # Fallback: all sellers (used when a category has no specialist)
    all_seller_ids     = sellers_df["seller_id"].values
    all_seller_weights = sellers_df["seller_order_weight"].values.astype(float)
    all_seller_weights /= all_seller_weights.sum()

    all_rows  = []
    order_seq = 0

    for user_idx in range(len(profiles_df)):
        profile = profiles_df.iloc[user_idx]
        user    = users_df.iloc[user_idx]

        user_id       = user["user_id"]
        n_orders      = int(profile["n_orders"])
        price_seg     = profile["price_segment"]
        cat_prefs     = profile["category_prefs"]
        return_rate   = float(profile["return_rate"])
        behavior_type = profile["behavior_type"]
        account_age   = int(user["account_age_days"])

        price_lo, price_hi = BUYER_PRICE_SEGMENTS[price_seg]["price_range"]

        if n_orders == 0:
            continue

        # ── First order date ───────────────────────────────────────────────────
        acct_start     = date_end - timedelta(days=account_age)
        acct_start     = max(acct_start, date_start)
        available_days = max((date_end - acct_start).days - 10, 1)
        first_offset   = int(rng.integers(0, available_days))
        first_date     = acct_start + timedelta(days=first_offset)

        # ── Inter-order gaps (lognormal) ───────────────────────────────────────
        p    = ORDER["gap_days_lognormal"]
        gaps = rng.lognormal(p["mu"], p["sigma"], size=max(n_orders - 1, 0))
        gaps = np.clip(gaps, p["min_val"], p["max_val"]).astype(int)

        order_dates = [first_date]
        for gap in gaps:
            next_d = order_dates[-1] + timedelta(days=int(gap))
            if next_d >= date_end:
                break
            order_dates.append(next_d)

        # ── Generate individual orders ─────────────────────────────────────────
        for od in order_dates:

            # Category (from user's affinity or random exploration)
            category = _sample_category(rng, cat_prefs)

            # Product (from this category + user's price range)
            cat_prods = cat_product_index.get(category, products_df)
            product   = _sample_product(rng, cat_prods, price_lo, price_hi)

            product_id     = product["product_id"]
            base_price     = float(product["price"])
            non_returnable = bool(product["is_non_returnable"])
            return_window  = int(product["return_window_days"])

            # Discount
            d_a = ORDER["discount_beta"]["a"]
            d_b = ORDER["discount_beta"]["b"]
            discount_pct = round(rng.beta(d_a, d_b) * ORDER["discount_max_pct"], 1)

            order_value = round(base_price * (1.0 - discount_pct / 100.0), 2)
            order_value = max(order_value, 1.0)

            # Seller — filtered to those who stock this category
            seller_id = _sample_seller(rng, category, cat_seller_index,
                                       all_seller_ids, all_seller_weights)

            # Payment mode
            is_prepaid = int(rng.random() < ORDER["prepaid_prob"])

            # Delivery date
            dlv_gap       = int(rng.integers(ORDER["delivery_days_min"],
                                             ORDER["delivery_days_max"] + 1))
            delivery_date = od + timedelta(days=dlv_gap)

            # Return flag: non-returnable products can never be returned
            if non_returnable or return_window == 0:
                is_returned = 0
            else:
                is_returned = int(rng.random() < return_rate)

            order_seq += 1
            all_rows.append(
                {
                    "order_id":       f"ORD_{order_seq:08d}",
                    "user_id":        user_id,
                    "product_id":     product_id,
                    "seller_id":      seller_id,
                    "order_value":    order_value,
                    "is_prepaid":     is_prepaid,
                    "discount_pct":   discount_pct,
                    "order_date":     od.strftime("%Y-%m-%d"),
                    "delivery_date":  delivery_date.strftime("%Y-%m-%d"),
                    "is_returned":    is_returned,
                    "_behavior_type": behavior_type,   # internal; dropped before CSV
                }
            )

    orders_df = pd.DataFrame(all_rows)

    # ── Global sort by order_date ──────────────────────────────────────────────
    # Naturally interleaves orders from different users while preserving
    # each user's intra-user chronological order.
    orders_df["_order_date_dt"] = pd.to_datetime(orders_df["order_date"])
    orders_df = (
        orders_df
        .sort_values("_order_date_dt")
        .reset_index(drop=True)
        .drop(columns=["_order_date_dt"])
    )

    # Re-number order_ids sequentially in final sort order
    orders_df["order_id"] = [f"ORD_{i + 1:08d}" for i in range(len(orders_df))]

    return orders_df


if __name__ == "__main__":
    from gen_sellers  import generate_sellers
    from gen_products import generate_products
    from gen_users    import generate_users

    sellers          = generate_sellers()
    products         = generate_products()
    users, profiles  = generate_users()
    orders           = generate_orders(users, profiles, products, sellers)

    print(f"Total orders   : {len(orders):,}")
    print(f"Returned orders: {orders['is_returned'].sum():,} "
          f"({orders['is_returned'].mean()*100:.1f}%)")
    print(f"\nDate range: {orders['order_date'].min()} to {orders['order_date'].max()}")
    print(f"\nOrder value stats:")
    print(orders["order_value"].describe().round(2))

    # Verify category-seller alignment
    print("\nSample seller-category checks (should be consistent):")
    merged = orders.merge(sellers[["seller_id", "seller_type", "_seller_categories"]],
                          on="seller_id")
    merged = merged.merge(products[["product_id", "category"]], on="product_id")
    mismatch = merged[~merged.apply(
        lambda r: r["category"] in r["_seller_categories"], axis=1
    )]
    print(f"  Category-seller mismatches: {len(mismatch)} "
          f"(should be ~0 or very few from fallback)")
