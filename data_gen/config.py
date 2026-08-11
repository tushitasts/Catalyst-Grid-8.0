"""
========================================================
 Flipkart Grid 8.0 — Synthetic Data Generation Config
========================================================
ALL distribution parameters live here.
Tweak this file to adjust skewness, ranges, fraud mix, etc.
No other file needs to be edited for parameter changes.
"""

# ─── GLOBAL SETTINGS ──────────────────────────────────────────────────────────
RANDOM_SEED = 42

N_USERS        = 6_000
N_PRODUCTS     = 1_000
N_SELLERS      = 500
# N_ORDERS is the PLANNING target fed to gen_users.py for order count scaling.
# Due to date-range trimming (users with short account age have fewer actual orders),
# the actual row count in orders.csv will be ~75-85% of this value.
# Set to 30,000 so the real output lands near 25,000.
N_ORDERS       = 33_000   # planning target (actual output ~24-26k after date trimming)
N_RETURNS      = 10_000   # exact target size for returns.csv

FRAUD_TARGET_RATIO = 0.40     # ~40% of returns.csv labeled fraud

DATE_START = "2022-01-01"     # earliest order date in the simulation
DATE_END   = "2025-07-01"     # latest order date in the simulation

import os
ROOT_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
OUTPUT_DIR = os.path.join(ROOT_DIR, "output")


# ─── USER BASE FEATURES ───────────────────────────────────────────────────────
# Only base features that CANNOT be derived from other tables.
# Derived features (return_to_order_ratio, days_since_last_return, etc.)
# will be computed in the feature-engineering pipeline.

USER = dict(
    # Account age in days. Right-skewed lognormal: most 90–730d, tail to 2500d.
    account_age_lognormal       = dict(mu=6.2, sigma=0.75, min_val=30, max_val=2500),

    email_verified_prob         = 0.95,
    # New accounts (<120 days) have lower verification rates
    email_verified_prob_new     = 0.80,

    # Shared device: more common for newer or suspicious accounts
    shared_device_prob          = 0.06,   # general population
    shared_device_prob_new_acct = 0.16,   # new accounts (<120 days)
    new_account_threshold_days  = 120,
)


# ─── BUYER PROFILE (internal — not saved, drives order generation) ───────────
# Each user gets a hidden price-segment + category-affinity profile.

BUYER_PRICE_SEGMENTS = {
    # key: (probability, (min_price_INR, max_price_INR))
    "budget":   dict(prob=0.40, price_range=(80,    4_000)),
    "mid":      dict(prob=0.40, price_range=(1_000, 18_000)),
    "premium":  dict(prob=0.20, price_range=(8_000, 200_000)),
}

BUYER_CATEGORY_AFFINITY = {
    # how many categories a user regularly purchases from
    "focused":  dict(prob=0.30, n_cats_range=(1, 2)),
    "moderate": dict(prob=0.50, n_cats_range=(3, 5)),
    "explorer": dict(prob=0.20, n_cats_range=(6, 10)),
}

# Probability a user deviates from their preferred categories in a given order
CATEGORY_EXPLORATION_PROB = 0.18


# ─── USER BEHAVIOR / FRAUD TYPE ───────────────────────────────────────────────
# Drives how many orders a user places and what fraction result in returns.
# Fraud users naturally have higher return rates.
#
# Math check (rough):
#   Legit:  3600 users x 4.0 orders x 0.22 return rate x 0.88 returnable = ~2788 returns
#   Fraud:  2400 users x 7.5 orders x 0.72 return rate x 0.88 returnable = ~11405 returns
#   Total before sampling = ~14,193  ->  sample down to N_RETURNS = 10,000

USER_BEHAVIOR_TYPES = {
    # key: (proportion of users, avg_orders_lognormal_mean, return_rate)
    "legitimate":             dict(prob=0.60, avg_orders=4.0,  return_rate=0.22),
    "wardrober":              dict(prob=0.10, avg_orders=7.5,  return_rate=0.75),
    "serial_returner":        dict(prob=0.10, avg_orders=9.0,  return_rate=0.82),
    "empty_box":              dict(prob=0.08, avg_orders=7.0,  return_rate=0.72),
    "item_swap":              dict(prob=0.07, avg_orders=6.0,  return_rate=0.68),
    "seller_buyer_collusion": dict(prob=0.05, avg_orders=8.0,  return_rate=0.72),
}

# Spread around avg_orders (lognormal sigma)
ORDER_COUNT_SIGMA = 0.55


# ─── PRODUCT FEATURES ─────────────────────────────────────────────────────────
CATEGORIES = [
    "Electronics", "Clothing",   "Books",     "Appliances", "Footwear",
    "Beauty",      "Furniture",  "Sports",    "Toys",       "Grocery",
]
# Global sampling weights (must sum to 1)
CATEGORY_WEIGHTS = [0.18, 0.25, 0.07, 0.10, 0.12, 0.08, 0.05, 0.08, 0.04, 0.03]

# Product base price (log-normal, median ≈ exp(mu) INR)
CATEGORY_PRICE_PARAMS = {
    "Electronics":  dict(mu=9.0,  sigma=1.0),   # median ~₹8,100
    "Clothing":     dict(mu=6.7,  sigma=0.9),   # median ~₹812
    "Books":        dict(mu=5.5,  sigma=0.7),   # median ~₹245
    "Appliances":   dict(mu=9.5,  sigma=0.9),   # median ~₹13,360
    "Footwear":     dict(mu=7.1,  sigma=0.8),   # median ~₹1,212
    "Beauty":       dict(mu=6.3,  sigma=0.8),   # median ~₹545
    "Furniture":    dict(mu=9.8,  sigma=0.8),   # median ~₹18,034
    "Sports":       dict(mu=7.5,  sigma=0.9),   # median ~₹1,808
    "Toys":         dict(mu=6.8,  sigma=0.8),   # median ~₹898
    "Grocery":      dict(mu=5.0,  sigma=0.6),   # median ~₹148
}

PRODUCT = dict(
    # Review count: heavily right-skewed (few blockbuster products, many niche)
    review_count_lognormal = dict(mu=3.8, sigma=1.6, min_val=0, max_val=50_000),
    # SKU = stock keeping unit (units in stock), lognormal
    sku_lognormal          = dict(mu=4.5, sigma=1.2, min_val=0, max_val=5_000),
)

# Probability a product is non-returnable, by category
NON_RETURNABLE_PROB = {
    "Electronics": 0.05,  "Clothing":   0.03,  "Books":    0.25,
    "Appliances":  0.05,  "Footwear":   0.04,  "Beauty":   0.40,
    "Furniture":   0.10,  "Sports":     0.08,  "Toys":     0.08,
    "Grocery":     0.95,
}

# Return window in days per category (0 = no return allowed)
# Edit these to test different policy scenarios.
RETURN_WINDOW_DAYS = {
    "Electronics": 10,   "Clothing":   10,  "Books":    7,
    "Appliances":  10,   "Footwear":   10,  "Beauty":   7,
    "Furniture":   7,    "Sports":     7,   "Toys":     7,
    "Grocery":     0,
}

# Base defect rate: fraction of products in this category that have real defects.
# Used as a product-level feature (same value for all products in the category).
CATEGORY_DEFECT_RATE = {
    "Electronics": 0.04,  "Clothing":  0.02,  "Books":    0.01,
    "Appliances":  0.06,  "Footwear":  0.03,  "Beauty":   0.02,
    "Furniture":   0.05,  "Sports":    0.03,  "Toys":     0.04,
    "Grocery":     0.08,
}


# ─── SELLER FEATURES ──────────────────────────────────────────────────────────
SELLER = dict(
    # Seller account age: right-skewed, most 6 months to 3 years
    age_days_lognormal   = dict(mu=6.4, sigma=0.85, min_val=30, max_val=2500),

    # Rating (scale 0–5): new sellers are rated lower → consistent profiles
    rating_veteran_beta  = dict(a=7, b=2),    # veteran (>180d): skewed to 3.5–5
    rating_new_beta      = dict(a=2, b=3),    # new (≤180d):     skewed to 1.5–3.5
    new_seller_threshold = 180,               # days to be considered "new"
    rating_min           = 1.0,
    rating_max           = 5.0,

    # Seller return rate: fraction of their orders that come back as returns.
    # Beta(1.5, 12): most sellers 5–15%, rare outliers up to 40%
    return_rate_beta     = dict(a=1.5, b=12),

    # Seller customer frequency: fraction of buyers who are repeat customers.
    customer_freq_beta   = dict(a=2, b=5),
)

# Seller popularity for order assignment.
# alpha closer to 1 = more skewed (one seller dominates)
# alpha = 2.0 = moderately skewed (realistic marketplace)
SELLER_POPULARITY_ALPHA = 2.0


# ─── SELLER TYPE CONFIG ───────────────────────────────────────────────────────
# Each seller is assigned a type that determines which categories they stock.
# This makes orders realistic: a furniture seller won't sell electronics.
#
# 'categories'       : list of categories this seller type carries
# 'primary_cat'      : the 1-2 core categories (heavy weight in product assignment)
# 'secondary_cat'    : optional related categories (lighter weight)
# 'prob'             : fraction of sellers of this type
#
# Design rationale:
#   Fashion sellers (clothing/footwear/beauty) are the most common.
#   General marketplace sellers handle broad inventory.
#   Specialists (electronics-only, clothing-only) are focused.

SELLER_TYPES = {
    # ── COMBO / MULTI-CATEGORY SELLERS ────────────────────────────────────────
    "fashion_lifestyle": dict(
        # Clothing + Footwear + Beauty  (Ajio / Myntra-style sellers)
        prob            = 0.14,
        categories      = ["Clothing", "Footwear", "Beauty"],
        primary_cats    = ["Clothing", "Footwear"],
        secondary_cats  = ["Beauty"],
        primary_weight  = 0.85,
    ),
    "electronics_tech": dict(
        # Electronics + Appliances  (Croma / Vijay Sales-style)
        prob            = 0.10,
        categories      = ["Electronics", "Appliances"],
        primary_cats    = ["Electronics"],
        secondary_cats  = ["Appliances"],
        primary_weight  = 0.78,
    ),
    "furniture_home": dict(
        # Furniture + Appliances  (Pepperfry / Urban Ladder-style)
        prob            = 0.06,
        categories      = ["Furniture", "Appliances"],
        primary_cats    = ["Furniture"],
        secondary_cats  = ["Appliances"],
        primary_weight  = 0.82,
    ),
    "sports_outdoors": dict(
        # Sports + Toys  (Decathlon-style)
        prob            = 0.07,
        categories      = ["Sports", "Toys"],
        primary_cats    = ["Sports"],
        secondary_cats  = ["Toys"],
        primary_weight  = 0.80,
    ),
    "grocery_health": dict(
        # Grocery + Beauty  (FMCG / BigBasket-style)
        prob            = 0.05,
        categories      = ["Grocery", "Beauty"],
        primary_cats    = ["Grocery"],
        secondary_cats  = ["Beauty"],
        primary_weight  = 0.75,
    ),
    "general_marketplace": dict(
        # Large multi-category seller — handles almost everything
        prob            = 0.18,
        categories      = ["Electronics", "Clothing", "Footwear", "Appliances",
                           "Sports", "Books", "Beauty", "Toys"],
        primary_cats    = ["Electronics", "Clothing", "Footwear", "Appliances",
                           "Sports", "Books", "Beauty", "Toys"],
        secondary_cats  = [],
        primary_weight  = 1.00,
    ),

    # ── SINGLE-CATEGORY SPECIALISTS (one per each of the 10 categories) ───────
    "electronics_specialist": dict(
        prob            = 0.05,
        categories      = ["Electronics"],
        primary_cats    = ["Electronics"],
        secondary_cats  = [],
        primary_weight  = 1.00,
    ),
    "clothing_specialist": dict(
        # Ethnic wear boutique / fast fashion brand
        prob            = 0.05,
        categories      = ["Clothing"],
        primary_cats    = ["Clothing"],
        secondary_cats  = [],
        primary_weight  = 1.00,
    ),
    "footwear_specialist": dict(
        # Nike / Bata / Metro store-type seller
        prob            = 0.04,
        categories      = ["Footwear"],
        primary_cats    = ["Footwear"],
        secondary_cats  = [],
        primary_weight  = 1.00,
    ),
    "appliances_specialist": dict(
        # White goods / kitchen appliances dealer
        prob            = 0.04,
        categories      = ["Appliances"],
        primary_cats    = ["Appliances"],
        secondary_cats  = [],
        primary_weight  = 1.00,
    ),
    "beauty_specialist": dict(
        # Cosmetics / skincare brand seller
        prob            = 0.04,
        categories      = ["Beauty"],
        primary_cats    = ["Beauty"],
        secondary_cats  = [],
        primary_weight  = 1.00,
    ),
    "furniture_specialist": dict(
        # Custom furniture / niche brand
        prob            = 0.03,
        categories      = ["Furniture"],
        primary_cats    = ["Furniture"],
        secondary_cats  = [],
        primary_weight  = 1.00,
    ),
    "books_specialist": dict(
        # Publisher outlet / second-hand bookstore
        prob            = 0.05,
        categories      = ["Books"],
        primary_cats    = ["Books"],
        secondary_cats  = [],
        primary_weight  = 1.00,
    ),
    "sports_specialist": dict(
        # Fitness gear / cricket / cycling focused
        prob            = 0.03,
        categories      = ["Sports"],
        primary_cats    = ["Sports"],
        secondary_cats  = [],
        primary_weight  = 1.00,
    ),
    "toys_specialist": dict(
        # Dedicated toy shop / kids brand
        prob            = 0.03,
        categories      = ["Toys"],
        primary_cats    = ["Toys"],
        secondary_cats  = [],
        primary_weight  = 1.00,
    ),
    "grocery_specialist": dict(
        # Local grocery / organic food seller
        prob            = 0.04,
        categories      = ["Grocery"],
        primary_cats    = ["Grocery"],
        secondary_cats  = [],
        primary_weight  = 1.00,
    ),
    # Probability check:
    # Combo:       0.14+0.10+0.06+0.07+0.05+0.18 = 0.60
    # Specialists: 0.05+0.05+0.04+0.04+0.04+0.03+0.05+0.03+0.03+0.04 = 0.40
    # Total = 1.00
}


# ─── ORDER FEATURES ───────────────────────────────────────────────────────────
ORDER = dict(
    # Fraction of orders that are prepaid (vs cash on delivery)
    prepaid_prob     = 0.70,

    # Discount percentage: Beta distribution × max_discount
    # Beta(1.2, 6): most orders have 0–20% discount, occasional spike
    discount_beta    = dict(a=1.2, b=6),
    discount_max_pct = 80,

    # Delivery lag (calendar days from order placement to delivery)
    delivery_days_min = 2,
    delivery_days_max = 7,

    # Inter-order gap for a user (days between consecutive orders, lognormal)
    gap_days_lognormal = dict(mu=3.3, sigma=0.75, min_val=2, max_val=180),
)


# ─── RETURN FEATURES (per fraud sub-type) ─────────────────────────────────────
# For each sub-type, we define:
#   reason_probs     – distribution over reason_category
#   return_type_probs– distribution over return/replace
#   image_upload_prob– probability user uploaded an image
#   days_beta        – Beta params for (days_after_delivery / return_window)

RETURN_SUB_TYPES = {

    # ── LEGITIMATE ────────────────────────────────────────────────────────────
    "legitimate_defect": dict(
        # Genuine product defect: images uploaded, reason matches product type
        reason_probs      = dict(damaged=0.50, quality_issue=0.30, wrong_item=0.20),
        return_type_probs = dict(refund=0.55, replace=0.45),
        image_upload_prob = 0.78,
        days_beta         = dict(a=2.5, b=3.0),  # tends to wait a few days
    ),

    "changed_mind": dict(
        # Buyer's remorse: no image needed, changed_mind reason
        reason_probs      = dict(changed_mind=0.60, not_fit=0.30, quality_issue=0.10),
        return_type_probs = dict(refund=0.85, replace=0.15),
        image_upload_prob = 0.38,
        days_beta         = dict(a=2.0, b=3.5),
    ),

    # ── FRAUD ─────────────────────────────────────────────────────────────────
    "wardrober": dict(
        # Buys high-value item for event, returns it used
        # → changed_mind / not_fit reason, returns quickly after event
        reason_probs      = dict(changed_mind=0.55, not_fit=0.35, quality_issue=0.10),
        return_type_probs = dict(refund=0.95, replace=0.05),
        image_upload_prob = 0.38,
        # Wardrobers return near the END of the window (used for an event, then return).
        # Beta(5, 1.8): mean = 5/6.8 ≈ 0.74 of window → day 5 of 7, day 10 of 14, etc.
        days_beta         = dict(a=5.0, b=1.8),
    ),

    "serial_returner": dict(
        # High-frequency returner; mixed reasons
        reason_probs      = dict(not_fit=0.30, quality_issue=0.30,
                                 changed_mind=0.25, damaged=0.15),
        return_type_probs = dict(refund=0.80, replace=0.20),
        image_upload_prob = 0.48,
        days_beta         = dict(a=2.0, b=3.5),
    ),

    "empty_box": dict(
        # Claims package arrived empty; very vague reason, returns immediately
        reason_probs      = dict(damaged=0.70, wrong_item=0.30),
        return_type_probs = dict(refund=0.92, replace=0.08),
        image_upload_prob = 0.22,  # rarely provides real image
        days_beta         = dict(a=1.2, b=6.5),  # claims within 1-2 days
    ),

    "item_swap": dict(
        # Swaps real product for counterfeit / cheaper item before returning
        reason_probs      = dict(damaged=0.50, wrong_item=0.30, quality_issue=0.20),
        return_type_probs = dict(refund=0.85, replace=0.15),
        image_upload_prob = 0.22,  # image wouldn't show the swap
        days_beta         = dict(a=2.0, b=4.5),
    ),

    "seller_buyer_collusion": dict(
        # User and seller coordinate to abuse return policy
        reason_probs      = dict(damaged=0.40, quality_issue=0.30,
                                 wrong_item=0.20, not_fit=0.10),
        return_type_probs = dict(refund=0.75, replace=0.25),
        image_upload_prob = 0.42,
        days_beta         = dict(a=2.0, b=4.5),
    ),
}

# Mapping from user behavior_type → sub_type(s) used for that type's returns
# Legitimate users split between 'legitimate_defect' and 'changed_mind'
LEGIT_SUBTYPE_SPLIT = dict(legitimate_defect=0.60, changed_mind=0.40)

# Fraud users: behavior_type maps directly to a single sub_type
FRAUD_TO_SUBTYPE = {
    "wardrober":              "wardrober",
    "serial_returner":        "serial_returner",
    "empty_box":              "empty_box",
    "item_swap":              "item_swap",
    "seller_buyer_collusion": "seller_buyer_collusion",
}


# ─── FRAUD SCORING FORMULA ────────────────────────────────────────────────────
# Fraud score = weighted sum of normalized signals.
# Higher score → more likely to be labeled fraud.
# After scoring, sigmoid + Gaussian noise is applied.
# Threshold is calibrated so FRAUD_TARGET_RATIO of labels = 'fraud'.
#
# To make the model harder/easier:
#   - Increase FRAUD_SCORE_NOISE_SCALE → more label noise → harder to learn
#   - Adjust weights to emphasize different signals

FRAUD_SCORE_WEIGHTS = dict(
    return_to_order_ratio   = 2.5,  # high RTOR  → very suspicious
    high_value_ratio        = 1.8,  # order_value >> user_avg → suspicious
    new_account             = 1.2,  # newer account → more suspicious
    no_image                = 1.0,  # no image → suspicious
    seller_repeat_ratio     = 1.5,  # same seller repeated in returns
    high_return_freq        = 2.0,  # many returns in last 30d
    early_return            = 0.8,  # returned same/next day after delivery
    unverified_account      = 0.7,  # unverified phone or email
    changed_mind_reason     = 0.5,  # reason = changed_mind (not inherently bad but signal)
    shared_device           = 0.7,  # shared device flag
    high_discount           = 0.4,  # exploiting heavy discounts (>30%)
)

# Gaussian noise added to raw fraud score before thresholding.
# Increases label ambiguity → realistic noisy labels.
FRAUD_SCORE_NOISE_SCALE = 0.90

# Normalization denominators for each signal (to bring to [0,1]):
FRAUD_SCORE_NORMALIZATION = dict(
    rtor_saturation         = 0.50,   # RTOR ≥ 50% → max score
    value_ratio_saturation  = 2.0,    # val_ratio ≥ 3× avg → max
    age_saturation_days     = 730,    # account ≥ 2 years → age score = 0
    freq_saturation_returns = 5,      # ≥5 returns in 30d → max score
    early_return_days       = 2,      # returned within 2 days → max
    high_discount_floor     = 0.30,   # only discounts > 30% score
)
