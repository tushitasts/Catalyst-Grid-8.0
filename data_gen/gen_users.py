"""
Generate user profiles (users.csv) + internal buyer profiles.

BASE FEATURES saved to users.csv:
  - account_age_days
  - email_verified
  - shared_device_flag

INTERNAL ONLY (returned as profiles_df, NOT saved):
  - behavior_type       (legitimate / fraud archetype)
  - return_rate         (per-behavior return probability)
  - price_segment       (budget / mid / premium)
  - category_prefs      (list of preferred categories)
  - n_orders            (total orders this user will place)

The profiles_df is passed to gen_orders.py to drive order generation.
Derived user features (return_to_order_ratio, days_since_last_return, etc.)
are computed in the feature-engineering pipeline.
"""

import numpy as np
import pandas as pd

from config import (
    RANDOM_SEED, N_USERS, N_ORDERS,
    USER, BUYER_PRICE_SEGMENTS, BUYER_CATEGORY_AFFINITY,
    USER_BEHAVIOR_TYPES, ORDER_COUNT_SIGMA,
    CATEGORIES, CATEGORY_WEIGHTS, CATEGORY_EXPLORATION_PROB,
)


# ── Internal helpers ───────────────────────────────────────────────────────────

def _assign_behavior_types(rng: np.random.Generator, n: int) -> np.ndarray:
    types = list(USER_BEHAVIOR_TYPES.keys())
    probs = np.array([USER_BEHAVIOR_TYPES[t]["prob"] for t in types], dtype=float)
    probs /= probs.sum()
    return rng.choice(types, size=n, p=probs)


def _assign_buyer_profiles(rng: np.random.Generator, n: int):
    """Return (price_segments, category_pref_lists)."""
    # Price segment
    segs  = list(BUYER_PRICE_SEGMENTS.keys())
    sprob = np.array([BUYER_PRICE_SEGMENTS[s]["prob"] for s in segs], dtype=float)
    sprob /= sprob.sum()
    price_segments = rng.choice(segs, size=n, p=sprob)

    # Category affinity type
    aff_types  = list(BUYER_CATEGORY_AFFINITY.keys())
    aff_probs  = np.array([BUYER_CATEGORY_AFFINITY[t]["prob"] for t in aff_types], dtype=float)
    aff_probs  /= aff_probs.sum()
    aff_type   = rng.choice(aff_types, size=n, p=aff_probs)

    # Build category preference list for each user
    cat_weights = np.array(CATEGORY_WEIGHTS, dtype=float)
    cat_weights /= cat_weights.sum()

    category_prefs = []
    for i in range(n):
        lo, hi = BUYER_CATEGORY_AFFINITY[aff_type[i]]["n_cats_range"]
        n_cats = int(rng.integers(lo, hi + 1))
        # Sample without replacement, weighted by global category distribution
        chosen = rng.choice(CATEGORIES, size=n_cats, replace=False, p=cat_weights)
        category_prefs.append(list(chosen))

    return price_segments, category_prefs


def _compute_order_counts(rng: np.random.Generator, behavior_types: np.ndarray) -> np.ndarray:
    """
    Sample order counts per user from a log-normal centred on each behavior
    type's avg_orders. Then rescale so the sum hits N_ORDERS_TARGET.
    """
    n = len(behavior_types)
    raw_counts = np.zeros(n, dtype=float)
    for i, bt in enumerate(behavior_types):
        avg = USER_BEHAVIOR_TYPES[bt]["avg_orders"]
        mu  = np.log(avg)
        raw_counts[i] = rng.lognormal(mu, ORDER_COUNT_SIGMA)

    raw_counts = np.clip(raw_counts, 1, 250)

    # Scale to target total
    scale = N_ORDERS / raw_counts.sum()
    scaled = np.round(raw_counts * scale).astype(int)
    scaled = np.maximum(scaled, 1)   # every user has at least 1 order

    # Correct rounding drift
    diff = N_ORDERS - scaled.sum()
    if diff != 0:
        # Add/remove from heaviest users
        idx = np.argsort(-scaled)[:abs(diff)]
        scaled[idx] += np.sign(diff)

    return scaled


# ── Public function ────────────────────────────────────────────────────────────

def generate_users(seed: int = RANDOM_SEED):
    """
    Returns
    -------
    users_df    : pd.DataFrame — base features only (saved to users.csv)
    profiles_df : pd.DataFrame — internal buyer/behavior profiles (NOT saved)
    """
    rng = np.random.default_rng(seed + 2)

    # ── Account age ────────────────────────────────────────────────────────────
    p = USER["account_age_lognormal"]
    account_age = rng.lognormal(p["mu"], p["sigma"], size=N_USERS)
    account_age = np.clip(account_age, p["min_val"], p["max_val"]).astype(int)

    new_acct = account_age < USER["new_account_threshold_days"]

    # ── Verification flags (consistent with account age) ──────────────────────

    email_verified = np.where(
        new_acct,
        rng.random(N_USERS) < USER["email_verified_prob_new"],
        rng.random(N_USERS) < USER["email_verified_prob"],
    ).astype(int)

    # ── Shared device (more likely for new or unverified accounts) ────────────
    # Unverified + new account: highest shared-device probability
    shared_prob = np.where(
        new_acct,
        USER["shared_device_prob_new_acct"],
        USER["shared_device_prob"],
    )
    # Boost slightly if also unverified
    shared_prob = np.where(
        (email_verified == 0),
        np.minimum(shared_prob * 1.5, 0.35),
        shared_prob,
    )
    shared_device_flag = (rng.random(N_USERS) < shared_prob).astype(int)

    # ── Behavior types + buyer profiles ───────────────────────────────────────
    behavior_types = _assign_behavior_types(rng, N_USERS)
    price_segments, category_prefs = _assign_buyer_profiles(rng, N_USERS)

    # ── Order counts ──────────────────────────────────────────────────────────
    n_orders_per_user = _compute_order_counts(rng, behavior_types)

    # ── Assemble outputs ──────────────────────────────────────────────────────
    user_ids = [f"USR_{i + 1:06d}" for i in range(N_USERS)]

    users_df = pd.DataFrame(
        {
            "user_id":            user_ids,
            "account_age_days":   account_age,
            "email_verified":     email_verified,
            "shared_device_flag": shared_device_flag,
        }
    )

    profiles_df = pd.DataFrame(
        {
            "user_id":        user_ids,
            "behavior_type":  behavior_types,
            "return_rate":    [USER_BEHAVIOR_TYPES[bt]["return_rate"] for bt in behavior_types],
            "price_segment":  price_segments,
            "category_prefs": category_prefs,  # list-typed column, kept in memory only
            "n_orders":       n_orders_per_user,
        }
    )

    return users_df, profiles_df


if __name__ == "__main__":
    users, profiles = generate_users()
    print(f"Users: {len(users)}")
    print("\nBehavior type distribution:")
    print(profiles["behavior_type"].value_counts(normalize=True).round(3))
    print("\nPrice segment distribution:")
    print(profiles["price_segment"].value_counts(normalize=True).round(3))
    print("\nAccount age (days) stats:")
    print(users["account_age_days"].describe().round(0))
    print(f"\nTotal planned orders: {profiles['n_orders'].sum():,}")
