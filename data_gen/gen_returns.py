"""
Generate returns.csv — the return-request dataset used to train the LGBM model.

OVERVIEW
--------
1. Pull all is_returned=1 orders from orders_df.
2. For each return, determine the fraud sub-type based on the user's behavior_type.
3. Generate return-specific BASE FEATURES per sub-type:
     - reason_category
     - return_type
     - image_uploaded
     - return_request_date  (controls days_since_delivery, days_left_to_return)
4. Track user state INCREMENTALLY (chronological order per user):
     - total_orders_at_time   = orders placed by this user up to the return date
     - total_returns_at_time  = how many returns this user has made so far
     (These are "at-time" snapshots; would require temporal joins otherwise.)
5. Compute the FRAUD SCORING FORMULA on inline derived signals:
     - return_to_order_ratio_at_time
     - value_ratio (order_value / user_avg_order_value_at_time)
     - account_age_norm, no_image, seller_repeat_ratio, return_freq_last_30d, etc.
6. Add Gaussian noise → sigmoid → threshold → binary label (fraud/legitimate).
   Threshold calibrated so FRAUD_TARGET_RATIO ≈ 40% of rows = 'fraud'.

COLUMNS SAVED TO returns.csv
-----------------------------
Return-specific (base features):
  return_id, order_id, user_id, product_id, seller_id
  reason_category, return_type, image_uploaded
  return_request_date, within_return_window, days_left_to_return

At-time-of-return user state (stored to avoid temporal joins during training):
  total_orders_at_time, total_returns_at_time

Metadata (for analysis / model debugging, NOT used as ML features directly):
  fraud_type   – the sub-type archetype assigned during generation
  label        – 'fraud' or 'legitimate'  (the training target)
"""

import numpy as np
import pandas as pd
from datetime import datetime, timedelta

from config import (
    RANDOM_SEED, N_RETURNS, FRAUD_TARGET_RATIO,
    RETURN_SUB_TYPES, LEGIT_SUBTYPE_SPLIT, FRAUD_TO_SUBTYPE,
    FRAUD_SCORE_WEIGHTS, FRAUD_SCORE_NOISE_SCALE, FRAUD_SCORE_NORMALIZATION,
    RETURN_WINDOW_DAYS,
)


# ── Helpers ────────────────────────────────────────────────────────────────────

def _sample_from_probs(rng, prob_dict: dict) -> str:
    """Sample a key proportionally from a {key: prob} dict."""
    keys  = list(prob_dict.keys())
    probs = np.array(list(prob_dict.values()), dtype=float)
    probs /= probs.sum()
    return str(rng.choice(keys, p=probs))


def _compute_fraud_score(
    rtor_at_time: float,
    value_ratio: float,
    account_age_days: int,
    image_uploaded: int,
    seller_repeat_ratio: float,
    returns_last_30d: int,
    days_since_delivery: int,
    email_verified: int,
    reason_category: str,
    shared_device_flag: int,
    discount_pct: float,
    weights: dict,
    norm: dict,
) -> float:
    """
    Compute continuous fraud risk score (higher → more likely fraud).
    Each component is normalised to [0, 1] before weighting.

    Adjust FRAUD_SCORE_WEIGHTS in config.py to change signal importance.
    """
    w = weights
    n = norm

    # 1. Return-to-order ratio at time of return
    rtor_score = min(rtor_at_time / n["rtor_saturation"], 1.0)

    # 2. High-value return (relative to user's running avg order value)
    val_score  = min(max(0.0, (value_ratio - 1.0) / n["value_ratio_saturation"]), 1.0)

    # 3. New account (inversely proportional to age)
    age_score  = max(0.0, 1.0 - account_age_days / n["age_saturation_days"])

    # 4. No image uploaded
    no_img_score = float(1 - image_uploaded)

    # 5. Seller repeat ratio (same seller appears many times in returns)
    seller_score = min(seller_repeat_ratio, 1.0)

    # 6. High return frequency in last 30 days
    freq_score = min(returns_last_30d / n["freq_saturation_returns"], 1.0)

    # 7. Very early return (within 2 days of delivery → suspicious)
    if days_since_delivery <= n["early_return_days"]:
        early_score = max(0.0, 1.0 - days_since_delivery / n["early_return_days"])
    else:
        early_score = 0.0

    # 8. Unverified account (additive)
    unverified = 1.0 if not email_verified else 0.0

    # 9. "changed_mind" reason (soft signal — not inherently fraud, but correlated)
    reason_score = 1.0 if reason_category == "changed_mind" else 0.0

    # 10. Shared device flag
    device_score = float(shared_device_flag)

    # 11. Exploiting heavy discount
    disc_score = max(0.0, (discount_pct / 100.0) - n["high_discount_floor"])

    score = (
        w["return_to_order_ratio"] * rtor_score
        + w["high_value_ratio"]    * val_score
        + w["new_account"]         * age_score
        + w["no_image"]            * no_img_score
        + w["seller_repeat_ratio"] * seller_score
        + w["high_return_freq"]    * freq_score
        + w["early_return"]        * early_score
        + w["unverified_account"]  * unverified
        + w["changed_mind_reason"] * reason_score
        + w["shared_device"]       * device_score
        + w["high_discount"]       * disc_score
    )
    return score


# ── Main function ──────────────────────────────────────────────────────────────

def generate_returns(
    orders_df:   pd.DataFrame,
    users_df:    pd.DataFrame,
    products_df: pd.DataFrame,
    sellers_df:  pd.DataFrame,
    seed: int = RANDOM_SEED,
) -> pd.DataFrame:
    """
    Returns
    -------
    returns_df : pd.DataFrame  — the final returns.csv content
    """
    rng = np.random.default_rng(seed + 4)

    # ── Step 1: Pull returned orders ───────────────────────────────────────────
    returned = orders_df[orders_df["is_returned"] == 1].copy()

    if len(returned) > N_RETURNS:
        returned = returned.sample(n=N_RETURNS, random_state=seed).copy()
    elif len(returned) < N_RETURNS:
        print(
            f"[WARN] Only {len(returned)} returned orders found (target={N_RETURNS}). "
            "Consider increasing return rates in config.py."
        )

    # Sort by (user_id, order_date) for chronological state tracking
    returned["_order_date_dt"]    = pd.to_datetime(returned["order_date"])
    returned["_delivery_date_dt"] = pd.to_datetime(returned["delivery_date"])
    returned = returned.sort_values(["user_id", "_order_date_dt"]).reset_index(drop=True)

    # ── Step 2: Join reference tables ─────────────────────────────────────────
    returned = returned.merge(
        users_df[["user_id", "account_age_days",
                  "email_verified", "shared_device_flag"]],
        on="user_id", how="left",
    )
    returned = returned.merge(
        products_df[["product_id", "category", "return_window_days"]],
        on="product_id", how="left",
    )
    returned = returned.merge(
        sellers_df[["seller_id", "seller_rating"]],
        on="seller_id", how="left",
    )

    # ── Step 3: Pre-compute per-user order history for "at time" features ─────
    # For each user: cumulative order count and running sum of order_value
    # keyed by (user_id, order_date) so we can look up "orders placed before
    # this return date".
    user_orders_sorted = (
        orders_df[["user_id", "order_date", "order_value"]]
        .copy()
        .assign(order_date=lambda x: pd.to_datetime(x["order_date"]))
        .sort_values(["user_id", "order_date"])
    )

    # Per-user cumulative count and value sum
    user_orders_sorted["_cumcount"] = user_orders_sorted.groupby("user_id").cumcount() + 1
    user_orders_sorted["_cumvalue"] = user_orders_sorted.groupby("user_id")["order_value"].cumsum()

    def _orders_at_date(user_id: str, date: datetime):
        """Returns (count, value_sum) of orders placed by user up to `date`."""
        mask = (
            (user_orders_sorted["user_id"] == user_id)
            & (user_orders_sorted["order_date"] <= date)
        )
        sub = user_orders_sorted[mask]
        if len(sub) == 0:
            return 1, returned.loc[returned["user_id"] == user_id, "order_value"].mean() or 1.0
        last = sub.iloc[-1]
        return int(last["_cumcount"]), float(last["_cumvalue"])

    # ── Step 4: State tracking per user ───────────────────────────────────────
    user_return_count   = {}   # {user_id: int}
    user_return_dates   = {}   # {user_id: [datetime, ...]}
    user_seller_returns = {}   # {user_id: {seller_id: int}}

    # ── Step 5: Iterate row by row (chronological per user) ────────────────────
    rows = []
    raw_scores = []

    for _, ro in returned.iterrows():
        uid = ro["user_id"]
        sid = ro["seller_id"]
        behavior_type = ro.get("_behavior_type", "legitimate")

        # Initialise user state on first encounter
        if uid not in user_return_count:
            user_return_count[uid]   = 0
            user_return_dates[uid]   = []
            user_seller_returns[uid] = {}

        # ── Determine sub-type ─────────────────────────────────────────────────
        if behavior_type == "legitimate":
            sub_type = _sample_from_probs(rng, LEGIT_SUBTYPE_SPLIT)
        else:
            sub_type = FRAUD_TO_SUBTYPE.get(behavior_type, "legitimate_defect")

        cfg = RETURN_SUB_TYPES[sub_type]

        # ── Return window & request date ──────────────────────────────────────
        delivery_dt   = ro["_delivery_date_dt"].to_pydatetime()
        return_window = int(ro["return_window_days"]) if ro["return_window_days"] > 0 else 7

        # Sample days_after_delivery as fraction of return window (Beta → [0, 1])
        frac = float(rng.beta(cfg["days_beta"]["a"], cfg["days_beta"]["b"]))
        # Fraud types occasionally go slightly past the window (1–3 extra days)
        if behavior_type != "legitimate":
            overshoot = int(rng.integers(0, 4)) if rng.random() < 0.12 else 0
        else:
            overshoot = 0

        days_after_delivery  = max(0, round(frac * return_window) + overshoot)
        return_request_date  = delivery_dt + timedelta(days=days_after_delivery)
        within_return_window = int(days_after_delivery <= return_window)
        days_left_to_return  = max(0, return_window - days_after_delivery)

        # ── Return reason ─────────────────────────────────────────────────────
        reason_category = _sample_from_probs(rng, cfg["reason_probs"])

        # ── Return type ───────────────────────────────────────────────────────
        return_type = _sample_from_probs(rng, cfg["return_type_probs"])

        # ── Image uploaded ────────────────────────────────────────────────────
        image_uploaded = int(rng.random() < cfg["image_upload_prob"])

        # ── At-time user state ────────────────────────────────────────────────
        # Total orders placed by this user on or before the return request date
        orders_at_time, value_sum_at_time = _orders_at_date(uid, return_request_date)
        total_orders_at_time  = orders_at_time
        total_returns_at_time = user_return_count[uid]   # before this return

        rtor_at_time = total_returns_at_time / max(total_orders_at_time, 1)
        user_avg_ov  = value_sum_at_time / max(total_orders_at_time, 1)
        value_ratio  = ro["order_value"] / max(user_avg_ov, 1.0)

        # Returns in the last 30 days (before this return request)
        returns_last_30d = sum(
            1 for d in user_return_dates[uid]
            if (return_request_date - d).days <= 30
        )

        # Seller repeat ratio (fraction of user's past returns involving this seller)
        past_returns_total = max(user_return_count[uid], 1)
        seller_ret_count   = user_seller_returns[uid].get(sid, 0)
        seller_repeat_ratio = seller_ret_count / past_returns_total

        # ── Compute fraud score ───────────────────────────────────────────────
        raw_score = _compute_fraud_score(
            rtor_at_time       = rtor_at_time,
            value_ratio        = value_ratio,
            account_age_days   = int(ro["account_age_days"]),
            image_uploaded     = image_uploaded,
            seller_repeat_ratio= seller_repeat_ratio,
            returns_last_30d   = returns_last_30d,
            days_since_delivery= days_after_delivery,
            email_verified     = int(ro["email_verified"]),
            reason_category    = reason_category,
            shared_device_flag = int(ro["shared_device_flag"]),
            discount_pct       = float(ro["discount_pct"]),
            weights            = FRAUD_SCORE_WEIGHTS,
            norm               = FRAUD_SCORE_NORMALIZATION,
        )
        raw_scores.append(raw_score)

        # ── Update user state (AFTER computing score for this return) ──────────
        user_return_count[uid] += 1
        user_return_dates[uid].append(return_request_date)
        user_seller_returns[uid][sid] = seller_ret_count + 1

        # ── Append row ─────────────────────────────────────────────────────────
        rows.append(
            {
                "return_id":              f"RET_{len(rows) + 1:07d}",
                "order_id":              ro["order_id"],
                "user_id":               uid,
                "product_id":            ro["product_id"],
                "seller_id":             sid,
                # ── Return-specific base features ──
                "reason_category":        reason_category,
                "return_type":            return_type,
                "image_uploaded":         image_uploaded,
                "return_request_date":    return_request_date.strftime("%Y-%m-%d"),
                "within_return_window":   within_return_window,
                "days_left_to_return":    days_left_to_return,
                # ── At-time user state (snapshots, avoids temporal join) ──
                "total_orders_at_time":   total_orders_at_time,
                "total_returns_at_time":  total_returns_at_time,
                # ── Metadata columns (kept for analysis; not raw ML features) ──
                "fraud_type":             sub_type,   # archetype
                # label assigned below
            }
        )

    returns_df = pd.DataFrame(rows)

    # ── Step 6: Label assignment (sigmoid probability + binomial sampling)
    # The raw_score is a continuous value. We use a sigmoid function to map it to a probability [0, 1].
    # P(fraud) = 1 / (1 + exp(-(score - threshold) / FRAUD_SCORE_NOISE_SCALE))
    # We binary-search for a 'threshold' such that the expected proportion of frauds matches FRAUD_TARGET_RATIO.
    scores = np.array(raw_scores, dtype=float)

    def sigmoid_probs(t):
        # Clip to prevent overflow
        z = np.clip(-(scores - t) / FRAUD_SCORE_NOISE_SCALE, -50, 50)
        return 1.0 / (1.0 + np.exp(z))

    # Binary search for the threshold
    low, high = -20.0, 40.0
    for _ in range(50):
        mid = (low + high) / 2.0
        p = sigmoid_probs(mid)
        if p.mean() > FRAUD_TARGET_RATIO:
            # Too many frauds -> threshold is too low
            low = mid
        else:
            high = mid

    threshold = (low + high) / 2.0
    probs = sigmoid_probs(threshold)

    # Sample labels based on the sigmoid probability to add realistic noise
    # A high score has a high prob but might still occasionally be 'legitimate' (false negative in training data)
    # A low score has a low prob but might occasionally be 'fraud' (false positive)
    is_fraud = rng.random(size=len(scores)) < probs
    labels = np.where(is_fraud, "fraud", "legitimate")

    returns_df["label"]      = labels
    returns_df["_raw_score"] = scores.round(4)   # internal; useful for EDA

    print(
        f"[returns] {len(returns_df):,} rows | "
        f"fraud={np.sum(labels=='fraud'):,} ({np.mean(labels=='fraud')*100:.1f}%) | "
        f"threshold={threshold:.3f}"
    )

    # Store internal column separately (dropped before public CSV save)
    internal_cols = ["_raw_score"]
    returns_public = returns_df.drop(columns=internal_cols)

    return returns_public


if __name__ == "__main__":
    from gen_sellers  import generate_sellers
    from gen_products import generate_products
    from gen_users    import generate_users
    from gen_orders   import generate_orders

    sellers          = generate_sellers()
    products         = generate_products()
    users, profiles  = generate_users()
    orders           = generate_orders(users, profiles, products, sellers)
    returns          = generate_returns(orders, users, products, sellers)

    print("\nLabel distribution:")
    print(returns["label"].value_counts())
    print("\nFraud type distribution:")
    print(returns["fraud_type"].value_counts())
    print("\nReason category distribution:")
    print(returns["reason_category"].value_counts())
