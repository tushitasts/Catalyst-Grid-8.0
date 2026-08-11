"""
run_pipeline.py — Entry point for the full data generation pipeline.

Usage:
    cd e:\\Codes\\FlipKartGrid8.0\\data_gen
    python run_pipeline.py

Output CSVs are saved to the 'output/' directory.
"""

import os
import sys
import time
import pandas as pd

# Allow running from any directory
sys.path.insert(0, os.path.dirname(__file__))

from config import OUTPUT_DIR, RANDOM_SEED, N_USERS, N_PRODUCTS, N_SELLERS, N_RETURNS

from gen_sellers  import generate_sellers
from gen_products import generate_products
from gen_users    import generate_users
from gen_orders   import generate_orders
from gen_returns  import generate_returns


def drop_internal(df: pd.DataFrame) -> pd.DataFrame:
    """Remove any column whose name starts with '_'."""
    # Drop columns prefixed with '_' (always internal) and any named internal cols
    internal = [c for c in df.columns if c.startswith("_")]
    internal += ["seller_order_weight"]   # not prefixed but internal to gen_sellers
    return df.drop(columns=internal, errors="ignore")


def save(df: pd.DataFrame, name: str, out_dir: str) -> None:
    path = os.path.join(out_dir, name)
    df.to_csv(path, index=False)
    print(f"     [OK] Saved {name} ({len(df):,} rows, {os.path.getsize(path)/1024:.1f} KB)")


def main() -> None:
    os.makedirs(OUTPUT_DIR, exist_ok=True)

    DIVIDER = "-" * 62
    print(DIVIDER)
    print(" Flipkart Grid 8.0 - Synthetic Data Generation Pipeline")
    print(DIVIDER)

    t_total = time.time()

    # -- 1. Sellers ------------------------------------------------------------
    print(f"\n[1/5] Generating {N_SELLERS} sellers ...")
    t = time.time()
    sellers_df = generate_sellers(seed=RANDOM_SEED)
    save(drop_internal(sellers_df), "sellers.csv", OUTPUT_DIR)
    print(f"       age range : {sellers_df['seller_age_days'].min()}d – "
          f"{sellers_df['seller_age_days'].max()}d")
    print(f"       rating    : {sellers_df['seller_rating'].mean():.2f} avg")
    print(f"       done in {time.time()-t:.1f}s")

    # -- 2. Products -----------------------------------------------------------
    print(f"\n[2/5] Generating {N_PRODUCTS} products ...")
    t = time.time()
    products_df = generate_products(seed=RANDOM_SEED)
    save(products_df, "products.csv", OUTPUT_DIR)
    print(f"       categories : {products_df['category'].nunique()}")
    print(f"       non-returnable : {products_df['is_non_returnable'].mean()*100:.1f}%")
    print(f"       price range  : Rs.{products_df['price'].min():.0f} to Rs.{products_df['price'].max():,.0f}")
    print(f"       done in {time.time()-t:.1f}s")

    # -- 3. Users --------------------------------------------------------------
    print(f"\n[3/5] Generating {N_USERS} users ...")
    t = time.time()
    users_df, profiles_df = generate_users(seed=RANDOM_SEED)
    save(users_df, "users.csv", OUTPUT_DIR)

    print("       Behavior type distribution:")
    bt_counts = profiles_df["behavior_type"].value_counts()
    for bt, cnt in bt_counts.items():
        print(f"         {bt:<26s}: {cnt:>5,}  ({cnt/N_USERS*100:.1f}%)")
    print(f"       Planned total orders : {profiles_df['n_orders'].sum():,}")
    print(f"       done in {time.time()-t:.1f}s")

    # -- 4. Orders -------------------------------------------------------------
    print("\n[4/5] Generating orders ...")
    t = time.time()
    orders_df = generate_orders(users_df, profiles_df, products_df, sellers_df,
                                seed=RANDOM_SEED)
    save(drop_internal(orders_df), "orders.csv", OUTPUT_DIR)

    n_ret_orders = int(orders_df["is_returned"].sum())
    print(f"       total orders     : {len(orders_df):,}")
    print(f"       returned orders  : {n_ret_orders:,} ({n_ret_orders/len(orders_df)*100:.1f}%)")
    print(f"       date range       : {orders_df['order_date'].min()} to "
          f"{orders_df['order_date'].max()}")
    print(f"       avg order value  : Rs.{orders_df['order_value'].mean():,.0f}")
    print(f"       done in {time.time()-t:.1f}s")

    if n_ret_orders < N_RETURNS:
        print(
            f"\n  [WARNING] Only {n_ret_orders} returned orders found; "
            f"returns.csv will have {n_ret_orders} rows instead of {N_RETURNS}.\n"
            "  -> Increase return_rate values in USER_BEHAVIOR_TYPES in config.py."
        )

    # -- 5. Returns ------------------------------------------------------------
    print(f"\n[5/5] Generating up to {N_RETURNS} return requests ...")
    t = time.time()
    returns_df = generate_returns(orders_df, users_df, products_df, sellers_df,
                                  seed=RANDOM_SEED)
    save(returns_df, "returns.csv", OUTPUT_DIR)

    fraud_n   = (returns_df["label"] == "fraud").sum()
    legit_n   = (returns_df["label"] == "legitimate").sum()
    print(f"       label distribution :")
    print(f"         fraud      : {fraud_n:>6,}  ({fraud_n/len(returns_df)*100:.1f}%)")
    print(f"         legitimate : {legit_n:>6,}  ({legit_n/len(returns_df)*100:.1f}%)")
    print("\n       Fraud type breakdown:")
    for ft, cnt in returns_df["fraud_type"].value_counts().items():
        lbl = returns_df[returns_df["fraud_type"] == ft]["label"]
        fraud_pct = (lbl == "fraud").mean() * 100
        print(f"         {ft:<26s}: {cnt:>5,}  (fraud rate {fraud_pct:.0f}%)")
    print(f"       done in {time.time()-t:.1f}s")

    # ── Summary ───────────────────────────────────────────────────────────────
    print(f"\n{DIVIDER}")
    print(f"  DONE: Pipeline complete in {time.time()-t_total:.1f}s")
    print(f"  Output: {os.path.abspath(OUTPUT_DIR)}/")
    print(DIVIDER)

    # Quick referential-integrity check
    _validate(orders_df, returns_df, users_df, products_df, sellers_df)


def _validate(orders_df, returns_df, users_df, products_df, sellers_df) -> None:
    print("\n--- Referential integrity checks ---")
    ok = True

    # All user_ids in orders exist in users
    bad = ~orders_df["user_id"].isin(users_df["user_id"])
    if bad.any():
        print(f"  [FAIL] {bad.sum()} orders have unknown user_id")
        ok = False

    # All product_ids in orders exist in products
    bad = ~orders_df["product_id"].isin(products_df["product_id"])
    if bad.any():
        print(f"  [FAIL] {bad.sum()} orders have unknown product_id")
        ok = False

    # All seller_ids in orders exist in sellers
    bad = ~orders_df["seller_id"].isin(sellers_df["seller_id"])
    if bad.any():
        print(f"  [FAIL] {bad.sum()} orders have unknown seller_id")
        ok = False

    # All order_ids in returns exist in orders (and are marked as returned)
    ret_orders = returns_df["order_id"].isin(orders_df[orders_df["is_returned"] == 1]["order_id"])
    if not ret_orders.all():
        print(f"  [FAIL] {(~ret_orders).sum()} return rows reference non-returned orders")
        ok = False

    # No duplicate return_ids
    if returns_df["return_id"].duplicated().any():
        print("  [FAIL] Duplicate return_ids found")
        ok = False

    if ok:
        print("  [OK] All integrity checks passed")


if __name__ == "__main__":
    main()
