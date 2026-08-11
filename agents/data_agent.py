"""
Data Agent — Loads all CSVs, replicates model_pipeline.py feature engineering,
and provides per-case lookup with both raw fields and encoded feature vectors.
"""
import pandas as pd
import numpy as np
import os
import warnings

warnings.filterwarnings('ignore')

import settings
from data_gen.config import CATEGORY_DEFECT_RATE


class DataAgent:
    """Tool: Fetches and assembles all data for a given return case."""

    def __init__(self):
        print("[DataAgent] Loading and preparing data...")
        self._load_and_engineer()
        print(f"[DataAgent] Ready. {len(self.df)} returns loaded. "
              f"Test set: {len(self.df) - self.split_idx} cases.")

    # ── Internal: full-dataset feature engineering ────────────────────────────

    def _load_and_engineer(self):
        """Replicate model_pipeline.py feature engineering, keeping return_id."""
        from sqlalchemy import create_engine
        engine = create_engine(settings.DB_URI)

        returns_query = """
        SELECT 
            return_id, order_id, 
            rc.reason_category_name as reason_category,
            rt.return_type_name as return_type,
            image_uploaded, return_request_date, within_return_window, days_left_to_return,
            total_orders_at_time, total_returns_at_time,
            ft.fraud_name as fraud_type,
            lt.label_name as label
        FROM returns r
        LEFT JOIN reason_categories rc ON r.reason_category_id = rc.reason_category_id
        LEFT JOIN return_type rt ON r.return_type_id = rt.return_type_id
        LEFT JOIN fraud_type ft ON r.fraud_id = ft.fraud_id
        LEFT JOIN label_type lt ON r.label_id = lt.label_id
        """
        returns_df = pd.read_sql(returns_query, engine)

        orders_df = pd.read_sql("SELECT * FROM orders", engine)

        users_df = pd.read_sql("SELECT * FROM users", engine)

        products_query = """
        SELECT 
            product_id, pc.product_category_name as category, price, is_non_returnable, review_count, sku,
            category_base_defect_rate, return_window_days, base_defect_rate
        FROM products p
        LEFT JOIN product_categories pc ON p.product_category_id = pc.product_category_id
        """
        products_df = pd.read_sql(products_query, engine)

        sellers_query = """
        SELECT 
            seller_id, seller_age_days, seller_rating, seller_return_rate, seller_customer_frequency,
            st.seller_type_name as seller_type,
            pc.product_category_name as category_specialization,
            rating, registration_date
        FROM sellers s
        LEFT JOIN seller_types st ON s.seller_type_id = st.seller_type_id
        LEFT JOIN product_categories pc ON s.category_specialization_id = pc.product_category_id
        """
        sellers_df = pd.read_sql(sellers_query, engine)

        # Category defect rates from config (prevents target leakage)
        cat_defect = pd.DataFrame(
            list(CATEGORY_DEFECT_RATE.items()),
            columns=['category', 'category_base_defect'],
        )

        # ── Merge ─────────────────────────────────────────────────────────────
        df = returns_df.merge(
            orders_df, on='order_id', how='left',
        )
        df = df.merge(users_df, on='user_id', how='left')
        df = df.merge(products_df, on='product_id', how='left')
        df = df.merge(sellers_df, on='seller_id', how='left')
        df = df.merge(cat_defect, on='category', how='left')

        # ── Parse dates ───────────────────────────────────────────────────────
        df['return_request_date'] = pd.to_datetime(df['return_request_date'], format='mixed')
        df['order_date']          = pd.to_datetime(df['order_date'], format='mixed')
        df['delivery_date']       = pd.to_datetime(df['delivery_date'], format='mixed')

        # Global chronological sort
        df = df.sort_values(['return_request_date', 'order_date']).reset_index(drop=True)

        # ── Basic derived features ────────────────────────────────────────────
        df['days_since_delivered'] = (
            df['return_request_date'] - df['delivery_date']
        ).dt.days
        df['return_to_order_ratio'] = (
            df['total_returns_at_time'] / df['total_orders_at_time'].clip(lower=1)
        )

        # ── Temporal / rolling features (per-user) ────────────────────────────
        df = df.sort_values(['user_id', 'return_request_date']).reset_index(drop=True)
        gb = df.groupby('user_id')

        # Days since last return
        df['prev_return_date'] = gb['return_request_date'].shift(1)
        df['days_since_last_return'] = (
            df['return_request_date'] - df['prev_return_date']
        ).dt.days
        df['days_since_last_return'].fillna(-1, inplace=True)

        # User avg order value (expanding over past returns only)
        df['past_order_value_sum'] = gb['order_value'].apply(
            lambda x: x.shift(1).expanding().sum()
        ).reset_index(level=0, drop=True)
        df['past_returns_count'] = gb.cumcount()
        df['user_avg_order_value'] = (
            df['past_order_value_sum'] / df['past_returns_count'].clip(lower=1)
        )
        df['user_avg_order_value'].fillna(df['order_value'], inplace=True)

        # Order value ratio
        df['order_value_ratio'] = (
            df['order_value'] / df['user_avg_order_value'].clip(lower=1)
        )

        # High-value return features
        df['is_high_value_return'] = (
            df['order_value'] > (df['user_avg_order_value'] * 1.5)
        ).astype(int)
        df['high_value_return_ratio'] = gb['is_high_value_return'].apply(
            lambda x: x.shift(1).expanding().mean()
        ).reset_index(level=0, drop=True).fillna(0)

        # Weekend return
        df['is_weekend_return'] = (
            df['return_request_date'].dt.dayofweek.isin([5, 6])
        ).astype(int)
        df['weekend_return_ratio'] = gb['is_weekend_return'].apply(
            lambda x: x.shift(1).expanding().mean()
        ).reset_index(level=0, drop=True).fillna(0)

        # Same-category return ratio
        df['prev_category'] = gb['category'].shift(1)
        df['is_same_category'] = (df['category'] == df['prev_category']).astype(int)
        df['same_category_return_ratio'] = gb['is_same_category'].apply(
            lambda x: x.shift(1).expanding().mean()
        ).reset_index(level=0, drop=True).fillna(0)

        # Seller repeat ratio
        df['past_seller_returns'] = df.groupby(['user_id', 'seller_id']).cumcount()
        df['seller_repeat_ratio'] = (
            df['past_seller_returns'] / df['total_returns_at_time'].clip(lower=1)
        )

        # Rolling returns in last 30d / 90d
        temp = df[['user_id', 'return_request_date']].copy()
        temp['dummy'] = 1
        temp.set_index('return_request_date', inplace=True)
        rolling_30 = (
            temp.groupby('user_id')['dummy'].rolling('30D').sum()
            .reset_index(level=0, drop=True) - 1
        )
        rolling_90 = (
            temp.groupby('user_id')['dummy'].rolling('90D').sum()
            .reset_index(level=0, drop=True) - 1
        )
        df['returns_last_30d'] = rolling_30.values
        df['returns_last_90d'] = rolling_90.values
        df['returns_last_30d'] = df['returns_last_30d'].clip(lower=0)
        df['returns_last_90d'] = df['returns_last_90d'].clip(lower=0)

        # Return frequency score
        df['return_frequency_score'] = (
            df['returns_last_30d']
            + (df['returns_last_90d'] - df['returns_last_30d']) * 0.3
        )

        # Re-sort to global chronological order for train/test split
        df = df.sort_values('return_request_date').reset_index(drop=True)

        # Manual-verification flag
        df['requires_manual_verification'] = (
            (df['order_value_ratio'] > 2.0)
            & (df['image_uploaded'] == 0)
            & (df['return_to_order_ratio'] > 0.3)
        ).astype(int)

        # Target
        df['target'] = (df['label'] == 'fraud').astype(int)

        # ── Save original category strings BEFORE encoding ────────────────────
        for col in settings.CATEGORICAL_FEATURES:
            df[f'{col}_str'] = df[col].copy()

        # ── Encode categoricals (same as model_pipeline.py) ───────────────────
        for col in settings.NUMERIC_FEATURES:
            df[col] = df[col].astype('float32')
        for col in settings.CATEGORICAL_FEATURES:
            df[col] = df[col].astype('category').cat.codes.astype('float32')

        # Chronological 80/20 split index
        self.split_idx = int(len(df) * 0.8)
        self.df = df

    # ── Public API ────────────────────────────────────────────────────────────

    def get_case(self, return_id: str) -> dict:
        """Get all data for a single return case, ready for all agents."""
        mask = self.df['return_id'] == return_id
        if not mask.any():
            raise ValueError(f"Return ID '{return_id}' not found in dataset.")

        row = self.df[mask].iloc[0]

        # ── Raw dict (original readable values) ──────────────────────────────
        raw = {}
        for col in row.index:
            if col.endswith('_str'):
                continue      # handled separately
            val = row[col]
            # Swap encoded categoricals with original strings
            if col in settings.CATEGORICAL_FEATURES:
                raw[col] = row[f'{col}_str']
            elif isinstance(val, pd.Timestamp):
                raw[col] = val.strftime('%Y-%m-%d')
            elif isinstance(val, (np.integer,)):
                raw[col] = int(val)
            elif isinstance(val, (np.floating,)):
                raw[col] = float(val)
            else:
                raw[col] = val

        # Drop internal helper columns from raw
        for key in list(raw.keys()):
            if key.startswith(('prev_', 'past_', 'is_same_', 'is_high_value',
                               'is_weekend', 'account_creation', '_')):
                raw.pop(key, None)

        # ── Feature vector (encoded, ready for LightGBM) ─────────────────────
        feature_vector = row[settings.ALL_FEATURES].to_frame().T.copy()
        for col in settings.ALL_FEATURES:
            feature_vector[col] = feature_vector[col].astype('float32')

        return {
            'return_id': return_id,
            'raw': raw,
            'feature_vector': feature_vector,
            'feature_names': list(settings.ALL_FEATURES),
        }

    def get_test_return_ids(self) -> list:
        """Return all return_ids in the test set (last 20%)."""
        return self.df.iloc[self.split_idx:]['return_id'].tolist()

    def get_random_test_cases(self, n: int = 5) -> list:
        """Sample N random return_ids from the test set."""
        import random
        ids = self.get_test_return_ids()
        return random.sample(ids, min(n, len(ids)))

    def get_all_test_data(self) -> pd.DataFrame:
        """Return the full test slice (for batch evaluation)."""
        return self.df.iloc[self.split_idx:].copy()
