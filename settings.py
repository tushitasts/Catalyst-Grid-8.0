import os

# ─── DATABASE SETTINGS ───────────────────────────────────────────────────────
DB_URI = "postgresql://postgres@localhost:5432/grid_db"

# ─── GEMINI API ──────────────────────────────────────────────────────────────
GEMINI_API_KEY = ""
GEMINI_MODEL = "gemini-3.5-flash"

# ─── PATHS ───────────────────────────────────────────────────────────────────
ROOT_DIR = os.path.dirname(os.path.abspath(__file__))
OUTPUT_DIR = os.path.join(ROOT_DIR, "output")
POLICIES_DIR = os.path.join(ROOT_DIR, "final_policies")
INDEXES_DIR = os.path.join(ROOT_DIR, "indexes")

# ─── RAG SETTINGS ────────────────────────────────────────────────────────────
EMBEDDING_MODEL = "all-MiniLM-L6-v2"
RAG_TOP_K = 5
RRF_K = 60       # standard Reciprocal Rank Fusion constant
BM25_CANDIDATES = 20
FAISS_CANDIDATES = 20

# ─── RULE ENGINE THRESHOLDS ──────────────────────────────────────────────────
NEW_ACCOUNT_AGE_THRESHOLD = 120     # days
HIGH_VALUE_THRESHOLD = 10_000       # INR
HIGH_VALUE_RATIO_THRESHOLD = 1.5
MULTIPLE_RETURNS_THRESHOLD = 3

RULE_WEIGHTS = {
    "R0_NON_RETURNABLE": 50.0,     # Hard rule: product is non-returnable
    "R0_OUT_OF_WINDOW":  50.0,     # Hard rule: outside return window
    "R1": 15.0,                    # Delivery proof contradiction
    "R2": 15.0,                    # Attribute mismatch contradiction
    "R3": 15.0,                    # Multiple High-Value Items
    "R4": 15.0,                    # Sudden Behavior Change
    "R5": 10.0,                    # Return Window Manipulation
    "R6": 10.0,                    # New Account + High Value
    "R7": 10.0,                    # High Seller Fraud Rate
    "R8": 20.0,                    # Shared Device
    "R9": 15.0,                    # Discount-Driven High-Value (Wardrobing)
    "R10": 20.0,                   # Logistics Fraud (No Return)
    "R11": 20.0,                   # Duplicate Image Evidence
}

# ─── VERDICT THRESHOLDS ──────────────────────────────────────────────────────
AUTO_APPROVE_MAX_RISK = 0.20       # combined score below this -> Auto-Approve (no LLM)
AUTO_REJECT_MIN_RISK  = 0.85       # combined score above this -> Auto-Reject  (no LLM)
LGBM_WEIGHT = 0.60                 # weight of LightGBM score in combined
RULE_WEIGHT = 0.40                 # weight of rule engine score in combined

# ─── SHAP ─────────────────────────────────────────────────────────────────────
SHAP_TOP_K = 10

# ─── FEATURE DEFINITIONS (must match model_pipeline.py exactly) ──────────────
NUMERIC_FEATURES = [
    # User features
    'account_age_days', 'email_verified', 'shared_device_flag',
    'total_orders_at_time', 'total_returns_at_time', 'return_to_order_ratio',
    'days_since_last_return', 'returns_last_30d', 'returns_last_90d',
    'return_frequency_score',
    # Order features
    'order_value', 'is_prepaid', 'discount_pct', 'days_since_delivered',
    'user_avg_order_value', 'order_value_ratio',
    # Product features
    'price', 'is_non_returnable', 'review_count', 'category_base_defect',
    # Seller features
    'seller_age_days', 'seller_rating', 'seller_return_rate', 'seller_customer_frequency',
    # Return features
    'within_return_window', 'days_left_to_return', 'image_uploaded',
    'high_value_return_ratio', 'same_category_return_ratio', 'seller_repeat_ratio',
    'weekend_return_ratio', 'requires_manual_verification',
]

CATEGORICAL_FEATURES = ['category', 'reason_category', 'return_type']
ALL_FEATURES = NUMERIC_FEATURES + CATEGORICAL_FEATURES
