import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import classification_report, accuracy_score, roc_auc_score
import lightgbm as lgb
import warnings
import os
import joblib
from sqlalchemy import create_engine
import sys
sys.path.append(os.path.abspath(os.path.dirname(__file__)))
from settings import DB_URI

warnings.filterwarnings('ignore')
pd.set_option('display.max_columns', 100)

print("="*60)
print(" Flipkart Grid 8.0 - Fraud Detection Feature Engineering")
print("="*60)

# 1. Load Datasets
print("\n[1/7] Loading Datasets from PostgreSQL...")
engine = create_engine(DB_URI)

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

print(f"  Returns: {returns_df.shape}")
print(f"  Orders:  {orders_df.shape}")
print(f"  Users:   {users_df.shape}")
print(f"  Products:{products_df.shape}")
print(f"  Sellers: {sellers_df.shape}")

# 2. Global Target Encoding (Category Base Defect)
print("\n[2/7] Loading Category Base Defect rates from config...")
import sys
sys.path.append(os.path.abspath(os.path.dirname(__file__)))
from data_gen.config import CATEGORY_DEFECT_RATE

cat_defect_rate = pd.DataFrame(list(CATEGORY_DEFECT_RATE.items()), columns=['category', 'category_base_defect'])

# 3. Merge Data & Preprocess Base Dates
print("\n[3/7] Merging Data and Processing Base Features...")
df = returns_df.merge(orders_df, on='order_id', how='left')
df = df.merge(users_df, on='user_id', how='left')
df = df.merge(products_df, on='product_id', how='left')
df = df.merge(sellers_df, on='seller_id', how='left')
df = df.merge(cat_defect_rate, on='category', how='left')

df['return_request_date'] = pd.to_datetime(df['return_request_date'])
df['order_date'] = pd.to_datetime(df['order_date'])
df['delivery_date'] = pd.to_datetime(df['delivery_date'])

# Sort globally by return_request_date to ensure temporal integrity
df = df.sort_values(['return_request_date', 'order_date']).reset_index(drop=True)

df['account_creation_date'] = df['return_request_date'] - pd.to_timedelta(df['account_age_days'], unit='d')
df['days_since_delivered'] = (df['return_request_date'] - df['delivery_date']).dt.days
df['return_to_order_ratio'] = df['total_returns_at_time'] / df['total_orders_at_time'].clip(lower=1)

# 4. Feature Engineering (Temporal & Rolling)
print("\n[4/7] Generating Temporal/Rolling Features...")
# Sort strictly by user and time for shifting
df = df.sort_values(['user_id', 'return_request_date']).reset_index(drop=True)
gb = df.groupby('user_id')

# Days since last return
df['prev_return_date'] = gb['return_request_date'].shift(1)
df['days_since_last_return'] = (df['return_request_date'] - df['prev_return_date']).dt.days
df['days_since_last_return'].fillna(-1, inplace=True)

# Average Order Value (Past)
df['past_order_value_sum'] = gb['order_value'].apply(lambda x: x.shift(1).expanding().sum()).reset_index(level=0, drop=True)
df['past_returns_count'] = gb.cumcount()
df['user_avg_order_value'] = df['past_order_value_sum'] / df['past_returns_count'].clip(lower=1)
df['user_avg_order_value'].fillna(df['order_value'], inplace=True)

# Order Value Ratio
df['order_value_ratio'] = df['order_value'] / df['user_avg_order_value'].clip(lower=1)

# Complex Return Features
df['is_high_value_return'] = (df['order_value'] > (df['user_avg_order_value'] * 1.5)).astype(int)
df['high_value_return_ratio'] = gb['is_high_value_return'].apply(lambda x: x.shift(1).expanding().mean()).reset_index(level=0, drop=True).fillna(0)

df['is_weekend_return'] = df['return_request_date'].dt.dayofweek.isin([5, 6]).astype(int)
df['weekend_return_ratio'] = gb['is_weekend_return'].apply(lambda x: x.shift(1).expanding().mean()).reset_index(level=0, drop=True).fillna(0)

# Same category return ratio
df['prev_category'] = gb['category'].shift(1)
df['is_same_category'] = (df['category'] == df['prev_category']).astype(int)
df['same_category_return_ratio'] = gb['is_same_category'].apply(lambda x: x.shift(1).expanding().mean()).reset_index(level=0, drop=True).fillna(0)

# Seller Repeat Ratio
df['past_seller_returns'] = df.groupby(['user_id', 'seller_id']).cumcount()
df['seller_repeat_ratio'] = df['past_seller_returns'] / df['total_returns_at_time'].clip(lower=1)

# Returns in Last 30d and 90d
temp_df = df[['user_id', 'return_request_date']].copy()
temp_df['dummy'] = 1
temp_df.set_index('return_request_date', inplace=True)

rolling_30 = temp_df.groupby('user_id')['dummy'].rolling('30D').sum().reset_index(level=0, drop=True) - 1
rolling_90 = temp_df.groupby('user_id')['dummy'].rolling('90D').sum().reset_index(level=0, drop=True) - 1

df['returns_last_30d'] = rolling_30.values
df['returns_last_90d'] = rolling_90.values
df['returns_last_30d'] = df['returns_last_30d'].clip(lower=0)
df['returns_last_90d'] = df['returns_last_90d'].clip(lower=0)

# Return Frequency Score
df['return_frequency_score'] = df['returns_last_30d'] + (df['returns_last_90d'] - df['returns_last_30d']) * 0.3

# Re-sort to global chronological order for train/test splitting
df = df.sort_values('return_request_date').reset_index(drop=True)

# Rule-based manual verification flag
df['requires_manual_verification'] = ((df['order_value_ratio'] > 2.0) & (df['image_uploaded'] == 0) & (df['return_to_order_ratio'] > 0.3)).astype(int)
df['target'] = (df['label'] == 'fraud').astype(int)

# 5. Final Feature Selection & Preprocessing
print("\n[5/7] Finalizing Features...")
USER_FEATURES = [
    'account_age_days', 'email_verified', 'shared_device_flag',
    'total_orders_at_time', 'total_returns_at_time', 'return_to_order_ratio',
    'days_since_last_return', 'returns_last_30d', 'returns_last_90d',
    'return_frequency_score'
]
ORDER_FEATURES = [
    'order_value', 'is_prepaid', 'discount_pct', 'days_since_delivered',
    'user_avg_order_value', 'order_value_ratio'
]
PRODUCT_FEATURES = [
    'price', 'is_non_returnable', 'review_count', 'category_base_defect'
]
SELLER_FEATURES = [
    'seller_age_days', 'seller_rating', 'seller_return_rate', 'seller_customer_frequency'
]
RETURN_FEATURES = [
    'within_return_window', 'days_left_to_return', 'image_uploaded',
    'high_value_return_ratio', 'same_category_return_ratio', 'seller_repeat_ratio',
    'weekend_return_ratio', 'requires_manual_verification'
]
CATEGORICAL_FEATURES = ['category', 'reason_category', 'return_type']

NUMERIC_FEATURES = USER_FEATURES + ORDER_FEATURES + PRODUCT_FEATURES + SELLER_FEATURES + RETURN_FEATURES

for col in NUMERIC_FEATURES:
    df[col] = df[col].astype('float32')

for col in CATEGORICAL_FEATURES:
    df[col] = df[col].astype('category').cat.codes.astype('float32')

ALL_FEATURES = NUMERIC_FEATURES + CATEGORICAL_FEATURES

# 6. Temporal 80/20 Train/Test Split
print(f"\n[6/7] Splitting Data (Sequential 80/20)...")
X = df[ALL_FEATURES]
y = df['target'].astype('float32').values

split_idx = int(len(df) * 0.8)

X_train, X_test = X.iloc[:split_idx], X.iloc[split_idx:]
y_train, y_test = y[:split_idx], y[split_idx:]

print(f"  Train size: {X_train.shape[0]} ({len(X_train)/len(df)*100:.1f}%)")
print(f"  Test size:  {X_test.shape[0]} ({len(X_test)/len(df)*100:.1f}%)")
print(f"  Train Fraud Rate: {y_train.mean()*100:.1f}%")
print(f"  Test Fraud Rate:  {y_test.mean()*100:.1f}%")

# 7. Model Training & Evaluation
print(f"\n[7/7] Training LightGBM Classifier...")
model = lgb.LGBMClassifier(
    n_estimators=300,
    learning_rate=0.05,
    max_depth=7,
    num_leaves=63,
    subsample=0.8,
    colsample_bytree=0.8,
    random_state=42,
    n_jobs=-1
)

model.fit(
    X_train, y_train,
    eval_set=[(X_test, y_test)],
    categorical_feature=[ALL_FEATURES.index(c) for c in CATEGORICAL_FEATURES],
    callbacks=[lgb.early_stopping(stopping_rounds=50)]
)

print("\n" + "="*60)
print(" Evaluation Results")
print("="*60)
y_pred = model.predict(X_test)
y_prob = model.predict_proba(X_test)[:, 1]

acc = accuracy_score(y_test, y_pred)
print(f"Overall Accuracy: {acc:.4f}\n")
print("Classification Report (0 = Legitimate, 1 = Fraud):")
print(classification_report(y_test, y_pred, target_names=['Legitimate', 'Fraud']))

roc_auc = roc_auc_score(y_test, y_prob)
print(f"ROC-AUC Score: {roc_auc:.4f}")

print("\nPlotting Feature Importances...")
importance = pd.DataFrame({
    'Feature': ALL_FEATURES,
    'Importance': model.feature_importances_
}).sort_values(by='Importance', ascending=False)

plt.figure(figsize=(12, 10))
sns.barplot(x='Importance', y='Feature', data=importance.head(50), palette='viridis')
plt.title('Top 25 Feature Importances (LightGBM)')
plt.tight_layout()
plt.savefig('feature_importance.png')
print("Saved feature_importance.png")

print("\n[8/8] Saving Datasets and Model...")
train_data = X_train.copy()
train_data['target'] = y_train
test_data = X_test.copy()
test_data['target'] = y_test

train_data.to_csv('output/train_dataset.csv', index=False)
test_data.to_csv('output/test_dataset.csv', index=False)
print("Saved train_dataset.csv and test_dataset.csv")

joblib.dump(model, 'output/lgbm_model.pkl')
print("Saved lgbm_model.pkl")
