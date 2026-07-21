import numpy as np
import pandas as pd
from sklearn.datasets import make_classification
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from xgboost import XGBClassifier


np.random.seed(42)
 
X, y = make_classification(
    n_samples=5000,
    n_features=10,
    n_informative=7,
    n_redundant=2,
    weights=[0.85, 0.15],  # дисбаланс классов!
    random_state=42
)

feature_names = [
    'tenure_months',    # сколько месяцев клиент
    'monthly_charges',  # ежемесячная оплата
    'total_charges',    # общая сумма
    'num_products',     # количество продуктов
    'support_calls',    # обращения в поддержку
    'satisfaction_score',
    'age',
    'contract_type',    # (закодировано числом)
    'payment_method',
    'internet_service'
]
 
df = pd.DataFrame(X, columns=feature_names)
df['churn'] = y

X_train, X_test, y_train, y_test = train_test_split(
    X, y,
    test_size=0.2,
    random_state=42
)

X_train, X_val, y_train, y_val = train_test_split(
    X_train, y_train,
    test_size=0.1,
    random_state=42
)

rf = RandomForestClassifier(
    n_estimators = 200, max_depth=10,
    max_features='sqrt', bootstrap='True', 
    class_weight='balanced', n_jobs=-1
)
rf.fit(X_train, y_train)
print(rf.feature_importances_)

xgb = XGBClassifier(
    n_estimators=200, learning_rate=0.1,
    max_depth=4, subsample=0.8,
    colsample_bytree=0.8, eval_metric='logloss'
)

xgb.fit(X_train, y_train, eval_set=[(X_val, y_val)], early_stopping_rounds=20)
print(xgb.feature_importance_)