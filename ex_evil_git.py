import numpy as np 
import pandas as pd


np.random.seed(5)
n = 500
df = pd.DataFrame({
    'user_id': range(n),
    'signup_date': pd.to_datetime('2023-01-01') + pd.to_timedelta(np.random.randint(0, 700, n), unit='D'),
    'last_login': pd.to_datetime('2024-06-01') - pd.to_timedelta(np.random.exponential(30, n), unit='D'),
    'total_spent': np.random.exponential(200, n),
    'num_sessions': np.random.poisson(15, n),
    'referral_source': np.random.choice(['organic','ads','referral', None], n, p=[0.4,0.3,0.2,0.1]),
    'country': np.random.choice(['US','UK','DE','FR','ES'], n),
    })

# добавим немного грязи
df.loc[np.random.choice(n, 20), 'total_spent'] = np.nan
df.loc[np.random.choice(n, 15), 'num_sessions'] = -1# ошибка в данных

df.loc[df['num_sessions'] < 0, 'num_sessions'] = 0

df['total_spent'].fillna(df['total_spent'].median())
df['referral_source'].fillna(df['referral_source'].mode())
df['account_age_days'] = (pd.Timestamp('today') - df['signup_date']).dt.days
df['days_since_last_login'] = (pd.Timestamp('now') - df['last_login']).dt.days

df['total_spent'] = df['total_spent'].fillna(df['total_spent'].median())
df['referral_source'] = df['referral_source'].fillna(df['referral_source'].mode()[0])
df = pd.concat([df, pd.get_dummies(df['country'], prefix='country')], axis=1)
df = pd.concat([df, pd.get_dummies(df['referral_source'], prefix='ref')], axis=1)

df['engagement_rate'] = df['num_sessions'] / df['account_age_days'].clip(lower=1)
df['churn_risk_score'] = (
    (df['days_since_last_login'] > df['days_since_last_login'].quantile(0.75)).astype(int) +
    (df['engagement_rate'] < df['engagement_rate'].quantile(0.25)).astype(int) +
    (df['total_spent'] < df['total_spent'].quantile(0.25)).astype(int)
)
#yeah
#yo