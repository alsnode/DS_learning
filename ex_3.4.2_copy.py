import numpy as np 
import pandas as pd


np.random.seed(5)
n = 500
df = pd.DataFrame({
    'user_id': range(n),
    'signup_date': pd.to_datetime('2023-01-01') + pd.to_timedelta(np.random.randint(0, 700, n), unit='D'),
    'last_login': pd.to_datetime('2024-06-01') - pd.to_timedelta(np.random.exponential(30, n)),
    'total_spent': np.random.exponential(200, n),
    'num_sessions': np.random.poisson(15, n),
    'referral_source': np.random.choice(['organic','ads','referral', None], n, p=[0.4,0.3,0.2,0.1]),
    'country': np.random.choice(['US','UK','DE','FR','ES'], n),
    'days_since_last_login': np.datetime64('now') - pd.to_timedelta(np.random.exponential(30,n))
    })

# добавим немного грязи
df.loc[np.random.choice(n, 20), 'total_spent'] = np.nan
df.loc[np.random.choice(n, 15), 'num_sessions'] = -1# ошибка в данных

df.loc[df['num_sessions'] < 0, 'num_sessions'] = 0

df['total_spent'].fillna(df['total_spent'].median())
df['referral_source'].fillna(df['referral_source'].mode())

df['account_age_days'] = np.datetime64('today') - df['signup_date']

pd.get_dummies(df['country'])
pd.get_dummies(df['referral_source'])













    
