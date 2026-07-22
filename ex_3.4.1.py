import pandas as pd
import numpy as np


df = pd.DataFrame({
    'price': [100, 250, 80, 500, 120],
    'category': ['A', 'B', 'A', 'C', 'B'],
    'date': pd.to_datetime(['2024-01-05','2024-03-15','2024-06-20','2024-11-01','2024-12-25'])
})

price_log = np.log(df['price'])
category_one_hot = pd.get_dummies(['category'])
month = df['date'].dt.month
day_of_week = df['date'].dt.dayofweek

def IsWeekend():
    for i in day_of_week:
        return True if i in (5, 6) else False

IsWeekend()

price_bucket = pd.cut(df['price'], bins=3, labels=['low', 'mid', 'high'])

print(
    price_log, category_one_hot, month, day_of_week, price_bucket
)