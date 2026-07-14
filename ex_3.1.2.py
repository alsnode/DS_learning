from sklearn.datasets import make_classification
from sklearn.linear_model import LogisticRegression, Ridge, Lasso, ElasticNet
from sklearn.metrics import accuracy_score
import numpy as np


X, y = make_classification(
    n_samples=2000, n_features=6, n_informative=4,
    weights=[0.85, 0.15], random_state=1
)


feature_names = ['income', 'debt_ratio', 'age', 'credit_history_len', 'num_late_payments', 'employment_years']

model = LogisticRegression().fit(X, y)

print(model.coef_)

unique, counts = np.unique(y, return_counts=True)
print(dict(zip(unique, counts)))

#Первый вес уменьшает шанс дефолта
#Второй вес уменьшает шанс дефолты
#Третий вес уменьшает шанс дефолты
#Четвертый вес увеличивает шанс дефолта
#Пятый вес увеличивает шанс дефолта
#Шестой вес увеличивает шанс дефолта

dummy_predictions = np.zeros(len(y))
baseline_accuracy = accuracy_score(y, dummy_predictions)
print(baseline_accuracy)
