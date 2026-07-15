from sklearn.datasets import make_classification
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression, Lasso

X, y = make_classification(
    n_samples=1000, n_features=20, n_informative=5,
    n_redundant=10, random_state=7
)

X_train, X_test, y_train, y_test = train_test_split(
    X, y,
    random_state=42,
    test_size=0.3
)

model = LogisticRegression(
    solver='saga',
    C=1.0,
    penalty='l1',
    max_iter=1000000
)

model.fit(X_train, y_train)

weights = model.coef_

zero_count = (weights == 0).sum()
print(zero_count)

    


