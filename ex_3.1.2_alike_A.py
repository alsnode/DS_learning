import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression, Ridge 
from sklearn.metrics import r2_score


np.random.seed(10)


n = 60  # намеренно МАЛО наблюдений
X = np.random.normal(0, 1, size=(n, 25))  # намеренно МНОГО признаков (25)
true_weights = np.zeros(25)
true_weights[:3] = [5, -3, 2]  # только первые 3 признака реально влияют на y
y = X @ true_weights + np.random.normal(0, 1, n)

X_train, X_test, y_train, y_test = train_test_split(
    X, y,
    test_size=0.3,
    random_state=42
)

model = LinearRegression().fit(X_train, y_train)
train_predictions = model.predict(X_train)
test_predictions = model.predict(X_test)

train_r2 = r2_score(y_train, train_predictions)
test_r2 = r2_score(y_test, test_predictions)

print(train_r2, test_r2)

for i in (0.1, 1, 10, 100):
    clf = Ridge(alpha=i)
    flf = clf.fit(X_train, y_train)
    train_pred = flf.predict(X_train)
    test_pred  = flf.predict(X_test)
    r2_train_ridge = r2_score(y_train, train_pred)
    r2_test_ridge = r2_score(y_test, test_pred)
    print(f"This is r2 of train ridge {r2_train_ridge}")
    print(f"This is r2 of test ridge {r2_test_ridge}")

   
