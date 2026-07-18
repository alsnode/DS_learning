from sklearn.datasets import make_classification
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression, Lasso
from sklearn.metrics import roc_auc_score, f1_score

X, y = make_classification(
    n_samples=1000, n_features=20, n_informative=5,
    n_redundant=10, random_state=7
)

X_train, X_test, y_train, y_test = train_test_split(
    X, y,
    random_state=42,
    test_size=0.3
)

model_l1 = LogisticRegression(
    solver='saga',
    C=1.0,
    penalty='l1',
    max_iter=1000000
)

model_l1.fit(X_train, y_train)

weights = model_l1.coef_

zero_count = (weights == 0).sum()


model_l2 = LogisticRegression(
    solver='saga',
    C=1.0,
    penalty='l2',
    max_iter=10000
)

model_l2.fit(X_train, y_train)

l2_weights = model_l2.coef_

l2_zero_count = (weights == 0).sum()


#print(f'{l2_zero_count} - l2')
#print(weights)
#print('-' * 50)
#print(l2_weights)

model_l1_predictions = model_l1.predict_proba(X_test)[:, 1]
model_l2_predictions = model_l2.predict_proba(X_test)[:, 1]

print(f"Predictions of l1 model {model_l1_predictions}")
print(f"Predictions of l2 model {model_l2_predictions}")

#Scoring both models by Roc Auc score 
model_l1_RocAucScore = roc_auc_score(y_test, model_l1_predictions)
model_l2_RocAucScore = roc_auc_score(y_test, model_l2_predictions)

print(f"Roc Auc score of l1 model {model_l1_RocAucScore}")
print(f"Roc Auc score of l2 model {model_l2_RocAucScore}")

#Scoring both models by f1_score
model_l1_predictions_classes = model_l1.predict(X_test)
model_l2_predictions_classes = model_l2.predict(X_test)
model_l1_f1Score = f1_score(y_test, model_l1_predictions_classes)
model_l2_f1Score = f1_score(y_test, model_l2_predictions_classes)

print(f"F1 score of l1 model - {model_l1_f1Score}")
print(f"F1 score of l2 model - {model_l2_f1Score}")






