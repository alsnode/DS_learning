from sklearn.linear_model import LinearRegression
import numpy as np

X = np.array([[1, 1],
              [1, 2], 
              [1, 3],
              [1, 4],
              [1, 5],
              [1, 6],
              [1, 7],
              [1, 8]])

y = np.array([52, 58, 62, 68, 74, 78, 84, 88])
# w = (X^T X)^{-1} X^T y
def matrix_formula():
    m_to_inv = X.T @ X
    inverted_X = np.linalg.inv(m_to_inv)
    transponded_X = X.T @ y
    res = inverted_X @ transponded_X
    print(res)
    

matrix_formula()

reg = LinearRegression().fit(X, y)

print(reg)