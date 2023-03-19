import numpy as np
from sklearn.linear_model import LinearRegression
import pandas as pd
import math

data = pd.read_csv("test_scores.csv")
X = data['math']
Y = data['cs']
X = np.array(X)
Y = np.array(Y)


def gradient_descent(x, y):
    m = b = 0
    n = len(x)
    learning_rate = 0.0002
    cost_previous = 0
    i = 0
    while True:
        y_predicted = m * x + b
        cost = (1 / n) * sum([val ** 2 for val in (y - y_predicted)])
        md = -(2 / n) * sum(x * (y - y_predicted))
        bd = -(2 / n) * sum(y - y_predicted)
        m = m - learning_rate * md
        b = b - learning_rate * bd
        if math.isclose(cost, cost_previous, rel_tol=1e-20):
            break
        cost_previous = cost
        i += 1
        print(
            f"m {m}, b {b}, cost {cost}, iterations {i}")  # m 1.0177381667793246, b 1.9150826134339467,  cost
        # 31.604511334602297, iterations 415533 Final Value


def model_check(x, y):
    model = LinearRegression()
    model.fit(data[['math']], data.cs)
    print(model.intercept_)
    print(model.coef_)


model_check(X, Y)  # b = 1.9152193111569034, m = [1.01773624]
