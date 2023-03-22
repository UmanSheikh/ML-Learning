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
    m = b = 0  # Starting value of m and b from 0 //////// Main equation is y = mx + b where y is our predicted value
    n = len(x)  # N is the total numbers of x value
    learning_rate = 0.0002  # It is the maximum learning rate I could found, Others are just increasing the cost
    cost_previous = 0  # Created to check the last cost value to current value. There will be a point where cost will not decrease anymore
    i = 0  # For Keeping track of itreation (It is not recommend becaue for loop is faster then while loop)
    while True:
        y_predicted = m * x + b  # Calculating the initial value of y
        cost = (1 / n) * sum([val ** 2 for val in (y - y_predicted)])  #Formula of cost (cost = 1/n * Σ (y - y_predicted)^2)  Here Σ (sigma) is generally used to denote a sum
        md = -(2 / n) * sum(x * (y - y_predicted))  #Formula of partial derivative of m (d/dx m = - 2/n Σ x( y- y_predicted)  Here Σ (sigma) is generally used to denote a sum
        bd = -(2 / n) * sum(y - y_predicted)  #Formula of partial derivative of b (d/dx b = - 2/n Σ ( y- y_predicted)  Here Σ (sigma) is generally used to denote a sum
        m = m - learning_rate * md  # formula of calculating m ( m = m - learning_rate * md)
        b = b - learning_rate * bd  # formula of calculating b ( b = b - learning_rate * bd)
        if math.isclose(cost, cost_previous, rel_tol=1e-20):  # Checking if the previous value of cost is euqal to current value of cost
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
