import pandas as pd
from sklearn.linear_model import LinearRegression

df = pd.read_csv("canada_per_capita_income.csv")

model = LinearRegression()
model.fit(df[['year']], df['per capita income (US$)'])
print(model.predict([[2021]]))
