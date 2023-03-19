from sklearn.linear_model import LinearRegression
import pandas as pd
from word2number import w2n as wn

df = pd.read_csv("hiring.csv")
df.experience = df['experience'].fillna('zero')

df.experience = df['experience'].apply(wn.word_to_num)
df['test_score(out of 10)'] = df['test_score(out of 10)'].fillna(df['test_score(out of 10)'].median())


reg = LinearRegression()
reg.fit(df[['experience', 'test_score(out of 10)', 'interview_score(out of 10)']], df['salary($)'])

print(reg.predict([[2, 9, 6]]))
print(reg.predict([[12, 10, 10]]))
