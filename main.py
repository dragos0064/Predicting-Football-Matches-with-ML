import pandas as pd

matches = pd.read_csv("matches.csv", index_col=0)
matches.head()
# matches.shape
# 38*20*2

matches["team"].value_counts()
# print(matches[matches["team"] == "Liverpool"])
# print(matches["round"].value_counts())