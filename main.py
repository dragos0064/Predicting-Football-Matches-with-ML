import pandas as pd

matches = pd.read_csv("matches.csv", index_col=0)
matches.head()
# matches.shape
# 38*20*2

matches["team"].value_counts()
# print(matches[matches["team"] == "Liverpool"])
# print(matches["round"].value_counts())

# matches.dtypes

matches["date"] = pd.to_datetime(matches["date"])

# ML part
matches["venue_code"] = matches["venue"].astype("category").cat.codes
matches["opp_code"] = matches["opponent"].astype("category").cat.codes
matches["hour"] = matches["time"].str.replace(":.+", "", regex=True).astype("int")
matches["day_code"] = matches["date"].dt.dayofweek
matches["target"] = (matches["result"] == "W").astype("int")
print(matches)
