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
# print(matches)

from sklearn.ensemble import RandomForestClassifier

rf = RandomForestClassifier(n_estimators=50, min_samples_split=10, random_state=1)
train = matches[matches["date"] < '2022-01-01']
test = matches[matches["date"] > '2022-01-01']
predictors = ["venue_code", "opp_code", "hour", "day_code"]
print(rf.fit(train[predictors], train["target"]))

preds = rf.predict(test[predictors])
from sklearn.metrics import accuracy_score

acc = accuracy_score(test["target"], preds)
print(acc)

combined = pd.DataFrame(dict(actual=test["target"], prediction=preds))
print(pd.crosstab(index=combined["actual"], columns=combined["prediction"]))

# Revising the predicting method
from sklearn.metrics import precision_score

precision_score(test["target"], preds)
grouped_matches = matches.groupby("team")
group = grouped_matches.get_group("Manchester City")


def rolling_averages(group, cols, new_cols):
    group = group.sort_values("date")
    rolling_stats = group[cols].rolling(3, closed='left').mean()
    group[new_cols] = rolling_stats
    group = group.dropna(subset=new_cols)
    return group


cols = ["gf", "ga", "sh", "sot", "dist", "fk", "pk", "pkatt"]
new_cols = [f"{c}_rolling" for c in cols]
rolling_averages(group, cols, new_cols)
matches_rolling = matches.groupby("team").apply(lambda x: rolling_averages(x, cols, new_cols))
matches_rolling = matches_rolling.droplevel('team')
matches_rolling.index = range(matches_rolling.shape[0])
print(matches_rolling)
