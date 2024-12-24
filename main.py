import pandas as pd
import requests
import time
from selenium import webdriver
from bs4 import BeautifulSoup

# Initialize the WebDriver
driver = webdriver.Chrome()  # Make sure to install the Chrome WebDriver
driver.get("https://fbref.com/en/comps/9/Premier-League-Stats")

# Get the rendered page source
soup = BeautifulSoup(driver.page_source, features="lxml")

# Find the table
standings_table = soup.select('table.stats_table')
if not standings_table:
    print("Stats table not found after rendering.")
else:
    standings_table = standings_table[0]
    print(standings_table)

driver.quit()

standings_table = soup.select('table.stats_table')[0]
print(standings_table)
links = standings_table.find_all('a')
links = [l.get("href") for l in links]
links = [l for l in links if '/squads/' in l]
team_urls = [f"https://fbref.com{l}" for l in links]
data = requests.get(team_urls[0])
import pandas as pd
matches = pd.read_html(data.text, match="Scores & Fixtures")[0]
soup = BeautifulSoup(data.text)
links = soup.find_all('a')
links = [l.get("href") for l in links]
links = [l for l in links if l and 'all_comps/shooting/' in l]
data = requests.get(f"https://fbref.com{links[0]}")
shooting = pd.read_html(data.text, match="Shooting")[0]
shooting.head()
shooting.columns = shooting.columns.droplevel()
team_data = matches.merge(shooting[["Date", "Sh", "SoT", "Dist", "FK", "PK", "PKatt"]], on="Date")
team_data.head()
years = list(range(2022, 2020, -1))
all_matches = []
standings_url = "https://fbref.com/en/comps/9/Premier-League-Stats"
import time

for year in years:
    data = requests.get(standings_url)
    soup = BeautifulSoup(data.text)
    standings_table = soup.select('table.stats_table')[0]

    links = [l.get("href") for l in standings_table.find_all('a')]
    links = [l for l in links if '/squads/' in l]
    team_urls = [f"https://fbref.com{l}" for l in links]

    previous_season = soup.select("a.prev")[0].get("href")
    standings_url = f"https://fbref.com{previous_season}"

    for team_url in team_urls:
        team_name = team_url.split("/")[-1].replace("-Stats", "").replace("-", " ")
        data = requests.get(team_url)
        matches = pd.read_html(data.text, match="Scores & Fixtures")[0]
        soup = BeautifulSoup(data.text)
        links = [l.get("href") for l in soup.find_all('a')]
        links = [l for l in links if l and 'all_comps/shooting/' in l]
        data = requests.get(f"https://fbref.com{links[0]}")
        time.sleep(15)
        shooting = pd.read_html(data.text, match="Shooting")[0]
        shooting.columns = shooting.columns.droplevel()
        try:
            team_data = matches.merge(shooting[["Date", "Sh", "SoT", "Dist", "FK", "PK", "PKatt"]], on="Date")
        except ValueError:
            continue
        team_data = team_data[team_data["Comp"] == "Premier League"]

        team_data["Season"] = year
        team_data["Team"] = team_name
        all_matches.append(team_data)
        time.sleep(15)
match_df = pd.concat(all_matches)
match_df.columns = [c.lower() for c in match_df.columns]
match_df.to_csv("matches.csv")
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


# Retraining our machine learning model
def make_predictions(data, predictors):
    train = data[data["date"] < '2022-01-01']
    test = data[data["date"] > '2022-01-01']
    rf.fit(train[predictors], train["target"])
    preds = rf.predict(test[predictors])
    combined = pd.DataFrame(dict(actual=test["target"], predicted=preds), index=test.index)
    precision = precision_score(test["target"], preds)
    return combined, precision


combined, precision = make_predictions(matches_rolling, predictors + new_cols)
print(precision)
print(combined)

combined = combined.merge(matches_rolling[["date", "team", "opponent", "result"]], left_index=True, right_index=True)
print(combined)


# Combining home and away predictions
class MissingDict(dict):
    __missing__ = lambda self, key: key


map_values = {
    "Brighton and Hove Albion": "Brighton",
    "Manchester United": "Manchester Utd",
    "Newcastle United": "Newcastle Utd",
    "Tottenham Hotspur": "Tottenham",
    "West Ham United": "West Ham",
    "Wolverhampton Wanderers": "Wolves"
}

mapping = MissingDict(**map_values)
print(mapping["West Ham United"])

combined["new_team"] = combined["team"].map(mapping)
print(combined)

merged = combined.merge(combined, left_on=["date", "new_team"], right_on=["date", "opponent"])
print(merged)
print(merged[(merged["predicted_x"] == 1) & (merged["predicted_y"] == 0)]["actual_x"].value_counts())



