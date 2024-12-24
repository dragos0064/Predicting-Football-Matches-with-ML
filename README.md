# Predicting-Football-Matches-with-ML⚽
This project leverages machine learning to predict the outcomes of football matches in the Premier League, using historical data and advanced statistical techniques. The model predicts whether a team will win, lose, or draw a match based on features such as venue, opponent, time, and rolling averages of key metrics like goals scored, shots, and possession.

## Project Highlights
Data Source: Match data is extracted from FBRef and preprocessed into a clean, structured format.
Machine Learning: A Random Forest Classifier is trained to predict match outcomes, achieving competitive accuracy.
Rolling Averages: Incorporates rolling averages of key performance metrics (e.g., goals, shots, possession) to capture team form over time.
Model Evaluation: Evaluates performance using metrics like accuracy and precision to ensure reliability.
Dataset
The dataset includes:

### General Match Details:
- date, time, venue, opponent, team
### Match Metrics:
- gf (Goals For), ga (Goals Against), sh (Shots), poss (Possession), and more.
### Categorical Features:
- Venue (home/away), Day of the Week, Opponent, and Round.
### Rolling Averages:
- 3-match rolling averages of goals, shots, and other metrics to reflect team performance trends.
### Features
- Rolling Averages:
  - Captures team performance trends over the last 3 matches, including:
  - Goals scored/conceded
  - Shots on target
  - Possession percentage
## Machine Learning:

  - A Random Forest Classifier is trained with key predictors:
  - Venue (home/away)
  - Opponent
  - Match start time
  - Day of the week
### Evaluation:

- Uses metrics like accuracy and precision for assessing the model's effectiveness.
- Generates confusion matrices to visualize prediction performance.
### Combined Predictions:

- Merges predictions for home and away teams to analyze outcomes at a match level.
- Getting Started
### Prerequisites
- Python 3.7+
### Libraries:
- pandas
- numpy
- scikit-learn
- matplotlib
- BeautifulSoup
### Install dependencies using:

```bash

pip install pandas numpy scikit-learn matplotlib beautifulsoup4
```
## Running the Project
### Prepare the Dataset:

- Scrape live data to provide matches.csv in the main.py script.
### Train the Model:
- Train a Random Forest model with preprocessed data:
```python
rf.fit(train[predictors], train["target"])
```
### Make Predictions:

- Generate predictions for upcoming matches and evaluate the model.
### Analyze Results:

- View combined predictions and rolling averages for deeper insights.
### File Structure
- matches.csv: Historical match data for the Premier League.
- main.py: The main script containing data preprocessing, feature engineering, model training, and evaluation.

### Contributing
- Contributions are welcome! Feel free to submit pull requests or raise issues to improve the project.
