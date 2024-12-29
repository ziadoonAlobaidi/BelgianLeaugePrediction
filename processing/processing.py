
import streamlit as st
import numpy as np
from xgboost import XGBRegressor
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error, mean_absolute_error
from sklearn import preprocessing
import pandas as pd 
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.preprocessing import LabelEncoder
import numpy as np
from xgboost import XGBClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
import joblib
import os
import time


le = preprocessing.LabelEncoder()
os.chdir(os.path.dirname(os.path.abspath(__file__)))


df = pd.read_csv('new.csv')
# Initialize the LabelEncoder
data = df.copy()


def avg_Team_goal(team,overLastMatch) :
    # Your function code here

    last_fiveM_frame = data[(data['HomeTeam'] == team) | (data['AwayTeam'] == team)]

    last_fiveM_frame = last_fiveM_frame.sort_values(by='Date', ascending=False).head(overLastMatch)
    num_matches = len(last_fiveM_frame)
    if num_matches < overLastMatch :
        return None

    last_fiveM_frame = last_fiveM_frame[['HomeTeam','AwayTeam', 'Home Full-T Goals','Away Full-T Goals']].copy()

    homeSumGoal = last_fiveM_frame[last_fiveM_frame['HomeTeam'] == team ]['Home Full-T Goals'].sum()
    AwaySumGoal =  last_fiveM_frame[last_fiveM_frame['AwayTeam'] == team ]['Away Full-T Goals'].sum()

    avgGoal = (homeSumGoal + AwaySumGoal ) / overLastMatch
    
    print(f"n lsflskf some_function: seconds")

    return avgGoal


columns_to_rename  =   {
    'FTHG': 'Home Full-T Goals',
    'FTAG': 'Away Full-T Goals',
    'FTR': 'Full-T Result',
    'HTHG': 'Home Half-T Goals',
    'HTAG': 'Away Half-T Goals',
    'HTR': 'Half-T Result',
    'HS': 'Home Shots',
    'AS': 'Away Shots',
    'HST': 'Home Shots Target',
    'AST': 'Away Shots Target',
    'HF': 'Home Fouls',
    'AF': 'Away Fouls',
    'HC': 'Home Corners',
    'AC': 'Away Corners',
    'HY': 'Home Yellow Cards',
    'AY': 'Away Yellow Cards',
    'HR': 'Home Red Cards',
    'AR': 'Away Red Cards'
}

# Rename the columns using the dictionary
data.rename(columns=columns_to_rename, inplace=True)


data['Full Result_enc'] = data['Full-T Result'].map({'H': 0, 'A': 1, 'D': 2})
data['Half Result_enc'] =data['Half-T Result'].map({'H': 0, 'A': 1, 'D': 2})
data = data.drop(columns = ['Full-T Result','Half-T Result'], axis=1)


data['Home_avg5Goal'] = data['HomeTeam'].apply(lambda team: avg_Team_goal(team, 5))
data['Away_avg5Goal'] = data['AwayTeam'].apply(lambda team: avg_Team_goal(team, 5))

data['Home_avg10Goal'] = data['HomeTeam'].apply(lambda team: avg_Team_goal(team, 10))
data['Away_avg10Goal'] = data['AwayTeam'].apply(lambda team: avg_Team_goal(team, 10))

data['Home_avg20Goal'] = data['HomeTeam'].apply(lambda team: avg_Team_goal(team, 20))
data['Away_avg20Goal'] = data['AwayTeam'].apply(lambda team: avg_Team_goal(team, 20))


data = data.dropna()
dfNumeric = data.select_dtypes(include=['float64','int64','int32'])
dfNumeric.info()
#stuck ?
dfNumeric['HomeTeamEnc'] = le.fit_transform(data['HomeTeam'])
dfNumeric['AwayTeamEnc'] = le.fit_transform(data['AwayTeam'])


dfNumeric.to_pickle("cleaned_data.pkl")
dfNumeric.to_csv("dfNumeric.csv")

