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

os.chdir(os.path.dirname(os.path.abspath(__file__)))

dfNumeric = pd.read_pickle("cleaned_data.pkl")
clubs = ('Anderlecht', 'Antwerp', 'Beerschot VA', 'Cercle Brugge',
       'Charleroi', 'Club Brugge', 'Dender', 'Eupen', 'Genk', 'Gent',
       'Kortrijk', 'Mechelen', 'Mouscron', 'Oostende',
       'Oud-Heverlee Leuven', 'RWD Molenbeek', 'Seraing', 'St Truiden',
       'St. Gilloise', 'Standard', 'Waasland-Beveren', 'Waregem',
       'Westerlo')


clubs_selectBox = ('Select a club...','Anderlecht', 'Antwerp', 'Beerschot VA', 'Cercle Brugge',
       'Charleroi', 'Club Brugge', 'Dender', 'Eupen', 'Genk', 'Gent',
       'Kortrijk', 'Mechelen', 'Mouscron', 'Oostende',
       'Oud-Heverlee Leuven', 'RWD Molenbeek', 'Seraing', 'St Truiden',
       'St. Gilloise', 'Standard', 'Waasland-Beveren', 'Waregem',
       'Westerlo')


club_mapping = {'Anderlecht': 0,
 'Antwerp': 1,
 'Beerschot VA': 2,
 'Cercle Brugge': 3,
 'Charleroi': 4,
 'Club Brugge': 5,
 'Dender': 6,
 'Eupen': 7,
 'Genk': 8,
 'Gent': 9,
 'Kortrijk': 10,
 'Mechelen': 11,
 'Mouscron': 12,
 'Oostende': 13,
 'Oud-Heverlee Leuven': 14,
 'RWD Molenbeek': 15,
 'Seraing': 16,
 'St Truiden': 17,
 'St. Gilloise': 18,
 'Standard': 19,
 'Waasland-Beveren': 20,
 'Waregem': 21,
 'Westerlo': 22}



def GetClubEncodedValue(clubName)->int: 
    for key, value in club_mapping.items():  # for name, age in dictionary.iteritems():  (for Python 2.x)
        if key == clubName:
           
           return value
        

def GetClubXColumn(MatchSide,club) : 
    start_time = time.time()  # Record start time

    encodedClub = GetClubEncodedValue(club)

    if MatchSide == 'Home' : 
        AllClub_matches = dfNumeric[(dfNumeric['HomeTeamEnc'] == encodedClub) ]
    else :
        AllClub_matches = dfNumeric[(dfNumeric['AwayTeamEnc'] == encodedClub) ]

    awayColumns = ['AwayTeamEnc','Away Shots','Away Shots Target', 'Away Fouls', 
    'Away Corners',  'Away Yellow Cards',
    'Away Red Cards', 'Away_avg5Goal',
    'Away_avg10Goal','Away_avg20Goal','Away Half-T Goals','Away Full-T Goals']


    homeColumns = ['HomeTeamEnc','Home Shots','Home Shots Target', 'Home Fouls', 
    'Home Corners',  'Home Yellow Cards',
    'Home Red Cards', 'Home Half-T Goals','Home_avg5Goal',
    'Home_avg10Goal','Home_avg20Goal','Home Full-T Goals']

    AwayClub_matches = AllClub_matches[awayColumns]
    HomeClub_matches = AllClub_matches[homeColumns]

    HomeClub_matches_X = HomeClub_matches.drop(columns=['Home Full-T Goals'],axis=1)
    AwayClub_matches_X = AwayClub_matches.drop(columns=['Away Full-T Goals'],axis=1)

    if MatchSide=='Home' :
        X = HomeClub_matches_X
        y = HomeClub_matches['Home Full-T Goals']
    else:
        X = AwayClub_matches_X
        y = AwayClub_matches['Away Full-T Goals']

    return X,y

#X,y = GetClubXColumn('Away','Gent')

# Assuming you already have the features (X) and target (y) set up
def trainModel(MatchSide,club):

    X,y = GetClubXColumn(MatchSide,club)
    # Split the data into training and testing sets
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    
    xgb_model = XGBRegressor(n_estimators=100, learning_rate=0.1, random_state=42)

    # Initialize XGBRegressor for regression task
    #xgb_model = XGBRegressor()

    # Train the model
    xgb_model.fit(X_train, y_train)

    # Make predictions
    y_pred = xgb_model.predict(X_test)

    # Round the predictions to get integer values for goals (since goals are typically whole numbers)
    y_pred_rounded = [round(goal) for goal in y_pred]

    # Evaluate the model with Mean Squared Error and Mean Absolute Error
    mse = mean_squared_error(y_test, y_pred_rounded)
    mae = mean_absolute_error(y_test, y_pred_rounded)

    print(f"Mean Squared Error: {mse:.2f}")
    print(f"Mean Absolute Error: {mae:.2f}")


    return mae,xgb_model


def calculate_mean_columns(clubEnc, side):
    """Calculate the mean of other columns 
    """
    if side == "Home":
        # Filter the dataframe for home games
        Columns = ['HomeTeamEnc','Home Shots','Home Shots Target', 'Home Fouls', 
        'Home Corners',  'Home Yellow Cards',
        'Home Red Cards','Home Half-T Goals']
        club_matches = dfNumeric[dfNumeric['HomeTeamEnc'] == clubEnc][Columns]

    elif side == "Away":
        Columns = ['AwayTeamEnc','Away Shots', 'Away Shots Target', 'Away Fouls', 
        'Away Corners', 'Away Yellow Cards',
        'Away Red Cards','Away Half-T Goals' ]
        club_matches = dfNumeric[dfNumeric['AwayTeamEnc'] == clubEnc][Columns]

    else :
        raise ValueError("ClubSide must be either 'Home' or 'Away'")

    # Calculate the average of the specified columns
    averages = {}
    for column in Columns:
        if column in club_matches.columns:
            mean_value = club_matches[column].mean(skipna=True)
            # Handle NaN values by replacing them with 0
            if pd.isna(mean_value):
                averages[column] = 0
            else:
                averages[column] = round(mean_value)
        else:
            averages[column] = 0  # Handle missing columns if any
    return averages

#averages = calculate_averages('Gent',"Away")

#-------------------------------------------------------

def ExtractAvgGoal(teamSide,encoded_team_value) : 

    line = dfNumeric[dfNumeric['HomeTeamEnc']==encoded_team_value ]
    line = line.iloc[0]

    if teamSide == "Away" : 
        manual_averages = line[['Away_avg5Goal', 'Away_avg10Goal', 'Away_avg20Goal']].to_dict()
        return manual_averages
    else : 
        manual_averages = line[['Home_avg5Goal', 'Home_avg10Goal', 'Home_avg20Goal']].to_dict()
        return manual_averages


def create_feature_vector(averages, manual_averages, encoded_team_value, side):

    columns = [
        'Shots', 'Shots Target', 'Fouls', 'Corners', 'Yellow Cards', 'Red Cards', 'Half-T Goal'
    ]
    
    feature_keys = [f'{side} {col}' for col in columns]

    # Build the feature vector dynamically using loops
    feature_vector = [encoded_team_value]  # Start with the encoded team value
    # Add the features from averages dictionary
    for key in feature_keys:
        feature_vector.append(averages.get(key, 0))  # Add feature from averages, default to 0
    
    # Add the manual averages
    if side =='Home' : 
        columnsAvg = ['Home_avg5Goal', 'Home_avg10Goal', 'Home_avg20Goal']
    else :
        columnsAvg = ['Away_avg5Goal', 'Away_avg10Goal', 'Away_avg20Goal']

    for key in columnsAvg:
        feature_vector.append(manual_averages.get(key, 0))  # Add manual averages, default to 0

    # Convert the list to numpy array
    feature_vector = np.array(feature_vector)

    return feature_vector





#---------------------charts---------------------------------------

def ClubAwayHomediStat(clubNumb)->dict :


    #172 matchs, Away  = 87(31 win,22 draw,34lose) , Home 85(39win,lose 26,draw 20)
    #HOME--------------------------------------------------------------------------------
    Df_total_home_club_matches =  dfNumeric[dfNumeric['HomeTeamEnc']== clubNumb]
    total_home_club_matches = len(Df_total_home_club_matches)
    #home_win
    Df_Home_win_club_matches =  dfNumeric[(dfNumeric['HomeTeamEnc']== clubNumb)&((dfNumeric['Full Result_enc']== 0))]
    home_win_club_matches_count= len(Df_Home_win_club_matches)
    #home_lost
    Df_Home_lost_club_matches =  dfNumeric[(dfNumeric['HomeTeamEnc']== clubNumb)&((dfNumeric['Full Result_enc']== 1))]
    home_lost_club_matches_count= len(Df_Home_lost_club_matches)
    #home_draw
    Df_Home_draw_club_matches =  dfNumeric[(dfNumeric['HomeTeamEnc']== clubNumb)&((dfNumeric['Full Result_enc']== 2))]
    home_draw_club_matches_count= len(Df_Home_draw_club_matches)

    #Away--------------------------------------------------------------------------------

    Df_total_away_club_matches =  dfNumeric[dfNumeric['AwayTeamEnc']== clubNumb]
    total_away_club_matches = len(Df_total_away_club_matches)
    #home_win
    Df_away_win_club_matches =  dfNumeric[(dfNumeric['AwayTeamEnc']== clubNumb)&((dfNumeric['Full Result_enc']== 1))]
    away_win_club_matches_count= len(Df_away_win_club_matches)
    #home_lost
    Df_away_lost_club_matches =  dfNumeric[(dfNumeric['AwayTeamEnc']== clubNumb)&((dfNumeric['Full Result_enc']== 0))]
    away_lost_club_matches_count= len(Df_away_lost_club_matches)
    #home_draw
    Df_away_draw_club_matches =  dfNumeric[(dfNumeric['AwayTeamEnc']== clubNumb)&((dfNumeric['Full Result_enc']== 2))]
    away_draw_club_matches_count= len(Df_away_draw_club_matches)


    print("total_home_club_matches",total_home_club_matches,"total_away_club_matches",total_away_club_matches)

    wining_home_rate =  (home_win_club_matches_count / total_home_club_matches)
    wining_away_rate =  (away_win_club_matches_count / total_away_club_matches)

    losing_home_rate = (home_lost_club_matches_count / total_home_club_matches)
    losing_away_rate = (away_lost_club_matches_count / total_away_club_matches)

    draw_home_rate = (home_draw_club_matches_count / total_home_club_matches)
    draw_away_rate = (away_draw_club_matches_count / total_away_club_matches)

    #Df_Home_win_club_matches[['HomeTeamEnc','AwayTeamEnc','Home Full-T Goals','Away Full-T Goals','Full Result_enc']]

    return {
        'wining_home_rate' :wining_home_rate,
        'losing_home_rate': losing_home_rate,
        'draw_home_rate':draw_home_rate,
        'wining_away_rate' : wining_away_rate,
        'losing_away_rate':losing_away_rate,
        'draw_away_rate':draw_away_rate
    }

def generateRadarChart(clubNumb) :
    # Split the keys and values
    stats = ClubAwayHomediStat(clubNumb)

    labels = list(stats.keys())
    values = list(map(float, stats.values()))  # Convert string values to floats

    # Add the first value to the end to close the radar chart
    values += values[:1]

    # Create the angles for the radar chart
    angles = np.linspace(0, 2 * np.pi, len(labels), endpoint=False).tolist()
    angles += angles[:1]

    # Initialize the radar chart
    fig, ax = plt.subplots(figsize=(5, 3), subplot_kw=dict(polar=True))

    # Plot the radar chart
    ax.fill(angles, values, color='b', alpha=0.25)
    ax.plot(angles, values, color='b', linewidth=2)

    # Add labels
    ax.set_yticklabels([])
    ax.set_xticks(angles[:-1])
    ax.set_xticklabels(labels, fontsize=6)

    # Title and show the plot
#    ax.title('Club Performance Stats (Home vs. Away)')
    ax.set_title('Club Performance Stats (Home vs. Away)', fontsize=8)

    st.pyplot(fig)


def GenerateTotalGoalsTeamsBarChart(home_club,away_club):

    matchesBTTeams = dfNumeric[((dfNumeric['HomeTeamEnc'] ==home_club) | (dfNumeric['AwayTeamEnc'] ==home_club))&((dfNumeric['HomeTeamEnc'] ==away_club) | (dfNumeric['AwayTeamEnc'] ==away_club)) ] 

    #total Goals of both clubs
    N_matches = len(matchesBTTeams)


    # Data (Total goals for home and away teams)
    HomeTotalGoal = matchesBTTeams['Home Full-T Goals'].sum()
    AwayTotalGoal = matchesBTTeams['Away Full-T Goals'].sum()

    # Labels and values for the bar chart
    teams = ['Home Team', 'Away Team']
    total_goals = [HomeTotalGoal, AwayTotalGoal]

    # Create the bar chart

    fig, ax = plt.subplots(figsize=(4, 2))
    ax.bar(teams, total_goals, color=['green', 'red'])

    # Add title and labels
    ax.set_title(f'Total Goals for Both Teams in the Last {N_matches} Matches')
    ax.set_ylabel('Total Goals')
    # Add title and labels
    plt.title(f'Total both teams Goals of the last {N_matches} matches')
    plt.ylabel('Total Goals')

    # Show the chart
    st.pyplot(fig)

    
def generateWinLosePieChart(homeClub, awayClub):
    # Filter the dataframe for matches between the two clubs
    matchesBTTeams = dfNumeric[
        ((dfNumeric['HomeTeamEnc'] == homeClub) | (dfNumeric['AwayTeamEnc'] == homeClub)) &
        ((dfNumeric['HomeTeamEnc'] == awayClub) | (dfNumeric['AwayTeamEnc'] == awayClub))
    ]

    # Home winning
    homeTotalWinning = matchesBTTeams[matchesBTTeams['Full Result_enc'] == 0]
    N_match_Home_won = len(homeTotalWinning)

    # Away winning
    awayTotalWinning = matchesBTTeams[matchesBTTeams['Full Result_enc'] == 1]
    N_match_Away_won = len(awayTotalWinning)

    # Draw matches
    dfTotalDraw = matchesBTTeams[matchesBTTeams['Full Result_enc'] == 2]
    N_draw_match = len(dfTotalDraw)

    # Data for pie chart
    teams = ['Home Team Wins', 'Away Team Wins', 'Draws']
    total_wins = [N_match_Home_won, N_match_Away_won, N_draw_match]

    # Check if any value is NaN or all values are zero
    if np.isnan(total_wins).any():
        st.error("Error: One or more values are NaN.")
        return
    if sum(total_wins) == 0:
        st.error("Error: No matches found between the selected teams.")
        return

    # Create the pie chart
    fig, ax = plt.subplots(figsize=(4, 2), dpi=500)
    ax.pie(total_wins, labels=teams, autopct='%1.1f%%', colors=['green', 'red', 'blue'],textprops={'fontsize': 5}, startangle=90)
    
    # Add title
    ax.set_title(f'Win/Draw Distribution in Last {sum(total_wins)} Matches',fontsize=5)

    # Equal aspect ratio ensures that pie is drawn as a circle.
    ax.axis('equal')

    # Display chart in Streamlit
    st.pyplot(fig)



def GetClubWinningRate(Encoded_club,side):
    side = 'HomeTeamEnc' if side == 'Home' else 'AwayTeamEnc'

    dataFrame = dfNumeric[(dfNumeric[side] == Encoded_club)][['Home Full-T Goals','Away Full-T Goals','Full Result_enc','HomeTeamEnc','AwayTeamEnc']]
    if side == 'HomeTeamEnc':
        Full_Result_enc	= 0
    elif side == 'AwayTeamEnc':
        Full_Result_enc	= 1
    
    return round(len(dataFrame[dataFrame['Full Result_enc']==Full_Result_enc])/len(dataFrame),2)



def GetImagePathClub(Encoded_club,side):
    winingRate= GetClubWinningRate(Encoded_club,side)

    if winingRate < 0.2 : 
        st.write(f"my winning rate at {side} is",winingRate)
        return '../assets/img/1.png'
    elif winingRate >= 0.2 and winingRate < 0.3   :
        st.write(f"my winning rate at {side} is ",winingRate)
        return '../assets/img/2.jpg'
    elif winingRate >= 0.3 and winingRate < 0.4   :
        st.write(f"my winning rate at {side} is ",winingRate)
        return '../assets/img/3.png'
    elif winingRate >= 0.4 and winingRate < 0.5   :
        st.write(f"my winning rate at {side} is ",winingRate)
        return '../assets/img/4.jpg'
    else:
        st.write(f"my winning rate at {side} is ",winingRate)
        return '../assets/img/5.jpg'



#-------------------------------------------------------------------------------------------
#-------------------------------------------------------------------------------------------
#here
# Streamlit UI


st.markdown("""
    <style>
    .stApp {
        background-color: black;
    }
    .st-emotion-cache-1dp5vir {
            visibility : hidden
        }

    header {
           background-color: green !important; 
            }

            
    h1{
        color: green !important        
    }
    .st-emotion-cache-ue6h4q{
            color : green !important
    }
    .st-emotion-cache-uzeiqp p{
        color : green !important        
    }
    </style>
        
    """, unsafe_allow_html=True)



st.image('../assets/img/pepe.png')
st.title('Football Match Goal Prediction')
# Home team selection

col1, col2 = st.columns(2)

# Selectbox with the placeholder as the default value

# Only show the image if a valid club is selected (not the placeholder)

with col1:
    home_club = st.selectbox('Select Home Club', clubs_selectBox)
    encoded_Hometeam_value = GetClubEncodedValue(home_club)
    if home_club != 'Select a club...' : 
        st.image(GetImagePathClub(encoded_Hometeam_value,'Home'), caption=f"You selected: {home_club}")
    # Away club selection
with col2:
    away_club = st.selectbox('Select Away Club', clubs_selectBox)
    encoded_Awayteam_value = GetClubEncodedValue(away_club)
    if away_club != 'Select a club...' : 
        st.image(GetImagePathClub(encoded_Awayteam_value,'Away'),  caption=f"You selected: {away_club}")



# Show predictions and charts
if st.button("Play"):
    # Execute the prediction process
  
    averagesHome = calculate_mean_columns(encoded_Hometeam_value, 'Home')
    averagesAway = calculate_mean_columns(encoded_Awayteam_value, 'Away')

    manual_HomeAverages = ExtractAvgGoal('Home', encoded_Hometeam_value)
    manual_AwayAverages = ExtractAvgGoal('Away', encoded_Awayteam_value)
    
    # Create feature vector
    featureHome_vector = create_feature_vector(averagesHome, manual_HomeAverages, encoded_Hometeam_value, 'Home')
    featureAway_vector = create_feature_vector(averagesAway, manual_AwayAverages, encoded_Awayteam_value, 'Away')
    
    # Train and predict using the model
    homeMae,xgb_Homemodel = trainModel('Home', home_club)
    awayMae,xgb_Awaymodel = trainModel('Away', away_club)
    
    # Mock prediction (since we haven't trained the real model)
    predictedHome_goals = xgb_Homemodel.predict(featureHome_vector.reshape(1, -1))
    predictedAway_goals = xgb_Awaymodel.predict(featureAway_vector.reshape(1, -1))

    predicted_Homegoals_rounded = round(predictedHome_goals[0])
    predicted_Awaygoals_rounded = round(predictedAway_goals[0])
    
    # Display prediction result
    st.write(f" {home_club} ('Home'): {predicted_Homegoals_rounded}")
    st.write(f" {away_club} ('Away'): {predicted_Awaygoals_rounded}")

    generateRadarChart(encoded_Hometeam_value)
    generateRadarChart(encoded_Awayteam_value)
    GenerateTotalGoalsTeamsBarChart(encoded_Hometeam_value,encoded_Awayteam_value)
    generateWinLosePieChart(encoded_Awayteam_value,encoded_Hometeam_value) 
