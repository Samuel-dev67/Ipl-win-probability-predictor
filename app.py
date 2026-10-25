import streamlit as st
import pickle
import pandas as pd

teams = [
    'Sunrisers Hyderabad',
    'Mumbai Indians',
    'Royal Challengers Bangalore',
    'Kolkata Knight Riders',
    'Kings XI Punjab',
    'Chennai Super Kings',
    'Rajasthan Royals',
    'Delhi Capitals'
]

cities = [
    'Hyderabad', 'Bangalore', 'Mumbai', 'Indore', 'Kolkata', 'Delhi',
    'Chandigarh', 'Jaipur', 'Chennai', 'Cape Town', 'Port Elizabeth',
    'Durban', 'Centurion', 'East London', 'Johannesburg', 'Kimberley',
    'Bloemfontein', 'Ahmedabad', 'Cuttack', 'Nagpur', 'Dharamsala',
    'Visakhapatnam', 'Pune', 'Raipur', 'Ranchi', 'Abu Dhabi',
    'Sharjah', 'Mohali', 'Bengaluru'
]

# Load trained ML pipeline
pipe = pickle.load(open('pipe.pkl', 'rb'))

st.title('IPL Win Predictor')

# Modern Streamlit API
col1, col2 = st.columns(2)

with col1:
    batting_team = st.selectbox(
        'Select the batting team',
        sorted(teams)
    )

with col2:
    bowling_team = st.selectbox(
        'Select the bowling team',
        sorted(teams)
    )

selected_city = st.selectbox(
    'Select host city',
    sorted(cities)
)

target = st.number_input(
    'Target',
    min_value=1,
    step=1
)

# Modern Streamlit API
col3, col4, col5 = st.columns(3)

with col3:
    score = st.number_input(
        'Score',
        min_value=0,
        step=1
    )

with col4:
    overs = st.number_input(
        'Overs completed',
        min_value=0.0,
        max_value=20.0,
        step=0.1
    )

with col5:
    wickets_out = st.number_input(
        'Wickets out',
        min_value=0,
        max_value=10,
        step=1
    )

if st.button('Predict Probability'):

    # Basic validation
    if batting_team == bowling_team:
        st.error('Batting team and bowling team cannot be the same.')

    elif score > target:
        st.error('Current score cannot be greater than the target.')

    elif overs <= 0:
        st.error('Overs completed must be greater than 0.')

    elif overs >= 20:
        st.error('The innings is already complete.')

    else:

        runs_left = target - score
        balls_left = 120 - (overs * 6)

        wickets_remaining = 10 - wickets_out

        # Current Run Rate
        crr = score / overs

        # Required Run Rate
        rrr = (runs_left * 6) / balls_left

        # Create input DataFrame
        input_df = pd.DataFrame({
            'batting_team': [batting_team],
            'bowling_team': [bowling_team],
            'city': [selected_city],
            'runs_left': [runs_left],
            'balls_left': [balls_left],
            'wickets': [wickets_remaining],
            'total_runs_x': [target],
            'crr': [crr],
            'rrr': [rrr]
        })

        # Prediction
        result = pipe.predict_proba(input_df)

        loss = result[0][0]
        win = result[0][1]

        # Display probabilities
        st.header(
            batting_team + " - " + str(round(win * 100)) + "%"
        )

        st.header(
            bowling_team + " - " + str(round(loss * 100)) + "%"
        )