import os
import streamlit as st
import pandas as pd
import numpy as np
import torch
import torch.nn as nn
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split

# Define the neural network
class Net(nn.Module):
    def __init__(self, input_size, dropout_rate=0.3):
        super(Net, self).__init__()
        self.fc1 = nn.Linear(input_size, 64)
        self.bn1 = nn.BatchNorm1d(64)
        self.dropout1 = nn.Dropout(dropout_rate)
        self.fc2 = nn.Linear(64, 32)
        self.bn2 = nn.BatchNorm1d(32)
        self.dropout2 = nn.Dropout(dropout_rate)
        self.fc3 = nn.Linear(32, 16)
        self.bn3 = nn.BatchNorm1d(16)
        self.dropout3 = nn.Dropout(dropout_rate)
        self.fc4 = nn.Linear(16, 1)

    def forward(self, x):
        x = torch.relu(self.fc1(x))
        x = self.bn1(x)
        x = self.dropout1(x)
        x = torch.relu(self.fc2(x))
        x = self.bn2(x)
        x = self.dropout2(x)
        x = torch.relu(self.fc3(x))
        x = self.bn3(x)
        x = self.dropout3(x)
        x = torch.sigmoid(self.fc4(x))
        return x

# Load or train the model
def load_or_train_model(input_size, X_train, y_train):
    model_path = 'nba_model.pth'
    # if os.path.exists(model_path):
    #     model = Net(X_train.shape[1])#Net(input_size)
    #     try:
    #         model.load_state_dict(torch.load(model_path))
    #         model.eval()
    #         return model
    #     except RuntimeError as e:
    #         st.write("Model input size mismatch, retraining the model.")
    #         os.remove(model_path)
    
    # Train the model if it doesn't exist or if there's a size mismatch
    model = Net(input_size)
    optimizer = torch.optim.Adam(model.parameters(), lr=0.001, weight_decay=0.001)
    criterion = nn.BCELoss()

    train_dataset = torch.utils.data.TensorDataset(torch.tensor(X_train, dtype=torch.float32), torch.tensor(y_train.values, dtype=torch.float32).view(-1, 1))
    train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=32, shuffle=True)
    
    model.train()
    for epoch in range(50):
        running_loss = 0.0
        for inputs, labels in train_loader:
            optimizer.zero_grad()
            outputs = model(inputs)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()
            running_loss += loss.item()
        #st.write(f"Epoch {epoch+1}/50, Loss: {running_loss/len(train_loader)}")

    torch.save(model.state_dict(), model_path)
    model.eval()
    return model

possible_teams = pd.read_csv('team_groups.csv')

def get_team_id(team_city):
    return possible_teams[possible_teams['TEAM_CITY'] == team_city]['TEAM_ID'].values[0]

# Load the dataset
@st.cache_data
def load_data():
    df = pd.read_csv('with_city.csv').drop(columns=['Unnamed: 0'])
    return df

df = load_data()

# Input for prediction

city1 = st.selectbox(label='Team 1', options=possible_teams['TEAM_CITY'])
team1_id = get_team_id(city1)

inputtington = df.loc[df['TEAM_ID'] == team1_id][-1:][['cum_points_avg', 'CUM_AST_AVG', 'CUM_REB_AVG', 'CUM_TS_PCT_AVG',
       'CUM_USG_PCT_AVG', 'CUM_OFF_RATING_AVG', 'CUM_DEF_RATING_AVG',
       'CUM_PIE_AVG', 'CUM_PACE_AVG', 'CUM_MIN_AVG', 'REST_DAYS',
       'TEAM_ABBREVIATION_ATL', 'TEAM_ABBREVIATION_BKN',
       'TEAM_ABBREVIATION_BOS', 'TEAM_ABBREVIATION_CHA',
       'TEAM_ABBREVIATION_CHI', 'TEAM_ABBREVIATION_CLE',
       'TEAM_ABBREVIATION_DAL', 'TEAM_ABBREVIATION_DEN',
       'TEAM_ABBREVIATION_DET', 'TEAM_ABBREVIATION_GSW',
       'TEAM_ABBREVIATION_HOU', 'TEAM_ABBREVIATION_IND',
       'TEAM_ABBREVIATION_LAC', 'TEAM_ABBREVIATION_LAL',
       'TEAM_ABBREVIATION_MEM', 'TEAM_ABBREVIATION_MIA',
       'TEAM_ABBREVIATION_MIL', 'TEAM_ABBREVIATION_MIN',
       'TEAM_ABBREVIATION_NOP', 'TEAM_ABBREVIATION_NYK',
       'TEAM_ABBREVIATION_OKC', 'TEAM_ABBREVIATION_ORL',
       'TEAM_ABBREVIATION_PHI', 'TEAM_ABBREVIATION_PHX',
       'TEAM_ABBREVIATION_POR', 'TEAM_ABBREVIATION_SAC',
       'TEAM_ABBREVIATION_SAS', 'TEAM_ABBREVIATION_TOR',
       'TEAM_ABBREVIATION_UTA', 'TEAM_ABBREVIATION_WAS', 'OFF_RATING', 'DEF_RATING',
       'NET_RATING', 'HOME_ENC', 'CAREER_PTS', 'CAREER_AST', 'CAREER_REB', 'CAREER_STL',
       'CAREER_MIN', 'WAR', 'WAR/82']].reset_index().drop(columns='index')

city2 = st.selectbox(label='Team 2', options=possible_teams['TEAM_CITY'])
team2_id = get_team_id(city2)

second = df.loc[df['TEAM_ID'] == team2_id][-1:][['cum_points_avg', 'CUM_AST_AVG', 'CUM_REB_AVG', 'CUM_TS_PCT_AVG',
       'CUM_USG_PCT_AVG', 'CUM_OFF_RATING_AVG', 'CUM_DEF_RATING_AVG',
       'CUM_PIE_AVG', 'CUM_PACE_AVG', 'CUM_MIN_AVG', 'REST_DAYS',
       'TEAM_ABBREVIATION_ATL', 'TEAM_ABBREVIATION_BKN',
       'TEAM_ABBREVIATION_BOS', 'TEAM_ABBREVIATION_CHA',
       'TEAM_ABBREVIATION_CHI', 'TEAM_ABBREVIATION_CLE',
       'TEAM_ABBREVIATION_DAL', 'TEAM_ABBREVIATION_DEN',
       'TEAM_ABBREVIATION_DET', 'TEAM_ABBREVIATION_GSW',
       'TEAM_ABBREVIATION_HOU', 'TEAM_ABBREVIATION_IND',
       'TEAM_ABBREVIATION_LAC', 'TEAM_ABBREVIATION_LAL',
       'TEAM_ABBREVIATION_MEM', 'TEAM_ABBREVIATION_MIA',
       'TEAM_ABBREVIATION_MIL', 'TEAM_ABBREVIATION_MIN',
       'TEAM_ABBREVIATION_NOP', 'TEAM_ABBREVIATION_NYK',
       'TEAM_ABBREVIATION_OKC', 'TEAM_ABBREVIATION_ORL',
       'TEAM_ABBREVIATION_PHI', 'TEAM_ABBREVIATION_PHX',
       'TEAM_ABBREVIATION_POR', 'TEAM_ABBREVIATION_SAC',
       'TEAM_ABBREVIATION_SAS', 'TEAM_ABBREVIATION_TOR',
       'TEAM_ABBREVIATION_UTA', 'TEAM_ABBREVIATION_WAS', 'OFF_RATING', 'DEF_RATING',
       'NET_RATING', 'CAREER_PTS', 'CAREER_AST', 'CAREER_REB', 'CAREER_STL',
       'CAREER_MIN', 'WAR', 'WAR/82']].reset_index().drop(columns='index')

second = second.add_prefix('OPP_')

inputtington = pd.concat([inputtington, second], axis=1)

input_data = inputtington.drop(columns=['TEAM_ABBREVIATION_ATL', 'TEAM_ABBREVIATION_BKN',
       'TEAM_ABBREVIATION_BOS', 'TEAM_ABBREVIATION_CHA',
       'TEAM_ABBREVIATION_CHI', 'TEAM_ABBREVIATION_CLE',
       'TEAM_ABBREVIATION_DAL', 'TEAM_ABBREVIATION_DEN',
       'TEAM_ABBREVIATION_DET', 'TEAM_ABBREVIATION_GSW',
       'TEAM_ABBREVIATION_HOU', 'TEAM_ABBREVIATION_IND',
       'TEAM_ABBREVIATION_LAC', 'TEAM_ABBREVIATION_LAL',
       'TEAM_ABBREVIATION_MEM', 'TEAM_ABBREVIATION_MIA',
       'TEAM_ABBREVIATION_MIL', 'TEAM_ABBREVIATION_MIN',
       'TEAM_ABBREVIATION_NOP', 'TEAM_ABBREVIATION_NYK',
       'TEAM_ABBREVIATION_OKC', 'TEAM_ABBREVIATION_ORL',
       'TEAM_ABBREVIATION_PHI', 'TEAM_ABBREVIATION_PHX',
       'TEAM_ABBREVIATION_POR', 'TEAM_ABBREVIATION_SAC',
       'TEAM_ABBREVIATION_SAS', 'TEAM_ABBREVIATION_TOR',
       'TEAM_ABBREVIATION_UTA', 'TEAM_ABBREVIATION_WAS', 'OPP_TEAM_ABBREVIATION_ATL',
       'OPP_TEAM_ABBREVIATION_BKN', 'OPP_TEAM_ABBREVIATION_BOS',
       'OPP_TEAM_ABBREVIATION_CHA', 'OPP_TEAM_ABBREVIATION_CHI',
       'OPP_TEAM_ABBREVIATION_CLE', 'OPP_TEAM_ABBREVIATION_DAL',
       'OPP_TEAM_ABBREVIATION_DEN', 'OPP_TEAM_ABBREVIATION_DET',
       'OPP_TEAM_ABBREVIATION_GSW', 'OPP_TEAM_ABBREVIATION_HOU',
       'OPP_TEAM_ABBREVIATION_IND', 'OPP_TEAM_ABBREVIATION_LAC',
       'OPP_TEAM_ABBREVIATION_LAL', 'OPP_TEAM_ABBREVIATION_MEM',
       'OPP_TEAM_ABBREVIATION_MIA', 'OPP_TEAM_ABBREVIATION_MIL',
       'OPP_TEAM_ABBREVIATION_MIN', 'OPP_TEAM_ABBREVIATION_NOP',
       'OPP_TEAM_ABBREVIATION_NYK', 'OPP_TEAM_ABBREVIATION_OKC',
       'OPP_TEAM_ABBREVIATION_ORL', 'OPP_TEAM_ABBREVIATION_PHI',
       'OPP_TEAM_ABBREVIATION_PHX', 'OPP_TEAM_ABBREVIATION_POR',
       'OPP_TEAM_ABBREVIATION_SAC', 'OPP_TEAM_ABBREVIATION_SAS',
       'OPP_TEAM_ABBREVIATION_TOR', 'OPP_TEAM_ABBREVIATION_UTA',
       'OPP_TEAM_ABBREVIATION_WAS'])

# Standardize the data
scaler = StandardScaler()
X = df.drop(columns=['WIN', 'GAME_ID', 'TEAM_ID', 'GAME_DATE', 'MATCHUP',
                          'PTS', 'OPP_PTS', 'WIN', 'TEAM_CITY', 'TEAM_NAME', 'DATE', 'HOME_TEAM', 'AT_HOME',
                          'TEAM_ABBREVIATION_ATL', 'TEAM_ABBREVIATION_BKN',
       'TEAM_ABBREVIATION_BOS', 'TEAM_ABBREVIATION_CHA',
       'TEAM_ABBREVIATION_CHI', 'TEAM_ABBREVIATION_CLE',
       'TEAM_ABBREVIATION_DAL', 'TEAM_ABBREVIATION_DEN',
       'TEAM_ABBREVIATION_DET', 'TEAM_ABBREVIATION_GSW',
       'TEAM_ABBREVIATION_HOU', 'TEAM_ABBREVIATION_IND',
       'TEAM_ABBREVIATION_LAC', 'TEAM_ABBREVIATION_LAL',
       'TEAM_ABBREVIATION_MEM', 'TEAM_ABBREVIATION_MIA',
       'TEAM_ABBREVIATION_MIL', 'TEAM_ABBREVIATION_MIN',
       'TEAM_ABBREVIATION_NOP', 'TEAM_ABBREVIATION_NYK',
       'TEAM_ABBREVIATION_OKC', 'TEAM_ABBREVIATION_ORL',
       'TEAM_ABBREVIATION_PHI', 'TEAM_ABBREVIATION_PHX',
       'TEAM_ABBREVIATION_POR', 'TEAM_ABBREVIATION_SAC',
       'TEAM_ABBREVIATION_SAS', 'TEAM_ABBREVIATION_TOR',
       'TEAM_ABBREVIATION_UTA', 'TEAM_ABBREVIATION_WAS', 'OPP_TEAM_ABBREVIATION_ATL',
       'OPP_TEAM_ABBREVIATION_BKN', 'OPP_TEAM_ABBREVIATION_BOS',
       'OPP_TEAM_ABBREVIATION_CHA', 'OPP_TEAM_ABBREVIATION_CHI',
       'OPP_TEAM_ABBREVIATION_CLE', 'OPP_TEAM_ABBREVIATION_DAL',
       'OPP_TEAM_ABBREVIATION_DEN', 'OPP_TEAM_ABBREVIATION_DET',
       'OPP_TEAM_ABBREVIATION_GSW', 'OPP_TEAM_ABBREVIATION_HOU',
       'OPP_TEAM_ABBREVIATION_IND', 'OPP_TEAM_ABBREVIATION_LAC',
       'OPP_TEAM_ABBREVIATION_LAL', 'OPP_TEAM_ABBREVIATION_MEM',
       'OPP_TEAM_ABBREVIATION_MIA', 'OPP_TEAM_ABBREVIATION_MIL',
       'OPP_TEAM_ABBREVIATION_MIN', 'OPP_TEAM_ABBREVIATION_NOP',
       'OPP_TEAM_ABBREVIATION_NYK', 'OPP_TEAM_ABBREVIATION_OKC',
       'OPP_TEAM_ABBREVIATION_ORL', 'OPP_TEAM_ABBREVIATION_PHI',
       'OPP_TEAM_ABBREVIATION_PHX', 'OPP_TEAM_ABBREVIATION_POR',
       'OPP_TEAM_ABBREVIATION_SAC', 'OPP_TEAM_ABBREVIATION_SAS',
       'OPP_TEAM_ABBREVIATION_TOR', 'OPP_TEAM_ABBREVIATION_UTA',
       'OPP_TEAM_ABBREVIATION_WAS'])

y = df['WIN']  # Assuming 'WIN' is the target column
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
scaler.fit_transform(X_train)
input_data = scaler.fit_transform(input_data)

# Convert to tensor
input_tensor = torch.tensor(input_data, dtype=torch.float32)

# Load or train the model and make a prediction
if st.button("Predict"):
    
    model = load_or_train_model(input_tensor.shape[1], scaler.fit_transform(X_train), y_train)
    with torch.no_grad():
        prediction = model(input_tensor)
        prediction = prediction.item()
        st.write(f"Prediction: {prediction * 100}%")

# Display dataset
st.write("Dataset preview:")
st.write(df.head())