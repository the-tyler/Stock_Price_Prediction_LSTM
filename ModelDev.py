# IMPORT LIBRARIES

# Data Pre-processing & Extraction
import pandas as pd
import numpy as np
from datetime import datetime
from dateutil.relativedelta import relativedelta

from yahoo_fin import stock_info as yf
from sklearn.preprocessing import RobustScaler
from collections import deque

# Modelling 
import tensorflow as tf
from keras.models import Sequential
from keras.layers import Dense, LSTM, Dropout

# Visualization
import matplotlib.pyplot as plt

# Turn the chain assignment warning off 
pd.options.mode.chained_assignment = None


# DEFINE FUNCTIONS

# To extract the price data through Yahoo, 'Extract_Data' function is implemented. Default parameters are seen in the function definition
def Extract_Data(Stock = 'ROKU', year_back = 1, interval = '1d'):

  STOCK = Stock
  now = datetime.now() # Current date and time
  DATE_NOW = now.strftime('%Y-%m-%d') # Extract the today's date as a string 
  STARTING_DATE = (now - relativedelta(years=year_back)).strftime('%Y-%m-%d') # Extract the starting date as a string 

  raw_price_df = yf.get_data(STOCK, start_date = STARTING_DATE, end_date = DATE_NOW, interval = interval) # Extract the price data by yahoo finance

  return raw_price_df

# To visualize the extracted closing price data, 'Visualize_Data' function is implemented
def Visualize_Data(df, Stock, interval, figsize = (20,9)):

  plt.style.use(style='ggplot')
  plt.figure(figsize=figsize)
  plt.plot(df['close'])
  plt.xlabel('Date')
  plt.ylabel('Price in $')
  plt.legend([f'Price per {Stock} share'])
  plt.title(f'Share price of {Stock}')
  plt.show()
  
# To put the data set in the correct form for training, 'Prepare_Data' function is implemented
def Prepare_Data(dataframe, days):

  df = dataframe.copy()
  df['future'] = df['scaled_close'].shift(-days)
  last_sequence = np.array(df[['scaled_close']].tail(days))
  df.dropna(inplace=True)
  
  sequence_data = []
  sequences = deque(maxlen=NUMBER_of_STEPS_BACK)

  for entry, target in zip(df[['scaled_close','date']].values, df['future'].values):
      sequences.append(entry)
      if len(sequences) == NUMBER_of_STEPS_BACK:
          sequence_data.append([np.array(sequences), target])

  last_sequence = list([s[:1] for s in sequences]) + list(last_sequence)
  last_sequence = np.array(last_sequence).astype(np.float32)

  # build X and Y training set
  X, Y = [], []
  for seq, target in sequence_data:
      X.append(seq)
      Y.append(target)

  # convert X and Y to numpy arrays for compatibility
  X = np.array(X)
  Y = np.array(Y)

  return last_sequence, X, Y

# To train the one-of-a-kind LSTM model with set hyperparameters, 'Train_Model' function is implemented
def Train_Model(x_train, y_train, NUMBER_of_STEPS_BACK, BATCH_SIZE, UNITS, EPOCHS, DROPOUT, OPTIMIZER, LOSS):

  model = Sequential()

  model.add(LSTM(UNITS, return_sequences=True, input_shape=(NUMBER_of_STEPS_BACK, 1)))
  model.add(Dropout(DROPOUT))
  model.add(LSTM(UNITS, return_sequences=False))
  model.add(Dropout(DROPOUT))
  model.add(Dense(1)) # Makes sure that for each day, there is only one prediction

  model.compile(loss=LOSS, optimizer=OPTIMIZER)

  model.fit(x_train, y_train, epochs=EPOCHS, batch_size=BATCH_SIZE, verbose=1)

  model.summary()

  return model


# RUN THE MODEL

# Extract Data
STOCK = 'ROKU' # Specify the ticker symbol of the share
YEAR_BACK = 1 # Number of years of data back from today for extraction
INTERVAL = '1d' # The frequency of the extracted data. 1d: daily, 4h: 4 hourly

# Visualize Data
raw_price_df = Extract_Data(Stock = STOCK, year_back = YEAR_BACK, interval = INTERVAL )
Visualize_Data(raw_price_df, Stock = STOCK, interval = INTERVAL)

# Pre-process Data
raw_price_df = raw_price_df.drop(['open', 'high', 'low', 'adjclose', 'ticker', 'volume'], axis=1)
raw_price_df['date'] = raw_price_df.index
scaler = RobustScaler()
raw_price_df['scaled_close'] = scaler.fit_transform(np.expand_dims(raw_price_df['close'].values, axis=1))

#Set Parameters/Hyperparameters
NUMBER_of_STEPS_BACK = 30 # Number of days back that the model will be trained for
PREDICTION_STEPS = [1] # Number of days that the model will predict. To predict the next three days, modify it as follows: [1,2,3]
BATCH_SIZE = 16 # Number of training samples that will be passed to the network in one epoch
DROPOUT = 0.25 # Probability to exclude the input and recurrent connections to improve performance by regularization (25%)
UNITS = 60 # Number of neurons connected to the layer
EPOCHS = 10 # Number of times that the learning algorithm will work through the entire training set 
LOSS='mean_squared_error' # Methodology to measure the inaccuracy
OPTIMIZER='adam' # Optimizer used to iterate to better states

# Make Prediction
predictions = []

for step in PREDICTION_STEPS:
  last_sequence, x_train, y_train = Prepare_Data(raw_price_df, step)
  x_train = x_train[:, :, :1].astype(np.float32)

  model = Train_Model(x_train, y_train, NUMBER_of_STEPS_BACK, BATCH_SIZE, UNITS, EPOCHS, DROPOUT, OPTIMIZER, LOSS)
  
  last_sequence = last_sequence[-NUMBER_of_STEPS_BACK:]
  last_sequence = np.expand_dims(last_sequence, axis=0)
  prediction = model.predict(last_sequence)
  predicted_price = scaler.inverse_transform(prediction)[0][0]

  predictions.append(round(float(predicted_price), 2))
  
# Print Prediction
if len(predictions) > 0:
  predictions_list = [str(d)+'$' for d in predictions]
  predictions_str = ', '.join(predictions_list)
  message = f'{STOCK} share price prediction for the next day(s) {predictions_str}'
  print(message)
