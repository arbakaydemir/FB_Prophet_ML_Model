# To import necessary library
import pandas as pd
import numpy as np
import os
import time
import warnings
from numpy import newaxis
import matplotlib.dates as mdates
import matplotlib.pyplot as plt
import plotly.graph_objects as go
from fbprophet import Prophet
import seaborn as sns


help(shape)

# to import datasets into memory
bitcoin_data = pd.read_csv(
    r'C:\Users\Arbak Aydemir\Desktop\Dersler\M.Sc. Business Analytics\MS5103 Business Analytics Project\2nd\Machine Learning Models for the Final Project\Facebook Prophet\Bitcoin1.csv')
ethereum_data = pd.read_csv(
    r'C:\Users\Arbak Aydemir\Desktop\Dersler\M.Sc. Business Analytics\MS5103 Business Analytics Project\2nd\Machine Learning Models for the Final Project\Facebook Prophet\Etherum1.csv')
bitcoincash_data = pd.read_csv(
    r'C:\Users\Arbak Aydemir\Desktop\Dersler\M.Sc. Business Analytics\MS5103 Business Analytics Project\2nd\Machine Learning Models for the Final Project\Facebook Prophet\Bitcoincash1.csv')
xrpripple_data = pd.read_csv(
    r'C:\Users\Arbak Aydemir\Desktop\Dersler\M.Sc. Business Analytics\MS5103 Business Analytics Project\2nd\Machine Learning Models for the Final Project\Facebook Prophet\xrpripple1.csv')
litecoin_data = pd.read_csv(
    r'C:\Users\Arbak Aydemir\Desktop\Dersler\M.Sc. Business Analytics\MS5103 Business Analytics Project\2nd\Machine Learning Models for the Final Project\Facebook Prophet\litecoin1.csv')
tether_data = pd.read_csv(
    r'C:\Users\Arbak Aydemir\Desktop\Dersler\M.Sc. Business Analytics\MS5103 Business Analytics Project\2nd\Machine Learning Models for the Final Project\Facebook Prophet\tether1.csv')

# to create a dictionary for the each dataset
cryptocurrencies_dict = {
    "bitcoin": bitcoin_data,
    "ethereum": ethereum_data,
    "bitcoincash": bitcoincash_data,
    "xrpripple": xrpripple_data,
    "litecoin": litecoin_data,
    "tether": tether_data
}

# To choose Date and Close** coloumns for each crypto data
bitcoin_data = cryptocurrencies_dict["bitcoin"][['Date', 'Close**']]
ethereum_data = cryptocurrencies_dict["ethereum"][['Date', 'Close**']]
bitcoincash_data = cryptocurrencies_dict["bitcoincash"][['Date', 'Close**']]
xrpripple_data = cryptocurrencies_dict["xrpripple"][['Date', 'Close**']]
litecoin_data = cryptocurrencies_dict["litecoin"][['Date', 'Close**']]
tether_data = cryptocurrencies_dict["tether"][['Date', 'Close**']]

bitcoin_data.head(5)

# To create a new version of dictionary
dateclose_dict = {
    "bitcoin": bitcoin_data,
    "ethereum": ethereum_data,
    "bitcoincash": bitcoincash_data,
    "xrpripple": xrpripple_data,
    "litecoin": litecoin_data,
    "tether": tether_data
}

dateclose_dict['bitcoin'].head(5)

########################################### DATA EXPLORATION ################################################

# To change data type of date from string to timestamp
bitcoin_data['Date'] = pd.to_datetime(bitcoin_data['Date'])
bitcoin_data.loc[0:, 'Date'] = pd.to_datetime(bitcoin_data['Date'])
bitcoin_data.head(5)

# To apply changing data type of Date coloumn to timestamp format for every currency by using for loop
for currency in dateclose_dict:
    dateclose_dict[currency].loc[0:, 'Date'] = pd.to_datetime(dateclose_dict[currency]['Date'])

# Set the DataFrame index using Date column
bitcoin_ts = dateclose_dict['bitcoin'].set_index('Date')
ethereum_ts = dateclose_dict['ethereum'].set_index('Date')
bitcoincash_ts = dateclose_dict['bitcoincash'].set_index('Date')
xrpripple_ts = dateclose_dict['xrpripple'].set_index('Date')
litecoin_ts = dateclose_dict['litecoin'].set_index('Date')
tether_ts = dateclose_dict['tether'].set_index('Date')

# To create a dictionary for updated version of datasets
dateclose_time_series = {
    "bitcoin": bitcoin_ts,
    "ethereum": ethereum_ts,
    "bitcoincash": bitcoincash_ts,
    "xrpripple": xrpripple_ts,
    "litecoin": litecoin_ts,
    "tether": tether_ts
}

# To create graphs of each currencies' close price distribution
for ts in dateclose_time_series:
    fig, ax = plt.subplots(figsize=(7, 4))
    plt.plot(dateclose_time_series[ts])
    plt.xlabel('Date', fontsize=12)
    plt.ylabel('Price', fontsize=12)
    title_str = "Closing price distribution of " + ts
    plt.title(title_str, fontsize=15)
    plt.show()

# Calculating rolling statistics to check for a trend/seasonality
for ts in dateclose_time_series:
    rolling_mean = dateclose_time_series[ts].rolling(window=20, center=False).mean()
    rolling_std = dateclose_time_series[ts].rolling(window=20, center=False).std()

    # Plot rolling statistics:
    fig, ax = plt.subplots(figsize=(8, 5))
    orig = plt.plot(dateclose_time_series[ts], color='blue', label='Original')
    mean = plt.plot(rolling_mean, color='red', label='Rolling Mean')
    std = plt.plot(rolling_std, color='black', label='Rolling Std')
    plt.legend(loc='best')
    plt.title('Rolling Mean & Standard Deviation for ' + ts)
    plt.show(block=False)

dateclose_time_series['bitcoin']['Close**'].hist()
plt.title('Data distribution for bitcoin')
plt.show()

dateclose_time_series['ethereum']['Close**'].hist()
plt.title('Data distribution for ethereum')
plt.show()

dateclose_time_series['bitcoincash']['Close**'].hist()
plt.title('Data distribution for bitcoincash')
plt.show()

dateclose_time_series['xrpripple']['Close**'].hist()
plt.title('Data distribution for xrpripple')
plt.show()

dateclose_time_series['litecoin']['Close**'].hist()
plt.title('Data distribution for litecoin')
plt.show()

dateclose_time_series['tether']['Close**'].hist()
plt.title('Data distribution for tether')
plt.show()

bitcoin_log = np.log(dateclose_time_series['bitcoin'])
bitcoin_log['Close**'].hist()
plt.title('Log Transformed Data distribution for bitcoin')
plt.show()


##########################################################################################################################
################################# MODEL FACEBOOK PROPHET ###############################################################
def split_data(data):
    splitIndex = int(np.floor(data.shape[0] * 0.70))
    trainDataset, testDataset = data[:splitIndex], data[splitIndex:]
    return (trainDataset, testDataset)


dataforFBPROPHET = dateclose_dict['bitcoin']
dataforFBPROPHET = dataforFBPROPHET.reset_index()

dataforFBPROPHETD_DATE_CLOSE = dataforFBPROPHET[['Date', 'Close**']]
dataforFBPROPHETD_DATE_CLOSE = dataforFBPROPHETD_DATE_CLOSE.rename(columns={"Date": "ds", "Close**": "y"})

dataforFBPROPHETD_DATE_CLOSE.head(5)

dataforFBPROPHETD_DATE_CLOSE['y_orig'] = dataforFBPROPHETD_DATE_CLOSE['y']
###To make close data logaritmic
dataforFBPROPHETD_DATE_CLOSE['y'] = np.log(dataforFBPROPHETD_DATE_CLOSE['y'])
# dataforFBPROPHETD_DATE_CLOSE.head(5)

splitIndex = int(np.floor(dataforFBPROPHETD_DATE_CLOSE.shape[0] * 0.70))
X_train_prophet, X_test_prophet = dataforFBPROPHETD_DATE_CLOSE[:splitIndex], dataforFBPROPHETD_DATE_CLOSE[splitIndex:]
print("No. of samples in the training set: ", len(X_train_prophet))
print("No. of samples in the test set", len(X_test_prophet))

# To choose daily, monthly, yearly prediction options for the model
model = Prophet(daily_seasonality=True, yearly_seasonality=True)
model.fit(X_train_prophet)

# To define that how many days will be predicted and create its dataframe
futureFORECAST_data = model.make_future_dataframe(periods=90)

# If we want we would not include the historical data in the prediction by using
# futureFORECAST_data = model.make_future_dataframe(periods = 365, freq = 'D', include_history=False)

# Show the dataframe for 90 days period
futureFORECAST_data.tail()

# To predict price for defined period with historical data
forecasting_data = model.predict(futureFORECAST_data)

# To show predicted values
forecasting_data.head(5)

# To show predicted values of defined period
# Predictedpricesforbitcoin = forecasting_data[['ds','yhat', 'yhat_lower', 'yhat_upper']]
# to show numberical representation of log predictions
# np.exp(Predictedpricesforbitcoin[['yhat', 'yhat_lower', 'yhat_upper']])

# To import predicted dataset into a seperated csv file
# Predictedpricesforbitcoin.to_csv(r'C:\Users\Arbak Aydemir\Desktop\Dersler\M.Sc. Business Analytics\MS5103 Business Analytics Project\2nd\Facebook Prophet\Results\Predictedpricesforbitcoin.csv')

# To make simple the result of the model
# simple_forecast = forecasting_data[['ds','yhat']]
# simple_forecast.head()
# simple_forecastfor3months = simple_forecast.tail(90)
# simple_forecastfor3months.to_csv(r'C:\Users\Arbak Aydemir\Desktop\Dersler\M.Sc. Business Analytics\MS5103 Business Analytics Project\2nd\Facebook Prophet\Results\90DayForecast.csv')


# to plot the model graphical visualization
model.plot(forecasting_data)
# to plot the components of the model
model.plot_components(forecasting_data)

# to test the model
from sklearn.metrics import mean_absolute_error, mean_squared_error

testdataframe = X_test_prophet
del testdataframe['y_orig']
testdataframe.set_index('ds')
test1 = model.predict(testdataframe)
test1.set_index('ds')

# Plot results :
neworiginal = testdataframe[['ds', 'y']].reset_index(drop=True)
neworiginalsorted = neworiginal.sort_values(by=['ds'], ascending=True).set_index('ds')
predicted = test1[['ds', 'yhat']].set_index('ds')
fig, ax = plt.subplots(figsize=(8, 5))
original_data = plt.plot(neworiginalsorted[['y']], color='blue', label='Original')
predicted_data = plt.plot(predicted[['yhat']], color='red', label='Predicted')
plt.legend(loc='best')
plt.title('Original and Predicted Price for Bitcoin ')
plt.show(block=False)

######################CHECK QUALITY OF THE MODEL##############################################
# To calculate mean squared error
MSE = mean_squared_error(np.exp(testdataframe['y']), np.exp(test1['yhat']))
print("Mean Squared Error: ", MSE)
# To calculate mean absolute error
MAE = mean_absolute_error(np.exp(testdataframe['y']), np.exp(test1['yhat']))
print("Mean Absolute Error: ", MAE)

# To calculate Root mean squared error
from math import sqrt

rms = sqrt(mean_squared_error(np.exp(testdataframe['y']), np.exp(test1['yhat'])))
print("Root Mean Squared Error:", rms)


# df = test1[['ds', 'yhat']].join(testdataframe['y_orig'])
# df.head()

# To create a function for the model
def fbProphetmodelforallcurrencies(data):
    dataProphet = data
    dataProphet = dataProphet.reset_index()
    dataforFBPROPHETD_DATE_CLOSE = dataProphet[['Date', 'Close**']]
    dataforFBPROPHETD_DATE_CLOSE = dataforFBPROPHETD_DATE_CLOSE.rename(columns={"Date": "ds", "Close**": "y"})
    dataforFBPROPHETD_DATE_CLOSE['y_orig'] = dataforFBPROPHETD_DATE_CLOSE['y']  # to save a copy of the original data
    # log transform y
    dataforFBPROPHETD_DATE_CLOSE['y'] = np.log(dataforFBPROPHETD_DATE_CLOSE['y'])
    splitIndex = int(np.floor(dataforFBPROPHETD_DATE_CLOSE.shape[0] * 0.70))
    X_train_prophet, X_test_prophet = dataforFBPROPHETD_DATE_CLOSE[:splitIndex], dataforFBPROPHETD_DATE_CLOSE[
                                                                                 splitIndex:]
    model = Prophet(yearly_seasonality=True, daily_seasonality=True)
    # model.fit(dataProphetRed)
    model.fit(X_train_prophet)
    test = X_test_prophet
    del test['y_orig']
    test.set_index('ds')
    prediction = model.predict(test)
    MSE = mean_squared_error(np.exp(test['y']), np.exp(prediction['yhat']))
    MAE = mean_absolute_error(np.exp(test['y']), np.exp(prediction['yhat']))
    return MSE, MAE


# To apply model function for every currency
for currency in cryptocurrencies_dict:
    original_data = dateclose_time_series[currency]
    mse = fbProphetmodelforallcurrencies(original_data)
    print("MSE and MAE using FB Prophet for " + currency + " :", mse)
