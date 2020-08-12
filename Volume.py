# To import necessary library
import pandas as pd
import numpy as np
import os
import time
import warnings
from numpy import newaxis
import matplotlib.dates as mdates
import matplotlib.pyplot as plt
from fbprophet import Prophet
import seaborn as sns

# to import datasets into memory
bitcoin_data = pd.read_csv(
    r'C:\Users\Arbak Aydemir\Desktop\Dersler\M.Sc. Business Analytics\MS5103 Business Analytics Project\2nd\Facebook Prophet\Bitcoin1.csv')
ethereum_data = pd.read_csv(
    r'C:\Users\Arbak Aydemir\Desktop\Dersler\M.Sc. Business Analytics\MS5103 Business Analytics Project\2nd\Facebook Prophet\Etherum1.csv')
bitcoincash_data = pd.read_csv(
    r'C:\Users\Arbak Aydemir\Desktop\Dersler\M.Sc. Business Analytics\MS5103 Business Analytics Project\2nd\Facebook Prophet\Bitcoincash1.csv')
xrpripple_data = pd.read_csv(
    r'C:\Users\Arbak Aydemir\Desktop\Dersler\M.Sc. Business Analytics\MS5103 Business Analytics Project\2nd\Facebook Prophet\xrpripple1.csv')
litecoin_data = pd.read_csv(
    r'C:\Users\Arbak Aydemir\Desktop\Dersler\M.Sc. Business Analytics\MS5103 Business Analytics Project\2nd\Facebook Prophet\litecoin1.csv')
tether_data = pd.read_csv(
    r'C:\Users\Arbak Aydemir\Desktop\Dersler\M.Sc. Business Analytics\MS5103 Business Analytics Project\2nd\Facebook Prophet\tether1.csv')

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
bitcoin_data = cryptocurrencies_dict["bitcoin"][['Date', 'Volume']]
ethereum_data = cryptocurrencies_dict["ethereum"][['Date', 'Volume']]
bitcoincash_data = cryptocurrencies_dict["bitcoincash"][['Date', 'Volume']]
xrpripple_data = cryptocurrencies_dict["xrpripple"][['Date', 'Volume']]
litecoin_data = cryptocurrencies_dict["litecoin"][['Date', 'Volume']]
tether_data = cryptocurrencies_dict["tether"][['Date', 'Volume']]

dateclose_dict = {
    "bitcoin": bitcoin_data,
    "ethereum": ethereum_data,
    "bitcoincash": bitcoincash_data,
    "xrpripple": xrpripple_data,
    "litecoin": litecoin_data,
    "tether": tether_data
}

# To change data type of date from string to timestamp
bitcoin_data['Date'] = pd.to_datetime(bitcoin_data['Date'])
bitcoin_data.loc[0:, 'Date'] = pd.to_datetime(bitcoin_data['Date'])
bitcoin_data.head(5)

# To apply changing data type of Date coloumn to timestamp format for every currency by using for loop
for currency in dateclose_dict:
    dateclose_dict[currency].loc[0:, 'Date'] = pd.to_datetime(dateclose_dict[currency]['Date'])

bitcoin_ts = dateclose_dict['bitcoin'].set_index('Date')
ethereum_ts = dateclose_dict['ethereum'].set_index('Date')
bitcoincash_ts = dateclose_dict['bitcoincash'].set_index('Date')
xrpripple_ts = dateclose_dict['xrpripple'].set_index('Date')
litecoin_ts = dateclose_dict['litecoin'].set_index('Date')
tether_ts = dateclose_dict['tether'].set_index('Date')

dateclose_time_series = {
    "bitcoin": bitcoin_ts,
    "ethereum": ethereum_ts,
    "bitcoincash": bitcoincash_ts,
    "xrpripple": xrpripple_ts,
    "litecoin": litecoin_ts,
    "tether": tether_ts
}

from matplotlib.ticker import ScalarFormatter, FormatStrFormatter



for ts in dateclose_time_series:
    fig, ax = plt.subplots(figsize=(7, 4))
    plt.plot(dateclose_time_series[ts])
    plt.xlabel('Date', fontsize=12)
    plt.ylabel('Price', fontsize=12)
    title_str = "Volume distribution of " + ts
    plt.title(title_str, fontsize=15)
    plt.show()

