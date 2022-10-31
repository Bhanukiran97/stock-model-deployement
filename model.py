# Importing Necessary Libraries

import pandas as pd
import nsepy
from datetime import date
from prophet import Prophet
import pickle
import os


# Extracting Data from NSE

# Infosys

infosys_data = nsepy.get_history(symbol = "INFY", start = date(2012,10,1), end = date.today())
infosys_daily_data = infosys_data.drop(['Symbol', 'Series', 'Prev Close', 'Last', 'VWAP', 'Turnover', 'Trades', 
                                        'Deliverable Volume','%Deliverble'],  axis = 1)
infosys_daily_data.index = pd.DatetimeIndex(infosys_daily_data.index)
infosys_daily_data = infosys_daily_data.asfreq('b')
infosys_daily_data.interpolate(method = 'linear', inplace = True)
infosys_daily_data.reset_index(inplace = True)
infosys_daily_data.to_csv('infosys_daily_data.csv')


# Reliance

reliance_data = nsepy.get_history(symbol = "RELIANCE", start = date(2012,10,1), end = date.today())
reliance_daily_data = reliance_data.drop(['Symbol', 'Series', 'Prev Close', 'Last', 'VWAP', 'Turnover', 'Trades', 
                                          'Deliverable Volume','%Deliverble'],  axis = 1)
reliance_daily_data.index = pd.DatetimeIndex(reliance_daily_data.index)
reliance_daily_data = reliance_daily_data.asfreq('b')
reliance_daily_data.interpolate(method = 'linear', inplace = True)
reliance_daily_data.reset_index(inplace = True)
reliance_daily_data.to_csv('reliance_daily_data.csv')


# Tata Motors

tatamotors_data = nsepy.get_history(symbol = "TATAMOTORS", start = date(2012,10,1), end = date.today())
tatamotors_daily_data = tatamotors_data.drop(['Symbol', 'Series', 'Prev Close', 'Last', 'VWAP', 'Turnover', 'Trades', 
                                              'Deliverable Volume','%Deliverble'],  axis = 1)
tatamotors_daily_data.index = pd.DatetimeIndex(tatamotors_daily_data.index)
tatamotors_daily_data = tatamotors_daily_data.asfreq('b')
tatamotors_daily_data.interpolate(method = 'linear', inplace = True)
tatamotors_daily_data.reset_index(inplace = True)
tatamotors_daily_data.to_csv('tatamotors_daily_data.csv')


# Wipro

wipro_data = nsepy.get_history(symbol = "WIPRO", start = date(2012,10,1), end = date.today())
wipro_daily_data = wipro_data.drop(['Symbol', 'Series', 'Prev Close', 'Last', 'VWAP', 'Turnover', 'Trades', 
                                    'Deliverable Volume','%Deliverble'],  axis = 1)
wipro_daily_data.index = pd.DatetimeIndex(wipro_daily_data.index)
wipro_daily_data = wipro_daily_data.asfreq('b')
wipro_daily_data.interpolate(method = 'linear', inplace = True)
wipro_daily_data.reset_index(inplace = True)
wipro_daily_data.to_csv('wipro_daily_data.csv')


# Model Building

stocks = [infosys_daily_data, reliance_daily_data, tatamotors_daily_data, wipro_daily_data]
files = ['infosys_daily_data.sav', 'reliance_daily_data.sav', 'tatamotors_daily_data.sav', 'wipro_daily_data.sav']
x = 0

for i in stocks:
    
    # Data Preprocessing
    i['Date'] = pd.to_datetime(i['Date'])
    train_data = pd.DataFrame()
    train_data[['ds', 'y']] = i[['Date', 'Close']]
    
    # Model Building
    model = Prophet()
    model.fit(train_data)
    
    # Saving the Model
    file = files[x]
    pickle.dump(model, open(file, 'wb'))
    x = x+1
                 



