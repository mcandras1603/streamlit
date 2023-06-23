import streamlit as st
import os
import pandas as pd
import numpy as np
import tensorflow as tf

from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import mean_squared_error, mean_absolute_error, mean_absolute_percentage_error
from statsmodels.tsa.arima.model import ARIMA

from tensorflow import keras
from tensorflow.keras.layers import Dense
from tensorflow.keras.models import Sequential
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.callbacks import EarlyStopping
from tensorflow.keras.layers import LSTM
from sklearn.model_selection import KFold, cross_val_score, train_test_split

import matplotlib.pyplot as plt
import seaborn as sns
import matplotlib.dates as mdates

import datetime

import statsmodels.tsa.api as smt
import statsmodels.api as sm
from statsmodels.tools.eval_measures import rmse

import pickle
import joblib

best_model = joblib.load('FK_lstm100_70p.joblib')
#model : obat FOLDA KAPLET@30
#20 data
pro = [50000, 210000, 36000, 52000, 102100, 1900201, 20001, 15040 ,270210 ,401201, 60809, 5110819, 111213, 9808, 86998, 567260, 18788, 56732, 60887, 5062716]
tgl = ['2021-02-01','2021-02-08','2021-02-15' ,'2021-02-22' ,'2021-03-01' ,'2021-03-08' ,'2021-03-15' ,'2021-03-22','2021-03-29','2021-04-05','2021-04-12', '2021-04-19', '2021-04-26', '2021-05-03', '2021-05-10', '2021-05-17', '2021-05-24', '2021-05-31', '2021-06-07', '2021-06-14']
data = pd.DataFrame({'date': tgl, 'profit':pro})
original = data

#stasionery
def get_diff(data):
    data['profit_diff'] = data.profit.diff()
    data = data.dropna()
    return data

stationary_df = get_diff(data)

#create dataframe for transformation from time series to supervised
def generate_supervised(data):
    supervised_df = data.copy()

    #create column for each lag
    for i in range(1,5):
        col_name = 'lag_' + str(i)
        supervised_df[col_name] = supervised_df['profit_diff'].shift(i)

    #drop null values
    supervised_df = supervised_df.dropna().reset_index(drop=True)
    return supervised_df

model_df = generate_supervised(stationary_df)

#Sampel Data
def tts(data):
    data = data.drop(['profit', 'date'], axis=1)
    test = data.sample(n=10, random_state=12).values
    return test

test = tts(model_df)

#Normalisasi
def scale_data(test_set):
    #apply Min Max Scaler
    scaler = MinMaxScaler()
    scaler = scaler.fit(test_set)

    # reshape test set
    test_set = test_set.reshape(test_set.shape[0], test_set.shape[1])
    test_set_scaled = scaler.transform(test_set)

    X_test, y_test = test_set_scaled[:, 1:], test_set_scaled[:, 0:1].ravel()

    return X_test, y_test, scaler

#Denormalisasi
def undo_scaling(y_pred, x_test, scaler_obj, lstm=False):
    #reshape y_pred
    y_pred = y_pred.reshape(y_pred.shape[0], 1, 1)

    if not lstm:
        x_test = x_test.reshape(x_test.shape[0], 1, x_test.shape[1])

    #rebuild test set for inverse transform
    pred_test_set = []
    for index in range(0,len(y_pred)):
        pred_test_set.append(np.concatenate([y_pred[index],x_test[index]],axis=1))

    #reshape pred_test_set
    pred_test_set = np.array(pred_test_set)
    pred_test_set = pred_test_set.reshape(pred_test_set.shape[0], pred_test_set.shape[2])

    #inverse transform
    pred_test_set_inverted = scaler_obj.inverse_transform(pred_test_set)

    return pred_test_set_inverted

#Load Original Data
def load_original_df():
    #load in original dataframe without scaling applied
    original_df = original
    return original_df

#Dataframe
def predict_df(unscaled_predictions, original_df):
    #create dataframe that shows the predicted sales
    result_list = []
    profit_dates = list(original_df[-11:].date)
    act_profit = list(original_df[-11:].profit)

    for index in range(len(unscaled_predictions)):
        result_dict = {}
        result_dict['pred_value'] = int(unscaled_predictions[index][0] + act_profit[index])
        if index+1 < len(profit_dates):
            result_dict['date'] = profit_dates[index+1]
        else:
            result_dict['date'] = None
        result_list.append(result_dict)

    df_result = pd.DataFrame(result_list)
    return df_result

#st.title('Forecasting Time Series LSTM MODEL')
#LSTM MODEL
def lstm_model_100(test_data):
    X_test, y_test, scaler_object = scale_data(test_data)
    X_test = X_test.reshape(X_test.shape[0], 1, X_test.shape[1])

    predictions = best_model.predict(X_test, batch_size=1)
    original_df = load_original_df()
    unscaled = undo_scaling(predictions, X_test, scaler_object, lstm=True)
    if (unscaled <= 0).any():
        unscaled[unscaled <= 0] = 0
    unscaled_df = predict_df(unscaled, original_df)

    # Prediksi future steps 4 hari
    future_steps = 5
    last_prediction = unscaled[-1][0]

    future_predictions = []
    for i in range(future_steps):
        non_zero_elements = [elem for elem in unscaled[:, i] if elem != 0]
        if non_zero_elements:
            prediction = last_prediction + non_zero_elements[0]
        else:
            prediction = last_prediction
        future_predictions.append(prediction)
        last_prediction = prediction

    last_date = unscaled_df['date'].iloc[-1]
    date_range = pd.date_range(start=last_date, periods=future_steps+1, freq='W-MON')[1:]
    date_strings = [date.strftime('%Y-%m-%d') for date in date_range]  # Convert dates to strings with format 'YYYY-MM-DD'

    future_df = pd.DataFrame({'date': date_strings, 'pred_value': future_predictions})
    unscaled_df = pd.concat([unscaled_df, future_df], ignore_index=True)

    fig, ax = plt.subplots(figsize=(20, 12))

    plt.plot(original_df['date'].dt.strftime('%Y-%m-%d'), original_df['profit'], label='Aktual', color='green', marker='o')

    future_dates = unscaled_df['date'].tail(future_steps)
    future_values = unscaled_df['pred_value'].tail(future_steps)
    plt.plot(future_dates, future_values, label='Prediksi Masa Depan', color='orange', marker='o')

    plt.plot([original_df['date'].iloc[-1].strftime('%Y-%m-%d'), future_dates.iloc[0]],
             [original_df['profit'].iloc[-1], future_values.iloc[0]],
             linestyle='solid', color='orange')

    ax.set_xlabel('Tanggal')
    ax.set_ylabel('Juta (Rupiah)')
    ax.set_title('Prediksi Profit FOLDA KAPLET@30 Model LSTM')

    plt.xticks(rotation=45)
    plt.legend()
    plt.show()
    st.pyplot(plt)

lstm_model_100(test)
