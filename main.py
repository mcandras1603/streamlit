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
data = pd.read_csv("data_penjualan_barang_lengkap.csv")
obat = "FOLDA KAPLET@30"
step = 5

# drop baris menjadi 0
data = data[data['jumlah'] != 0]
data = data[data['sub_total'] != 0]

# atribut baru
data['profit'] = (data['harga_jual'] - data['harga_beli']) * data['jumlah']

# Create a Date Column to set up aggregation by month
data['date'] = data.apply(lambda row: pd.Timestamp(row.tanggal), axis=1)
data.drop('tanggal', inplace=True, axis=1)

# Fokus Data
data = data[(data['jenis'] == "Vitamin & Food Suplement")]
data = data[(data['nama'] == obat)]
data = data[(data['date'] >= "2018-11-14") & (data['date'] <= "2021-06-21")]

# Weekly Data
data['date'] = pd.to_datetime(data['date']) - pd.to_timedelta(7, unit='d')
data = data.groupby(pd.Grouper(key='date', freq='W-MON', sort=True)).sum().reset_index()

data = data.drop(['barang_id', 'harga_beli', 'ppn', 'diskon', 'harga_jual', 'jumlah', 'sub_total'], axis=1)

# interpolasi nilai 0
data['profit'] = data['profit'].replace(0, np.nan)
data['profit'] = data['profit'].interpolate(method="linear")
original = data

# stasionery
def get_diff(data):
    data['profit_diff'] = data.profit.diff()
    data = data.dropna()
    return data

stationary_df = get_diff(data)

# create dataframe for transformation from time series to supervised
def generate_supervised(data):
    supervised_df = data.copy()

    # create column for each lag
    for i in range(1, 5):
        col_name = 'lag_' + str(i)
        supervised_df[col_name] = supervised_df['profit_diff'].shift(i)

    # drop null values
    supervised_df = supervised_df.dropna().reset_index(drop=True)
    return supervised_df

model_df = generate_supervised(stationary_df)

# Sampel Data
def tts(data):
    data = data.drop(['profit', 'date'], axis=1)
    test = data.sample(n=40, random_state=42).values
    return test

test = tts(model_df)

# Normalisasi
def scale_data(test_set):
    # apply Min Max Scaler
    scaler = MinMaxScaler()
    scaler = scaler.fit(test_set)

    # reshape test set
    test_set = test_set.reshape(test_set.shape[0], test_set.shape[1])
    test_set_scaled = scaler.transform(test_set)

    X_test, y_test = test_set_scaled[:, 1:], test_set_scaled[:, 0:1].ravel()

    return X_test, y_test, scaler

X_test, y_test, scaler_object = scale_data(test)

# Denormalisasi
def undo_scaling(y_pred, x_test, scaler_obj, lstm=False):
    # reshape y_pred
    y_pred = y_pred.reshape(y_pred.shape[0], 1, 1)

    if not lstm:
        x_test = x_test.reshape(x_test.shape[0], 1, x_test.shape[1])

    # rebuild test set for inverse transform
    pred_test_set = []
    for index in range(0, len(y_pred)):
        pred_test_set.append(np.concatenate([y_pred[index], x_test[index]], axis=1))

    # reshape pred_test_set
    pred_test_set = np.array(pred_test_set)
    pred_test_set = pred_test_set.reshape(pred_test_set.shape[0], pred_test_set.shape[2])

    # inverse transform
    pred_test_set_inverted = scaler_obj.inverse_transform(pred_test_set)

    return pred_test_set_inverted

# Load Original Data
def load_original_df():
    # load in original dataframe without scaling applied
    original_df = original
    return original_df

# Dataframe
def predict_df(unscaled_predictions, original_df):
    # create dataframe that shows the predicted sales
    result_list = []
    profit_dates = list(original_df[-41:].date)
    act_profit = list(original_df[-41:].profit)

    for index in range(len(unscaled_predictions)):
        result_dict = {}
        result_dict['pred_value'] = int(unscaled_predictions[index][0] + act_profit[index])
        if index + 1 < len(profit_dates):
            result_dict['date'] = profit_dates[index + 1]
        else:
            result_dict['date'] = None
        result_list.append(result_dict)

    df_result = pd.DataFrame(result_list)
    return df_result

# Plot
def plot_results(results, original_df, model_name):
    fig, ax = plt.subplots(figsize=(24, 8))
    sns.lineplot(x=original_df.date, y=original_df.profit, data=original_df, ax=ax,
                 label='Original', color='mediumblue')
    ax.set_xticklabels(ax.get_xticklabels(), rotation=45, fontsize=8)

    sns.lineplot(x=results.date, y=results.pred_value, data=results, ax=ax,
                 label='Predicted', color='Red')

    ax.set(xlabel="Date",
           ylabel="Profit",
           title=f"{model_name} Profit Forecasting Prediction")

    ax.legend()
    sns.despine()

# All in
def run_model(test_data, model, model_name):
    X_test, y_test, scaler_object = scale_data(test_data)

    mod = model
    mod.fit(X_test)
    predictions = mod.predict(X_test)

    # Undo scaling to compare predictions against original data
    original_df = load_original_df()
    unscaled = undo_scaling(predictions, X_test, scaler_object)
    unscaled_df = predict_df(unscaled, original_df)

# st.title('Forecasting Time Series LSTM MODEL')
# LSTM MODEL
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
    future_steps = step
    last_prediction = unscaled[-1][0]

    future_predictions = []
    for i in range(future_steps):
        non_zero_elements = [elem for elem in unscaled[i] if elem != 0]
        if non_zero_elements:
            prediction = last_prediction + non_zero_elements[0]
        else:
            prediction = last_prediction
        future_predictions.append(prediction)
        last_prediction = prediction

    last_date = unscaled_df['date'].iloc[-1]
    date_range = pd.date_range(last_date, periods=future_steps + 1, freq='W-MON')[1:]  # Generate rentang tanggal future steps
    date_strings = [str(date) for date in date_range]

    future_df = pd.DataFrame({'date': date_strings, 'pred_value': future_predictions})
    unscaled_df = pd.concat([unscaled_df, future_df], ignore_index=True)

    original_df['date'] = pd.to_datetime(original_df['date'])
    unscaled_df['date'] = pd.to_datetime(unscaled_df['date'])
    fig, ax = plt.subplots(figsize=(20, 12))

    sns.lineplot(x=original_df['date'], y=original_df['profit'], data=original_df, ax=ax,
                 label='Aktual', color='green', marker='o')

    original_df['date'] = original_df['date']
    unscaled_df['date'] = unscaled_df['date']

    future_dates = pd.date_range(last_date, periods=future_steps + 1, freq='W-MON')[1:]
    future_values = unscaled_df['pred_value'].tail(future_steps)
    sns.lineplot(x=future_dates, y=future_values, ax=ax, label='Prediksi Masa Depan', color='orange', marker='o')
    print(unscaled_df.tail(5))

    sns.lineplot(x=[original_df['date'].iloc[-1], future_dates[0]],
                 y=[original_df['profit'].iloc[-1], future_values.iloc[0]],
                 ax=ax, linestyle='solid', color='orange')

    ax.set_xlabel('Tanggal')
    ax.set_ylabel('Juta (Rupiah)')
    ax.set_title('Prediksi Profit FOLDA KAPLET@30 Model LSTM')

    plt.xticks(rotation=45)
    plt.legend()
    plt.show()
    st.pyplot(plt)

lstm_model_100(test)
