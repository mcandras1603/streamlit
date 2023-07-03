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

import streamlit as st

st.sidebar.title("MENU")

# Sidebar untuk pilihan file CSV
uploaded_file = st.sidebar.file_uploader("Pilih file dataset CSV", type="csv")

# Memuat dan mengubah dataset jika file diunggah
if uploaded_file is not None:
    data = pd.read_csv(uploaded_file)
    data['stok'] = data'[profit']
    data['atribut'] = data['stok']
    data['atribut'] = data['profit']
    st.write(data)
else:
    st.write("Belum ada file yang diunggah.")

#pilih model
option = st.sidebar.selectbox("Pilih Model", ["FOLDA-STOK", "FOLDA-PROFIT", 
                            "OBDHAMIN-STOK", "OBDHAMIN-PROFIT", 
                            "OBICAL-STOK", "OBICAL-PROFIT", 
                            "SOLANEURON-STOK", "SOLANEURON-PROFIT", 
                            "VITACIMIN-STOK", "VITACIMIN-PROFIT"])

#button run
button_pressed = False
if st.sidebar.button("Forecast"):
    # Memproses input jika tombol "Run" ditekan
    button_pressed = True
    if option == "FOLDA-STOK":
        st.sidebar.write("Opsi yang dipilih: FOLDA-STOK")
        label_y = 'Jumlah (Ratusan)'
        label_title = 'Peramalan Stok Folda'
        best_model = joblib.load('FK_lstm100_70s.pkl')
        
    elif option == "FOLDA-PROFIT":
        st.sidebar.write("Opsi yang dipilih: FOLDA-PROFIT")
        label_y = 'Juta (Rupiah)'
        label_title = 'Peramalan Profit Folda'
        best_model = joblib.load('FK_lstm100_70p.pkl')

    elif option == "OBDHAMIN-STOK":
        st.sidebar.write("Opsi yang dipilih: OBDHAMIN-STOK")
        label_y = 'Jumlah (Ratusan)'
        label_title = 'Peramalan Stok Obdhamin'
        best_model = joblib.load('OK_lstm50_70s.pkl')
    
    elif option == "OBDHAMIN-PROFIT":
        st.sidebar.write("Opsi yang dipilih: OBDHAMIN-PROFIT")
        label_y = 'Juta (Rupiah)'
        label_title = 'Peramalan Profit Obdhamin'
        best_model = joblib.load('OK_lstm50_70p.pkl')
    
    elif option == "OBICAL-STOK":
        st.sidebar.write("Opsi yang dipilih: OBICAL-STOK")
        label_y = 'Jumlah (Ratusan)'
        label_title = 'Peramalan Stok Obical'
        best_model = joblib.load('OT_lstm100_70s.joblib')
    
    elif option == "OBICAL-PROFIT":
        st.sidebar.write("Opsi yang dipilih: OBICAL-PROFIT")
        label_y = 'Juta (Rupiah)'
        label_title = 'Peramalan Profit Obical'
        best_model = joblib.load('OT_lstm100_70p.pkl')
    
    elif option == "SOLANEURON-STOK":
        st.sidebar.write("Opsi yang dipilih: SOLANEURON-STOK")
        label_y = 'Jumlah (Ratusan)'
        label_title = 'Peramalan Stok Solaneuron'
        best_model = joblib.load('SK_lstm100_70s.pkl')
    
    elif option == "SOLANEURON-PROFIT":
        st.sidebar.write("Opsi yang dipilih: SOLANEURON-PROFIT")
        label_y = 'Juta (Rupiah)'
        label_title = 'Peramalan Profit Solaneuron'
        best_model = joblib.load('SK_lstm100_70p.joblib')
    
    elif option == "VITACIMIN-STOK":
        st.sidebar.write("Opsi yang dipilih: VITACIMIN-STOK")
        label_y = 'Jumlah (Ratusan)'
        label_title = 'Peramalan Stok Vitacimin'
        best_model = joblib.load('VS_lstm50_70s.pkl')
    
    elif option == "VITACIMIN-PROFIT":
        st.sidebar.write("Opsi yang dipilih: VITACIMIN-PROFIT")
        label_y = 'Juta (Rupiah)'
        label_title = 'Peramalan Profit Vitacimin'
        best_model = joblib.load('VS_lstm50_70p.joblib')

    tgl = data['date']
    angka_list = data['atribut']
    data = pd.DataFrame({'date': tgl, 'profit':angka_list})
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
            
    st.title('Forecasting Time Series Profit dan Stok Apotek XYZ')
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
            
        # Prediksi future steps 5 hari
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
            
        original_dates = pd.to_datetime(original_df['date'], format='%Y-%m-%d', errors='coerce')  # Convert to datetime type
        plt.plot(original_dates, original_df['profit'], label='Aktual', color='green', marker='o')
            
        future_dates = pd.to_datetime(unscaled_df['date'].tail(future_steps), format='%Y-%m-%d', errors='coerce')  # Convert to datetime type
        future_values = unscaled_df['pred_value'].tail(future_steps)
        plt.plot(future_dates, future_values, label='Prediksi Masa Depan', color='orange', marker='o')
            
        plt.plot([original_dates.iloc[-1], future_dates.iloc[0]],
                    [original_df['profit'].iloc[-1], future_values.iloc[0]],
                    linestyle='solid', color='orange')
        st.write("Hasil Peramalan Masa Depan")
        st.write(unscaled_df.tail(5))
        st.write("Grafik Peramalan Masa Depan")
            
        ax.set_xlabel('Tanggal')
        ax.set_ylabel(label_y)
        ax.set_title(label_title)
            
        plt.xticks(rotation=45)
        plt.legend()
        plt.show()
        st.pyplot(plt)
            
    lstm_model_100(test)
            
if not button_pressed:
    st.write("Silakan tekan tombol 'Forecast' untuk mendapatkan hasil")
