import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import math
import time
import datetime
from dateutil import parser
import matplotlib.dates as mdates
import matplotlib.ticker as plticker

from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import mean_squared_error

from keras.models import Sequential
from keras.layers import Dense
from keras.layers import LSTM, GRU
from tensorflow import keras

# function to split the data
def create_dataset(dataset, pred_col, look_back=1):
    dataX, dataY = [], []
    for i in range(len(dataset)-look_back-1):
        a = dataset[i:(i+look_back), :]
        dataX.append(a)
        dataY.append(dataset[i + look_back, pred_col])
    return np.array(dataX), np.array(dataY)

class Model:
    def __init__(self):
        pass
    
    def get_model(self):

        now = datetime.datetime.now()
        nowDate = now.strftime('%Y-%m-%d')

        org_df = pd.read_csv('static/csv/crypto_data_news_reddit_final_05_26_v3.csv',index_col=0)

        org_df['flair'] = org_df['gnews_flair'] + org_df['reddit_flair']
        org_df['tb_polarity'] = org_df['gnews_tb_polarity'] + org_df['reddit_tb_polarity']
        org_df['tb_subjectivity'] = org_df['gnews_tb_subjectivity'] + org_df['reddit_tb_subjectivity']
        org_df['sid_pos'] = org_df['gnews_sid_pos'] + org_df['reddit_sid_pos']
        org_df['sid_neg'] = org_df['gnews_sid_neg'] + org_df['reddit_sid_neg']
        org_df = org_df[['open_BTCUSDT','high_BTCUSDT','low_BTCUSDT', 'close_BTCUSDT',
                         'volume_BTCUSDT', 'close_LTCUSD', 'volume_LTCUSD', 'close_ETHUSD',
                         'volume_ETHUSD', 'flair', 'tb_polarity', 'tb_subjectivity', 'sid_pos', 'sid_neg']]

        dataset = org_df.values
        dataset = dataset.astype('float32')
        scaler = MinMaxScaler(feature_range=(0, 1))
        dataset = scaler.fit_transform(dataset)
        num_of_features = len(org_df.columns)

        expr_name = 'expr_14' + nowDate
        look_back = 24*120 # 60 days, as each entry is for 1 hour
        lstm_layers = 64
        epochs=5
        batch_size=64

        train_size_percent = 0.80
        pred_col = org_df.columns.get_loc('close_BTCUSDT')

        train_size = int(len(dataset) * train_size_percent)
        test_size = len(dataset) - train_size
        train, test = dataset[0:train_size, :], dataset[train_size:len(dataset), :]

        trainX, trainY = create_dataset(train, pred_col, look_back=look_back)
        testX, testY = create_dataset(test, pred_col, look_back=look_back)
        # reshape input to be  [samples, time steps, features]
        trainX = np.reshape(trainX, (trainX.shape[0], look_back, num_of_features))
        testX = np.reshape(testX, (testX.shape[0],look_back, num_of_features))

        reconstructed_model = keras.models.load_model('static/models')          #기존에 학습했던 model을 불러옴

        trainPredict = reconstructed_model.predict(trainX)
        testPredict = reconstructed_model.predict(testX)

        # Get something which has as many features as dataset
        trainPredict_extended = np.zeros((len(trainPredict),num_of_features))
        # Put the predictions there
        trainPredict_extended[:,pred_col] = trainPredict[:,0]
        # Inverse transform it and select the 3rd column.
        trainPredict = scaler.inverse_transform(trainPredict_extended) [:,pred_col]

        # Get something which has as many features as dataset
        testPredict_extended = np.zeros((len(testPredict),num_of_features))
        # Put the predictions there
        testPredict_extended[:,pred_col] = testPredict[:,0]
        # Inverse transform it and select the pred_col column.
        testPredict = scaler.inverse_transform(testPredict_extended)[:,pred_col] 

        trainY_extended = np.zeros((len(trainY),num_of_features))
        trainY_extended[:,pred_col]=trainY
        trainY = scaler.inverse_transform(trainY_extended)[:,pred_col]

        testY_extended = np.zeros((len(testY),num_of_features))
        testY_extended[:,pred_col]=testY
        testY = scaler.inverse_transform(testY_extended)[:,pred_col]

        # calculate root mean squared error
        trainScore_RMSE = math.sqrt(mean_squared_error(trainY, trainPredict))
        testScore_RMSE = math.sqrt(mean_squared_error(testY, testPredict))

        # calculate absolute mean error
        trainScore_MAE = np.sum(np.absolute(trainY - trainPredict))/len(trainY)
        testScore_MAE = np.sum(np.absolute(testY - testPredict))/len(testY)

        # shift train predictions for plotting
        trainPredictPlot = np.empty_like(dataset)
        trainPredictPlot[:, :] = np.nan
        trainPredictPlot[look_back:len(trainPredict)+look_back, pred_col] = trainPredict

        # shift test predictions for plotting
        testPredictPlot = np.empty_like(dataset)
        testPredictPlot[:, :] = np.nan
        testPredictPlot[len(trainPredict)+(look_back*2)+1:len(dataset)-1, pred_col] = testPredict

        # contruct pandas dataframe for plotting
        time_df = pd.DataFrame(org_df.index)
        time_df['Actual'] = scaler.inverse_transform(dataset)[:,pred_col]
        df1 = pd.DataFrame(trainPredictPlot[:,pred_col],columns=['Train'])
        df2 = pd.DataFrame(testPredictPlot[:,pred_col],columns=['Test'])
        time_df2 = pd.concat([time_df, df1, df2], axis=1, sort=False)
        time_df2.set_index('timestamp',inplace=True)

        # plot the output
        fig, ax = plt.subplots(figsize=(15,7))
        time_df2.plot(ax=ax,rot=90,alpha=0.7)
        plt.xlabel('Timestamp')
        plt.ylabel('Bitcoin Value')
        plt.title('Bitcoin Price Prediction')
        plt.savefig('static/img/' + expr_name + '.png',bbox_inches = "tight")
        
        print(time_df)
        # print(time_df2)
        # print(df1)
        # print(df2)


    def create_model(self):

        org_df = pd.read_csv('static/csv/crypto_data_news_reddit_final_05_26_v3.csv',index_col=0)

        org_df['flair'] = org_df['gnews_flair'] + org_df['reddit_flair']
        org_df['tb_polarity'] = org_df['gnews_tb_polarity'] + org_df['reddit_tb_polarity']
        org_df['tb_subjectivity'] = org_df['gnews_tb_subjectivity'] + org_df['reddit_tb_subjectivity']
        org_df['sid_pos'] = org_df['gnews_sid_pos'] + org_df['reddit_sid_pos']
        org_df['sid_neg'] = org_df['gnews_sid_neg'] + org_df['reddit_sid_neg']
        org_df = org_df[['open_BTCUSDT','high_BTCUSDT','low_BTCUSDT', 'close_BTCUSDT',
                         'volume_BTCUSDT', 'close_LTCUSD', 'volume_LTCUSD', 'close_ETHUSD',
                         'volume_ETHUSD', 'flair', 'tb_polarity', 'tb_subjectivity', 'sid_pos', 'sid_neg']]

        dataset = org_df.values
        dataset = dataset.astype('float32')
        scaler = MinMaxScaler(feature_range=(0, 1))
        dataset = scaler.fit_transform(dataset)
        num_of_features = len(org_df.columns)

        expr_name = 'expr_14'
        look_back = 24*120 # 60 days, as each entry is for 1 hour
        lstm_layers = 64
        epochs=5
        batch_size=64

        train_size_percent = 0.80
        pred_col = org_df.columns.get_loc('close_BTCUSDT')

        train_size = int(len(dataset) * train_size_percent)
        test_size = len(dataset) - train_size
        train, test = dataset[0:train_size, :], dataset[train_size:len(dataset), :]

        trainX, trainY = create_dataset(train, pred_col, look_back=look_back)
        testX, testY = create_dataset(test, pred_col, look_back=look_back)
        # reshape input to be  [samples, time steps, features]
        trainX = np.reshape(trainX, (trainX.shape[0], look_back, num_of_features))
        testX = np.reshape(testX, (testX.shape[0],look_back, num_of_features))

        model = Sequential()
        model.add(GRU(lstm_layers, input_shape=(look_back,num_of_features)))
        model.add(Dense(1))
        model.compile(loss='mean_squared_error', optimizer='adam')
        history= model.fit(trainX, trainY,validation_split=0.30, epochs=epochs, batch_size=batch_size,shuffle=False)

        model.save('static/models/')

        return model