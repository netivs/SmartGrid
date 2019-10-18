from statsmodels.tsa.stattools import adfuller
from pandas import read_csv, DataFrame, datetime
from pmdarima.arima.utils import ndiffs
from pandas.plotting import autocorrelation_plot
from statsmodels.tsa.arima_model import ARIMA
from utils.constant import LOAD_AREAS
from sklearn.metrics import mean_squared_error
import matplotlib.pyplot as plt
import numpy as np


# LOAD_AREAS = ('AECO', 'AEPAPT', 'AEPIMP', 'AEPKPT', 'AEPOPT',
# 'AP', 'BC', 'CE', 'DAY', 'DEOK', 'DOM', 'DPLCO', 'DUQ', 'EASTON',
# 'EKPC', 'JC', 'ME', 'OE', 'PAPWR', 'PE', 'PEPCO', 'PLCO', 'PN',
# 'PS', 'RECO', 'RTO', 'SMECO', 'UGI', 'VMEU')

def parser(x):
	# return dates.num2date(x).strftime('%m/%d/%Y %H:%M:%S %p')
    return datetime.strptime(x, '%m/%d/%Y %H:%M')

def check_stationary(data):
    i = 1
    print(i)
    for load_area in LOAD_AREAS:
        print(i)
        df = data[load_area]
        result = adfuller(df)
        print('ADF Statistic: %f' % result[0])
        print('p-value: %f' % result[1])
        i +=1
        print("\n")

def autocorrelation_plot(data):
    for load_area in LOAD_AREAS:
        series = data['mw',load_area]
        autocorrelation_plot(series)
        plt.show()

def model_arima(data):
    # run predict for 29 nodes
    for load_area in LOAD_AREAS:
        series = data[['AECO']].squeeze('columns')
        X = series.values
        size = int(len(X) * 0.7)
        train, test = X[0:size], X[size:len(X)]
        history = [x for x in train]
        predictions = list()
        for t in range(len(test)):
            model = ARIMA(history, order=(5,0,0))
            model_fit = model.fit(disp=0)
            output = model_fit.forecast()
            yhat = output[0]
            predictions.append(yhat)
            obs = test[t]
            history.append(obs)
            print('predicted=%f, expected=%f' % (yhat, obs))
        error = mean_squared_error(test, predictions)
        print('Test MSE: %.3f' % error)
        plt.plot(test)
        plt.plot(predictions, color='red')
        plt.show()