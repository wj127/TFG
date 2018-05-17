#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""moduleName.py Module

Explanation goes here.
    
Created by: Davis Yoel Armas Ayala
"""

from statsmodels.tsa.stattools import adfuller
from statsmodels.tsa.arima_model import ARIMA
import pandas as pd
from numpy import log
import numpy as np
import MySQLdb
import plotly.offline as py
import plotly.graph_objs as go


def to_matrix_format(dataset=[]):
    if dataset:
        print("The DataSet:")
        for num, elem in enumerate(dataset):
            print("{}. -> {}".format(num+1, elem))


if __name__ == '__main__':
    conn = MySQLdb.connect(host="127.0.0.1", user="root", passwd="DrStrange", db="TFG")
    sql_command = 'SELECT `date`,  weightedAverage FROM btcPOLONIEX_14400'
    df = pd.read_sql(sql_command, con=conn, index_col='date')  # Construction of the Pandas
    new = pd.Series(data=df['weightedAverage'], index=df.index)  # Create a time series from a single df column

    values = new.values  # Takes just the values (prices)
    datasets = list()

    # Construction of the data set. 20 examples of 6 values each
    for i in xrange(0, 20):
        datasets.append(values[i:i+6])

    # Joining of all examples, one after another. We need this to apply the ARIMA model
    the_prices = [value for elem in datasets for value in elem]
    print to_matrix_format(datasets)

    # >>> First fitting of ARIMA model into the raw prices (non stationary)
    model = ARIMA(the_prices, order=(6, 1, 0))  # Takes the previous 6 observations to make the prediction. Also takes
    # the difference from the previous value
    model_fit = model.fit(disp=0)
    print(model_fit.summary())

    print("\n~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~PREDICTIONS ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~")
    for i in xrange(5, len(the_prices), 6):
        predictions = model_fit.predict(i, i)  # Predicts always the 6th value
        print("> The real value is '{}' and the predicted is '{}'".format(the_prices[i], predictions[0]))


    # >>> Second fitting of ARIMA model into the transformed prices (stationary)
    log_prices = np.log(the_prices)  # We apply a normal log in order to subtract trend
    model = ARIMA(log_prices, order=(6, 1, 0))  # Same as above
    model_fit = model.fit(disp=0)
    print(model_fit.summary())

    print("\n~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~ PREDICTIONS 2 ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~")
    for i in xrange(5, len(log_prices), 6):
        predictions = np.exp(model_fit.predict(i, i, typ='levels'))  # Same as above, but transforms the data back into
        # normal (exponential)
        print("> The real value is '{}' and the predicted is '{}'".format(the_prices[i], predictions[0]))

