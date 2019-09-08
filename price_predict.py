import pandas as pd
import numpy as np
import warnings
import datetime
import pandas_datareader.data as web

from datetime import date
from sklearn import preprocessing
from sklearn import linear_model

from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression

from sklearn.linear_model import Ridge
from sklearn.preprocessing import PolynomialFeatures
from sklearn.pipeline import make_pipeline

from matplotlib import pyplot as plt
from pandas.plotting import register_matplotlib_converters

def predict_next_close(df, model, features = ['Open', 'High', 'Low', 'Close', 'Volume', 'MA_5', 'MA_10', 'DlyReturn']):
    df = df.copy()
    df['NextClose'] = df['Close'].shift(-1)
    last_day = df.tail(1)
    df = df.head(-1)
    df.dropna(inplace=True)
    X = df[features]
        
    y = df['NextClose']
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3)
    model.fit(X_train, y_train)
    
    next_close = model.predict(last_day[features])
    model_score = model.score(X_test, y_test)
    
    return (next_close[0], model_score)

def run(symbol, years=5):
    end = date.today()
    start = end - datetime.timedelta(days = 365 * years)

    df = web.DataReader(symbol, 'yahoo', start, end)
    df['MA_5'] = df['Close'].rolling(window=5).mean()
    df['MA_10'] = df['Close'].rolling(window=10).mean()
    df['DlyReturn'] = df['Close'].pct_change()

    linear_regression = LinearRegression(n_jobs=-1)
    (next_close, score) = predict_next_close(df, linear_regression)
    print("LinearRegression NextClose prediction: {0} (model score: {1})".format(next_close, score))

    poly2 = make_pipeline(PolynomialFeatures(2), Ridge())
    (next_close, score) = predict_next_close(df, poly2)
    print("Polynomial (degree 2) NextClose prediction: {0} (model score: {1})".format(next_close, score))

    clf = linear_model.Lasso(alpha=0.1, tol=0.01)
    (next_close, score) = predict_next_close(df, clf)
    print("Lasso NextClose prediction: {0} (model score: {1})".format(next_close, score))

    register_matplotlib_converters()
    fig_size = plt.rcParams["figure.figsize"]
    fig_size[0] = 12
    fig_size[1] = 9
    plt.rcParams["figure.figsize"] = fig_size

    plt.plot(range(-49,1,1),df['Close'].tail(50))
    plt.plot([1],[next_close],'rx') # next day
    plt.show()
	
if __name__ == "__main__":
    run("IBM")