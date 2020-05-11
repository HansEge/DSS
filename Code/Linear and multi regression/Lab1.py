import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import axes3d

import sklearn.linear_model as skl_lm
import statsmodels.api as sm
from scipy import stats


def estimate_simple_constants(x_data, y_data):
    x_mean = get_mean(x_data)
    y_mean = get_mean(y_data)

    num = 0
    den = 0

    for i in range(len(x_data)):
        num += (x_data[i] - x_mean) * (y_data[i] - y_mean)
        den += (x_data[i] - x_mean) ** 2

    b1_hat = num / den
    b0_hat = y_mean - b1_hat * x_mean

    return b0_hat, b1_hat


def get_mean(data):
    total = 0
    for i in range(len(data)):
        total += data[i]
    return total / len(data)


if __name__ == "__main__":
    boston = pd.read_csv('Data/boston.csv', usecols=[1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14])

    X = boston[['lstat']].values
    y = boston.medv

    # https://stackoverflow.com/questions/27928275/find-p-value-significance-in-scikit-learn-linearregression
    regr = skl_lm.LinearRegression()
    regr.fit(X,y)

    params = np.append(regr.intercept_, regr.coef_)
    predictions = regr.predict(X)

    newX = pd.DataFrame({"Constant": np.ones(len(X))}).join(pd.DataFrame(X))
    MSE = (sum((y - predictions) ** 2)) / (len(newX) - len(newX.columns))

    var_b = MSE * (np.linalg.inv(np.dot(newX.T, newX)).diagonal())
    sd_b = np.sqrt(var_b)
    ts_b = params / sd_b

    p_values = [2 * (1 - stats.t.cdf(np.abs(i), (len(newX) - 1))) for i in ts_b]

    sd_b = np.round(sd_b, 3)
    ts_b = np.round(ts_b, 3)
    p_values = np.round(p_values, 3)
    params = np.round(params, 4)

    myDF3 = pd.DataFrame()
    myDF3["Coefficients"], myDF3["Standard Errors"], myDF3["t-values"], myDF3["Probabilities"] = [params, sd_b, ts_b,
                                                                                                  p_values]
    print(myDF3)
    # ----------------------------------------------------------------------------------------------------------------

    b0, b1 = estimate_simple_constants(boston['lstat'], boston['medv'])

    plt.scatter(boston['medv'], boston['lstat'])
    plt.xlabel('medv')
    plt.ylabel('lstat')
    plt.legend()

    t = np.arange(0., 55, 0.2)

    plt.plot(t, b0+b1*t, 'r--', label='Regression line')
    plt.legend(loc='upper right', shadow=True, fontsize='large')
    plt.show()

    # END OF LAB 1--------------------------------------------------------END OF LAB 1
    regr = skl_lm.LinearRegression()

    X = boston[['crim', 'zn', 'indus', 'chas', 'nox', 'rm', 'age', 'dis', 'rad', 'tax', 'ptratio', 'black', 'lstat']].values
    y = boston.medv

    regr.fit(X,y)

    params = np.append(regr.intercept_, regr.coef_)
    predictions = regr.predict(X)

    newX = pd.DataFrame({"Constant": np.ones(len(X))}).join(pd.DataFrame(X))
    MSE = (sum((y - predictions) ** 2)) / (len(newX) - len(newX.columns))

    var_b = MSE * (np.linalg.inv(np.dot(newX.T, newX)).diagonal())
    sd_b = np.sqrt(var_b)
    ts_b = params / sd_b

    p_values = [2 * (1 - stats.t.cdf(np.abs(i), (len(newX) - 1))) for i in ts_b]

    sd_b = np.round(sd_b, 3)
    ts_b = np.round(ts_b, 3)
    p_values = np.round(p_values, 3)
    params = np.round(params, 4)

    myDF3 = pd.DataFrame()
    myDF3["Coefficients"], myDF3["Standard Errors"], myDF3["t-values"], myDF3["Probabilities"] = [params, sd_b, ts_b,
                                                                                                  p_values]
    print(myDF3)
    print("hej")

