import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import axes3d
import seaborn as sns

from sklearn.preprocessing import scale
import sklearn.linear_model as skl_lm
from sklearn.metrics import mean_squared_error, r2_score
import statsmodels.api as sm
import statsmodels.formula.api as smf


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


def estimate_multi_constants(x_data, y_data):
    regr = skl_lm.LinearRegression()

    # Estimate constants
    regr.fit(x_data,y_data)

    # aflæst til disse værdier
    y_hat = np.zeros((50, 300))

    for i in range(x_data.shape[1]):
        for k in range(x_data.shape[0]):
            y_hat[i][k] = regr.intercept_ + regr.coef_[0]*x_data[i][k]+regr.coef_[1]*x_data[i][k]
    return y_hat


def get_mean(data):
    total = 0
    for i in range(len(data)):
        total += data[i]
    return total / len(data)


if __name__ == "__main__":
    advertising = pd.read_csv('Data/Advertising.csv', usecols=[1, 2, 3, 4])
    advertising.info()

    y_data = advertising.Sales
    x_data = advertising.TV

    b0, b1 = estimate_simple_constants(x_data, y_data)

    print('b0 = ' + str(b0) + '\nb1 = ' + str(b1))

    # Multiple Regression
    regr = skl_lm.LinearRegression()

    X = advertising[['Radio', 'TV']].values
    y = advertising.Sales

    estimate_multi_constants(X,y)

    regr.fit(X, y)

    Radio = np.arange(0, 50)
    TV = np.arange(0, 300)

    B1, B2 = np.meshgrid(Radio, TV, indexing='xy')
    Z = np.zeros((TV.size, Radio.size))

    for (i, j), v in np.ndenumerate(Z):
        # The response on TV and Radio
        Z[i, j] = (regr.intercept_ + B1[i, j] * regr.coef_[0] + B2[i, j] * regr.coef_[1])

    fig = plt.figure(figsize=(10, 6))
    fig.suptitle('Sales ~ Radio + TV Advertising', fontsize=20)

    ax = axes3d.Axes3D(fig)

    ax.plot_surface(B1, B2, Z, rstride=10, cstride=5, alpha=0.4)
    ax.scatter3D(advertising.Radio, advertising.TV, advertising.Sales, c='r')

    ax.set_xlabel('Radio')
    ax.set_xlim(0, 50)
    ax.set_ylabel('TV')
    ax.set_ylim(bottom=0)
    ax.set_zlabel('Sales')
    fig.show()

    print(1)
'''
ax = sns.regplot(advertising.TV, advertising.Sales, order=1, ci=None, scatter_kws={'color':'r', 's':9})
plt.xlim(-10,310)
plt.ylim(ymin=0)
# plt.show(ax)

regr = skl_lm.LinearRegression()

#X = scale(advertising.TV, with_mean=True, with_std=False).reshape(-1,1)
X = scale(advertising.TV).reshape(-1,1)
y = advertising.Sales

regr.fit(X,y)
print(regr.intercept_/2)
print(regr.coef_)


plt.subplot(311)
plt.scatter(advertising.Newspaper,advertising.Sales)
plt.xlabel('Newspaper advertisement')
plt.ylabel('Sales')
plt.legend()

plt.subplot(312)
plt.scatter(advertising.TV,advertising.Sales)
plt.xlabel('TV advertisement')
plt.ylabel('Sales')
plt.legend()

plt.subplot(313)
plt.scatter(advertising.Radio,advertising.Sales)
plt.xlabel('Radio advertisement')
plt.ylabel('Sales')
plt.legend()
plt.show()'''
