import numpy as np
import matplotlib.pyplot as plt
from sklearn import preprocessing
from math import inf


def MSE(x_ones, y, m_b, size):
    return ((y - x_ones @ m_b) ** 2).sum() / size


def MAE(x_ones, y, m_b, size):
    return (y - x_ones @ m_b).sum() / size


def R_square(mse, y):
    return 1 - mse / np.var(y)


def update(lr, los, new_los):
    if new_los < los:
        lr *= 1.01
    else:
        lr *= 0.5
    return lr


def initialize_m_b(nums):
    return np.zeros(nums).reshape(-1, 1)


def calculate_derivatives(y, x, w, size):
    return (w.T @ x.T @ x - y.T @ x) / size


def prepare_data(df, type, stage, idx=None):
    if stage == 'train':
        if type == 'uni':
            y_data = df[0: 900, -1].reshape(-1, 1)
            # n * m matrix
            x_data = df[0: 900, idx].reshape(-1, 1)
            # n * m+1 matrix, last col are ones
            x_ones = np.concatenate([x_data, np.ones((x_data.shape[0], 1))], axis=1)
            return [x_ones, y_data]
        elif type == 'mul':
            # n * 1 y vector
            y_data = df[0: 900, -1].reshape(-1, 1)
            # n * m matrix
            x_data = df[0: 900, :-1]
            # n * m+1 matrix, last col are ones
            x_ones = np.concatenate([x_data, np.ones((x_data.shape[0], 1))], axis=1)
            return [x_ones, y_data]
    elif stage == 'test':
        if type == 'uni':
            y_data = df[900: 1030, -1].reshape(-1, 1)
            # n * m matrix
            x_data = df[900: 1030, idx].reshape(-1, 1)
            # n * m+1 matrix, last col are ones
            x_ones = np.concatenate([x_data, np.ones((x_data.shape[0], 1))], axis=1)
            return [x_ones, y_data]
        elif type == 'mul':
            # n * 1 y vector
            y_data = df[900: 1030, -1].reshape(-1, 1)
            # n * m matrix
            x_data = df[900: 1030, :-1]
            # n * m+1 matrix, last col are ones
            x_ones = np.concatenate([x_data, np.ones((x_data.shape[0], 1))], axis=1)
            return [x_ones, y_data]
    elif stage == 'ridge':
        xArr = df[:, 0:-1]
        yArr = df[:, -1]
        return [xArr, yArr]


def linearRegression(x_ones, y, lr, stop_val, epochs, loss_func='mse'):
    m_b = initialize_m_b(x_ones.shape[1])

    # init vars
    norm_derivatives = inf
    loss = inf
    steps = 0

    while norm_derivatives > stop_val and steps < epochs:
        derivatives = calculate_derivatives(y=y, x=x_ones, w=m_b, size=len(y))

        m_b -= (derivatives.T * lr)

        norm_derivatives = np.linalg.norm(derivatives)
        steps += 1

        if loss_func == 'mse':
            new_loss = MSE(x_ones=x_ones, y=y, m_b=m_b, size=len(y))
        else:
            new_loss = MAE(x_ones=x_ones, y=y, m_b=m_b, size=len(y))
        lr = update(lr=lr, los=loss, new_los=new_loss)
        loss = new_loss

    return m_b, loss, steps, norm_derivatives


class dataPreProcessing:
    def __init__(self, data):
        self.data = data

    def standardize(self):
        return self.preprocessing()[0]

    def preprocessing(self):
        # Z-Score Standardization
        zscore_scaler = preprocessing.StandardScaler()
        data_scaler_1 = zscore_scaler.fit_transform(self.data)
        # Max-Min Standardization
        minmax_scaler = preprocessing.MinMaxScaler()
        data_scaler_2 = minmax_scaler.fit_transform(self.data)
        # MaxAbs Standardization
        maxabs_scaler = preprocessing.MaxAbsScaler()
        data_scaler_3 = maxabs_scaler.fit_transform(self.data)
        # RobustScaler Standardization
        robust_scaler = preprocessing.RobustScaler()
        data_scaler_4 = robust_scaler.fit_transform(self.data)

        return data_scaler_1, data_scaler_2, data_scaler_3, data_scaler_4

    def plot(self):
        data_scaler_1, data_scaler_2, data_scaler_3, data_scaler_4 = self.preprocessing()
        data_list = [self.data, data_scaler_1, data_scaler_2, data_scaler_3, data_scaler_4]
        scaler_list = [15, 10, 15, 10, 15]
        color_list = ['pink', 'green', 'red', 'orange', 'blue']
        marker_list = ['o', ',', '+', 's', 'p']
        title_list = ['source data', 'zscore_scaler', 'minmax_scaler', 'maxabs_scaler', 'robust_scaler']

        plt.figure(figsize=[15, 10])
        for i, data_single in enumerate(data_list):
            plt.subplot(2, 3, i + 1)
            plt.scatter(data_single[:, 0], data_single[:, -1],
                        s=scaler_list[i],
                        marker=marker_list[i],
                        c=color_list[i])
            plt.title = title_list[i]
        plt.suptitle('Row Data And Standardized Data')
        plt.show()


class ridgeRegression(object):
    def __init__(self, xArr, yArr, alpha=0.2):
        self.xArr =xArr
        self.yArr = yArr
        self.alpha = alpha

    def ridgeRegres(self, xMat, yMat, lam=0.2):
        xTx = xMat.T * xMat
        denom = xTx + np.eye(np.shape(xMat)[1]) * lam
        if np.linalg.det(denom) == 0.0:
            print('This matrix is singular, cannot do inverse')
            return
        ws = denom.I * (xMat.T * yMat)
        return ws

    def ridgeTest(self):
        self.xMat = np.mat(self.xArr)
        self.yMat = np.mat(self.yArr).T
        self.yMean = np.mean(self.yMat)
        self.yMat = self.yMat - self.yMean
        self.xMeans = np.mean(self.xMat, 0)
        self.xVar = np.var(self.xMat, 0)
        self.xMat = (self.xMat - self.xMeans) / self.xVar
        numTestPts = 30
        self.wMat = np.zeros((numTestPts, np.shape(self.xMat)[1]))
        for i in range(numTestPts):
            ws = self.ridgeRegres(self.xMat, self.yMat, np.exp(i - 10))
            self.wMat[i, :] = ws.T
        return self.wMat


def plot(x, y, pred_y, name):
    # feature_name = 'Feature ' + name
    # print(type(feature_name))
    # plt.title(name)
    plt.xlabel('Predictor')
    plt.ylabel('Response')
    plt.scatter(x, y, c='green')
    plt.plot(x, pred_y, color='orange')
    plt.savefig(f'{name}.png')
    plt.clf()
