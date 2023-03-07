import pandas as pd
import utils
import time
import matplotlib.pyplot as plt


def train(train_data, derivative_stop=0.001, epochs=10000, learning_rate=0.0001, save_plt=False,
          name=None, loss_func='mse', if_std=False):
    start_time = time.time()
    x, y = train_data
    x_copy = x

    params, loss, steps, l2 = utils.linearRegression(x, y, stop_val=derivative_stop, epochs=epochs,
                                                     lr=learning_rate, loss_func=loss_func)

    print('*[TRAINING]*')
    print('「PARAMETERS」  %s' % params)
    print('「LOSS」  %f' % loss)
    print('「STEPS」  %d' % steps)
    print('「TIME_USED」  %f seconds' % (time.time() - start_time))
    print('「R SQUARE VALUE」  %f' % utils.R_square(loss, y))
    print()

    if save_plt:
        if if_std:
            utils.plot(x=x_copy[:, 0].reshape(-1, 1), y=y, pred_y=x @ params,
                       name=f'std-{name}-mse' if loss_func == 'mse' else f'std-{name}-mae')
        elif not if_std:
            utils.plot(x=x_copy[:, 0].reshape(-1, 1), y=y, pred_y=x @ params,
                       name=f'{name}-mse' if loss_func == 'mse' else f'{name}-mae')

    return params


def test(test_data, params, loss_func='mse'):
    x, y = test_data
    if loss_func == 'mse':
        loss = utils.MSE(x_ones=x, y=y, m_b=params, size=len(y))
    else:
        loss = utils.MAE(x_ones=x, y=y, m_b=params, size=len(y))
    print('*[TESTING]*')
    print('「LOSS」  %f' % loss)
    print('「R SQUARE VALUE」  %f' % utils.R_square(loss, y))


def train_ridge(train_data, save_plt=False, if_std=False):
    start_time = time.time()
    x, y = train_data

    rr = utils.ridgeRegression(x, y)
    params = rr.ridgeTest()

    print('*[TRAINING]*')
    print('「PARAMETERS」  %s' % params)
    print('「TIME_USED」  %f seconds' % (time.time() - start_time))
    print()

    if save_plt:
        plt.plot(params)
        plt.savefig(f'Ridge Regression.png' if not if_std else f'Ridge Regression_std.png')

    return params


if __name__ == '__main__':
    df = pd.read_csv('./Concrete_Data.csv').to_numpy()

    num_train = 900
    num_test = 130

    # A) Data pre-processing
    print('======================== Show The Standarization ========================\n')
    scaler = utils.dataPreProcessing(df)
    df_scaled = scaler.standardize()
    scaler.plot()

    ifStandardized = True

    # B) Univariate linear regression
    print('=================== Uni-Variate Linear Regression (MSE Based)===================\n')
    for idx in range(len(df[0])-1):
        print('---------- Column: %d ----------' % idx)
        params = train(utils.prepare_data(df if not ifStandardized else df_scaled, type='uni', stage='train', idx=idx),
                       derivative_stop=0.000001, save_plt=False, name=idx, if_std=ifStandardized)
        test(utils.prepare_data(df if not ifStandardized else df_scaled, type='uni', stage='test', idx=idx), params)
        print()
    print()

    # C) Multivariate linear regression
    print('================== Multi-Variate Linear Regression (MSE Based)===================\n')
    params = train(utils.prepare_data(df=df if not ifStandardized else df_scaled, type='mul', stage='train'),
                   save_plt=False, if_std=ifStandardized)
    test(utils.prepare_data(df if not ifStandardized else df_scaled, type='mul', stage='test'), params)
    print()

    # D) Optional extension 1 – Mean Absolute Error as the loss function
    print('=================== Uni-Variate Linear Regression (MAE Based)===================\n')
    for idx in range(len(df[0])):
        print('---------- Column: %d ----------' % idx)
        params = train(utils.prepare_data(df if not ifStandardized else df_scaled, type='uni', stage='train', idx=idx),
                       derivative_stop=0.000001, save_plt=False, name=idx, loss_func='mae', if_std=ifStandardized)
        test(utils.prepare_data(df if not ifStandardized else df_scaled, type='uni', stage='test', idx=idx),
             params, loss_func='mae')
        print()
    print()

    print('================== Multi-Variate Linear Regression (MAE Based)===================\n')
    params = train(utils.prepare_data(df=df if not ifStandardized else df_scaled, type='mul', stage='train'),
                   loss_func='mae', save_plt=False, if_std=ifStandardized)
    test(utils.prepare_data(df if not ifStandardized else df_scaled, type='mul', stage='test'),
         params, loss_func='mae')
    print()

    # E) Optional extension 2 – Ridge Regression
    print('======================== Ridge Regression ========================\n')
    params = train_ridge(utils.prepare_data(df if not ifStandardized else df_scaled, type='mul', stage='ridge'),
                         save_plt=False, if_std=ifStandardized)
    print()
