import pandas as pd
import numpy as np
import talib as ta
from sklearn.pipeline import make_pipeline
from sklearn.svm import SVC
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score, classification_report
import time

start_time = time.time()

# Evaluate trading system defined in current file.
if __name__ == '__main__':
    import quantiacsToolbox

    result = quantiacsToolbox.runts(__file__)
    values_test = result['settings']['y_test'].values
    values_pred = result['settings']['y_pred'].values

    print('stats:', result['stats'])
    print('evalDate', result['evalDate'])
    print('runtime', result['runtime'])
    print('errorLog', result['errorLog'])
    # print(result['settings']['y_test'].shape)
    # print(result['settings']['y_pred'].shape)
    # print('Accuracy Score is {:f}'.format(accuracy_score(values_test, values_pred)))
    # print('Classification report is {:s}\n'.format(classification_report(values_test, values_pred)))


def mySettings():
    """ Define your trading system settings here """

    settings = {}

    print('your call')
    # Futures Contracts
    # settings['markets'] = ['F_CL', 'F_NG', 'F_HO', 'F_SB', 'F_RB', 'F_GC', 'F_C', 'F_W', 'F_S', 'F_HG',
    #                        'F_BO', 'F_SI', 'F_CC', 'F_TY', 'F_FV', 'F_US', 'F_JY', 'F_TU', 'F_ES', 'F_ED', 'CASH']

    # settings['markets'] = ['F_GC']
    # settings['markets'] = ['CASH', 'VLO', 'ZTS', 'GOOG', 'AMZN', 'AAPL', 'BA', 'FB', 'NFLX', 'THC']
    # best one for this 0.61 with 2520 lookback

    settings['markets'] = ['CASH','GOOG', 'AMZN', 'AAPL', 'BA', 'FB', 'NFLX', 'THC', 'F_US',
                           'F_TU', 'F_ES']

    settings['lookback'] = 87
    settings['budget'] = 10 ** 6
    settings['slippage'] = 0.05
    # settings['X_train'] = pd.DataFrame(columns=['close', 'open', 'low', 'high'])
    # settings['X_test'] = pd.DataFrame(columns=['close', 'open', 'low', 'high'])
    # settings['y_train'] = pd.Series()
    # settings['y_test'] = pd.Series()
    # settings['y_pred'] = pd.Series()

    settings['market'] = ''
    settings['iterations'] = 0

    return settings


def myTradingSystem(DATE, OPEN, HIGH, LOW, CLOSE, VOL, OI, P, R, RINFO, exposure, equity, settings):
    nMarkets = len(settings['markets'])
    pos = np.zeros((1, nMarkets), dtype='float')
    for i in range(nMarkets):
        try:
            pos[0, i] = predict(OPEN[:, i], HIGH[:, i], LOW[:, i], CLOSE[:, i], settings, i)
        # print('Position is  {:f}'.format(pos[0, i]))
        #  for NaN data set position to 0
        except ValueError:
            pos[0, i] = 0.


    return pos, settings


################################# MY FUNCTIONS #################################
# define helper function
# use close price predict the trend of the next day
def predict(OPEN, HIGH, LOW, CLOSE, settings, i):
    n = 10

    data_temp = pd.DataFrame()
    data_temp['close'] = CLOSE
    data_temp['open'] = OPEN
    data_temp['low'] = LOW
    data_temp['high'] = HIGH

    fee = (data_temp['close'] - data_temp['open']) * settings['slippage']
    data_temp.dropna(axis=0, inplace=True)
    data_temp['SMA'] = data_temp['close'].rolling(window=n).mean()
    data_temp['Corr'] = data_temp['SMA'].rolling(window=n).corr(data_temp['close'])
    data_temp['Corr'][data_temp.Corr > 1] = 1
    data_temp['Corr'][data_temp.Corr < -1] = -1

    close_open_diff = data_temp['close'] - data_temp['open']
    fee = (close_open_diff) * settings['slippage']
    data_temp['sign_close_diff'] = np.where(abs(close_open_diff) > fee, np.sign(close_open_diff), 0.0)
    # drop nan values
    data_temp.dropna(axis=0, inplace=True)
    split_rate = 0.8
    split = int(split_rate * len(data_temp))
    y = data_temp.loc[:, 'sign_close_diff']
    X = data_temp.drop(['sign_close_diff', 'high', 'low', 'close', 'open'], axis=1)

    X_train = X[:split]
    X_test = X[split:]
    y_train = y[:split]
    # y_test = y[split:]

    svc = SVC(C=100.0, cache_size=200, class_weight=None, coef0=0.0,
              decision_function_shape=None, degree=3, gamma=0.001, kernel='rbf',
              max_iter=-1, probability=False, random_state=None, shrinking=True,
              tol=0.001, verbose=False)

    pipeline = make_pipeline(
        StandardScaler(),
        svc,
    )

    pipeline.fit(X_train, y_train)
    y_pred = pipeline.predict(X_test)
    unique, counts = np.unique(y_pred, return_counts=True)
    y_dict = dict(zip(unique, counts))
    if (not (1.0 in y_dict)):
        y_dict[1.0] = 0
    if (not (0.0 in y_dict)):
        y_dict[0.0] = 0

    # print(y_dict)
    y_pred = np.where(y_dict[1.0] > y_dict[0.0], 1.0, 0.0)
    # save old data

    return y_pred.item(0)
