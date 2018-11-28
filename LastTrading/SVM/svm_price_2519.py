import pandas as pd
import numpy as np
from sklearn.pipeline import make_pipeline
from sklearn.svm import SVC
from sklearn.preprocessing import StandardScaler

# Evaluate trading system defined in current file.
if __name__ == '__main__':
    import quantiacsToolbox

    result = quantiacsToolbox.runts(__file__)
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
    settings['markets'] = ['CASH', 'KSU', 'VLO', 'ZTS', 'GOOG', 'AMZN', 'AAPL', 'BA', 'FB', 'NFLX', 'THC', 'F_US',
                           'F_TU', 'F_ES']

    # settings['markets'] = ['F_GC']
    # settings['markets'] = ['CASH', 'VLO', 'ZTS', 'GOOG', 'AMZN', 'AAPL', 'BA', 'FB', 'NFLX', 'THC']
    # best one for this 0.61 with 2520 lookback
    settings['beginInSample'] = '19900102'
    settings['endInSample'] = '20180610'
    settings['lookback'] = 2519
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
    actual_market = settings['markets'][0]
    for i in range(nMarkets):
        try:
            pos[0, i] = predict(OPEN[:, i], HIGH[:, i], LOW[:, i], CLOSE[:, i], settings,i)
        #  for NaN data set position to 0
        except ValueError:
            pos[0, i] = 0.
    return pos, settings


################################# MY FUNCTIONS #################################

def predict(OPEN, HIGH, LOW, CLOSE, settings,i):
    # settings['market'] = settings['markets'][i]

    data_temp = pd.DataFrame()
    data_temp['close'] = CLOSE
    data_temp['open'] = OPEN
    data_temp['low'] = LOW
    data_temp['high'] = HIGH
    close_open_diff = data_temp['close'] - data_temp['open']
    fee = (close_open_diff) * settings['slippage']
    data_temp['sign_close_diff'] = np.where(abs(close_open_diff) > fee, 1.0, 0.0)

    # drop nan values
    data_temp.dropna(axis=0, inplace=True)
    split_rate = 0.8
    split = int(split_rate * len(data_temp))
    y = data_temp.loc[:, 'sign_close_diff']
    X = data_temp.drop(['sign_close_diff'], axis=1)
    # y[y == -1] = 0


    X_train = X[:split]
    X_test = X[split:]
    y_train = y[:split]
    y_test = y[split:]

    cls = SVC(C=1000.0, cache_size=200, class_weight=None, coef0=0.0,
              decision_function_shape=None, degree=3, gamma=0.0001, kernel='rbf',
              max_iter=-1, probability=False, random_state=None, shrinking=True,
              tol=0.001, verbose=False)

    pipeline = make_pipeline(
        StandardScaler(),
        cls,
    )

    pipeline.fit(X_train, y_train)
    y_pred = pipeline.predict(X_test)
    y_pred_series = pd.Series(y_pred)

    # settings['X_train'] = pd.concat([X_train, settings['X_train']], ignore_index=True)
    # settings['X_test'] = pd.concat([X_test, settings['X_test']], ignore_index=True)
    # settings['y_train'] = pd.concat([y_train, settings['y_train']], ignore_index=True,sort=True)
    # settings['y_test'] = pd.concat([y_test, settings['y_test']], ignore_index=True,sort=True)
    # settings['y_pred'] = pd.concat([y_pred_series, settings['y_pred']], ignore_index=True, sort=True)

    # import pdb;
    # pdb.set_trace()

    unique, counts = np.unique(y_pred, return_counts=True)
    y_dict = dict(zip(unique, counts))
    if (not (1.0 in y_dict)):
        y_dict[1.0] = 0
    if (not (-1.0 in y_dict)):
        y_dict[-1.0] = 0

    # print(y_dict)
    y_pred = np.where(y_dict[1.0] > y_dict[-1.0], 1.0, -1.0)
    # save old data

    return y_pred.item(0)