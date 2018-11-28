import pandas as pd
import numpy as np
from sklearn.svm import SVC
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import TimeSeriesSplit


# Evaluate trading system defined in current file.
if __name__ == '__main__':
    import quantiacsToolbox

    result = quantiacsToolbox.runts(__file__)
    print('stats:', result['stats'])
    print('evalDate', result['evalDate'])
    print('runtime', result['runtime'])
    print('errorLog', result['errorLog'])

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
    settings['beginInSample'] = '19900101'
    settings['endInSample'] = '20180101'
    settings['lookback'] = 252
    settings['budget'] = 10 ** 6
    settings['slippage'] = 0.05
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
 # train and test using roll forward cross validation scheme
def roll_forward_cross_validation(target, features_scaled, model, train_test_splits):
     i = 1
     accuracy_list = []
     # f1_score_list = []
     for train_index, test_index in train_test_splits:
          print('Split :', i)
          X_train, X_test = features_scaled.values[train_index], features_scaled.values[test_index]
          y_train, y_test = target.values[train_index], target.values[test_index]
          model.fit(X_train, y_train)
          y_pred = model.predict(X_test)
          accuracy = accuracy_score(y_test, y_pred)
          # f1_score = f1_score(y_test, y_pred,average=None)
          print('accuracy for split %d = %f' % (i, accuracy))
          # print('f1-score for split %d = %f' % (i, f1_score))
          accuracy_list.append(accuracy)
          # f1_score.append(f1_score)
          i += 1

     return accuracy_list, test_index, y_pred

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

    rf = RandomForestClassifier(3, max_depth=5, random_state=1)

    # Cross-validation using roll-forward scheme
    scaler = StandardScaler()
    scaler.fit_transform(X)
    tscv = TimeSeriesSplit(n_splits=11)
    accuracy_list, test_index, y_pred = roll_forward_cross_validation(y, X, rf, tscv.split(X))
    # Average accuracy?
    print('average accuracy = ', np.mean(accuracy_list))
    # print('average f1-score = ', np.mean(f1_score_list))
    # y_pred_series = pd.Series(y_pred)
    # import pdb;
    # pdb.set_trace()

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