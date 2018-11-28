import numpy as np
import pandas as pd
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression

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
    settings['markets'] = ['CASH', 'KSU', 'VLO', 'ZTS', 'GOOG', 'AMZN', 'AAPL', 'BA', 'FB', 'NFLX', 'THC', 'F_US',
                           'F_TU', 'F_ES']
    settings['lookback'] = 87
    settings['budget'] = 10 ** 6
    settings['slippage'] = 0.05
    settings['market'] = ''
    settings['iterations'] = 0

    return settings


def myTradingSystem(DATE, OPEN, HIGH, LOW, CLOSE, VOL, OI, P, R, RINFO, exposure, equity, settings):
    nMarkets = len(settings['markets'])
    pos = np.zeros((1, nMarkets), dtype='float')
    for i in range(nMarkets):
        try:
            pos[0, i] = predict(OPEN[:, i], HIGH[:, i], LOW[:, i], CLOSE[:, i], VOL[:, i], settings)

        # for NaN data set position to 0
        except ValueError:
            pos[0, i] = 0.

    return pos, settings


################################# MY FUNCTIONS #################################
# define helper function
def predict(OPEN, HIGH, LOW, CLOSE, VOL, settings):
    lookback = settings['lookback']
    data_temp = pd.DataFrame()

    data_temp['close'] = CLOSE
    data_temp['open'] = OPEN
    data_temp['low'] = LOW
    data_temp['high'] = HIGH
    data_temp['vol'] = VOL

    # percentage change
    data_temp['close_percentage_change'] = data_temp['close'].pct_change() * 100
    data_temp['open_percentage_change'] = data_temp['open'].pct_change() * 100
    data_temp['low_percentage_change'] = data_temp['low'].pct_change() * 100
    data_temp['high_percentage_change'] = data_temp['high'].pct_change() * 100
    data_temp['vol_percentage_change'] = data_temp['vol'].pct_change() * 100

    open = 'open'
    high_open = data_temp[open].rolling(window=5).max()
    low_open = data_temp[open].rolling(window=5).min()
    high_low_open_diff = high_open - low_open
    open_diff = data_temp.loc[:, 'open'].diff(1)
    vol = 'vol'
    high_vol = data_temp[vol].rolling(window=5).max()
    low_vol = data_temp[vol].rolling(window=5).min()
    high_low_vol_diff = high_vol - low_vol
    vol_diff = data_temp.loc[:, 'vol'].diff(1)

    # fractional change
    data_temp['open_fractional_change'] = (open_diff / high_low_open_diff) * 100
    data_temp['vol_fractional_change'] = (vol_diff / high_low_vol_diff) * 100

    # labels
    change_close_open = (data_temp['close'] - data_temp['open']) / data_temp['open']
    close_open_diff = data_temp['close'] - data_temp['open']
    fee = (close_open_diff) * settings['slippage']
    data_temp['sign_close_diff'] = np.where(abs(close_open_diff) > fee, 1.0, 0.0)  # replace inf values with nan values

    data_temp.replace([np.inf, -np.inf], np.nan, inplace=True)

    # drop nan values
    data_temp.dropna(axis=0, inplace=True)

    # drop columns
    y = data_temp['sign_close_diff']
    X = data_temp.drop(['close', 'open', 'low', 'high', 'vol', 'sign_close_diff'], axis=1)
    # y[y == -1] = 0

    # split data
    split_rate = 0.8
    split = int(split_rate * len(data_temp))

    X_train = X[:split]
    X_test = X[split:]
    y_train = y[:split]
    y_test = y[split:]

    # declare the classifier
    lr = LogisticRegression()

    pipeline = make_pipeline(
        StandardScaler(),
        lr,
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
