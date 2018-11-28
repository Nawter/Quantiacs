import numpy as np
import pandas as pd
from sklearn import mixture as mix
import talib as ta
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC

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
    # settings['markets'] = ['CASH', 'F_CL', 'F_NG', 'F_HO', 'F_SB', 'F_RB', 'F_GC', 'F_C', 'F_W', 'F_S', 'F_HG',
    #                        'F_BO', 'F_SI', 'F_CC', 'F_TY', 'F_FV', 'F_US', 'F_JY', 'F_TU', 'F_ES', 'F_ED', 'VLO', 'ZTS',
    #                        'GOOG', 'AMZN', 'AAPL', 'BA', 'FB', 'NFLX', 'THC']

    settings['markets'] = ['F_CL']

    settings['beginInSample'] = '19900101'
    settings['endInSample'] = '20180101'
    settings['lookback'] = 87
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
            pos[0, i] = predict(DATE, OPEN[:, i], HIGH[:, i], LOW[:, i], CLOSE[:, i], settings)
        # for NaN data set position to 0
        except ValueError:
            pos[0, i] = 0.
    return pos, settings


################################# MY FUNCTIONS #################################
def predict(DATE, OPEN, HIGH, LOW, CLOSE, settings):
    # data_temp = pd.DataFrame([['Open','High','Low','Close']],index=['Date'])

    data_temp = pd.DataFrame()
    data_temp['Close'] = CLOSE
    data_temp['Open'] = OPEN
    data_temp['Low'] = LOW
    data_temp['High'] = HIGH
    data_temp['Date'] = DATE
    data_temp.set_index(['Date'], inplace=True)

    n = 10
    t = 0.70
    split = int(t * len(data_temp))

    data_temp['high'] = data_temp['High'].shift(1)
    data_temp['low'] = data_temp['Low'].shift(1)
    data_temp['close'] = data_temp['Close'].shift(1)
    data_temp['RSI'] = ta.RSI(np.array(data_temp['close']), timeperiod=n)
    data_temp['SMA'] = data_temp['close'].rolling(window=n).mean()
    data_temp['Corr'] = data_temp['SMA'].rolling(window=n).corr(data_temp['close'])
    data_temp['SAR'] = ta.SAR(np.array(data_temp['high']), np.array(data_temp['low']), \
                              0.2, 0.2)
    data_temp['ADX'] = ta.ADX(np.array(data_temp['high']), np.array(data_temp['low']), \
                              np.array(data_temp['close']), timeperiod=n)
    data_temp['Corr'][data_temp.Corr > 1] = 1
    data_temp['Corr'][data_temp.Corr < -1] = -1
    data_temp['Return'] = np.log(data_temp['Open'] / data_temp['Open'].shift(1))

    data_temp = data_temp.dropna()

    scaler = StandardScaler()
    gaussian = mix.GaussianMixture(n_components=4,
                                   covariance_type="spherical",
                                   n_init=150,
                                   random_state=42)
    data_temp = data_temp.drop(['High', 'Low', 'Close'], axis=1)
    X_train = data_temp[:split]
    X_test = data_temp[split:]
    y_train = (-1, data_temp.shape[1])
    gaussian.fit(np.reshape(scaler.fit_transform(X_train), y_train))
    regime = gaussian.predict(np.reshape(scaler.fit_transform(X_test), y_train))

    Regimes = pd.DataFrame(regime, columns=['Regime'], index=X_test.index).join(X_test, how='inner').assign(
        market_cu_return=X_test.Return.cumsum()).reset_index(drop=False).rename(columns={'index': 'Date'})
    order = [0, 1, 2, 3]

    # for i in order:
    #     print('Mean for regime %i: ' % i, gaussian.means_[i][0])
    #     print('Co-Variance for regime %i: ' % i, (gaussian.covariances_[i]))

    columns = Regimes.columns.drop(['Regime', 'Date'])
    scaler2 = StandardScaler()
    Regimes[columns] = scaler2.fit_transform(Regimes[columns])
    Regimes['Signal'] = 0
    Regimes.loc[Regimes['Return'] > 0, 'Signal'] = 1
    Regimes.loc[Regimes['Return'] < 0, 'Signal'] = -1
    Regimes['return'] = Regimes['Return'].shift(1)
    Regimes = Regimes.dropna()

    cls = SVC(C=1.0, cache_size=200, class_weight=None, coef0=0.0,
              decision_function_shape=None, degree=3, gamma='auto', kernel='rbf',
              max_iter=-1, probability=False, random_state=None, shrinking=True,
              tol=0.001, verbose=False)

    split2 = int(.8 * len(Regimes))
    X = Regimes.drop(['Signal', 'Return', 'market_cu_return', 'Date'], axis=1)
    y = Regimes['Signal']

    cls.fit(X[:split2], y[:split2])

    p_data = len(X) - split2
    data_temp['Pred_Signal'] = 0
    data_temp.iloc[-p_data:, data_temp.columns.get_loc('Pred_Signal')] = cls.predict(X[split2:])

    data_temp['str_ret'] = data_temp['Pred_Signal'] * data_temp['Return'].shift(-1)
    data_temp['strategy_cu_return'] = 0.
    data_temp['market_cu_return'] = 0.
    data_temp.iloc[-p_data:, data_temp.columns.get_loc('strategy_cu_return')] = np.nancumsum(
        data_temp['str_ret'][-p_data:])
    data_temp.iloc[-p_data:, data_temp.columns.get_loc('market_cu_return')] = np.nancumsum(
        data_temp['Return'][-p_data:])

    y_pred = data_temp['Pred_Signal']



    unique, counts = np.unique(y_pred, return_counts=True)
    y_dict = dict(zip(unique, counts))
    if (not (1.0 in y_dict)):
        y_dict[1.0] = 0
    if (not (-1.0 in y_dict)):
        y_dict[-1.0] = 0
    if (not (0.0 in y_dict)):
        y_dict[0.0] = 0



    # print(y_dict)
    y_pred = np.where(y_dict[1.0] > y_dict[-1.0], 1.0, 0)
    # save old data

    # import pdb
    # pdb.set_trace()

    return y_pred.item(0)
