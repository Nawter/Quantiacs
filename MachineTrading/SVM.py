import quantiacsToolbox
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn import svm
import talib
import sys
from sklearn.metrics import mean_squared_error, r2_score


class myStrategy(object):
    def myTradingSystem(self, DATE, OPEN, HIGH, LOW, CLOSE, VOL, OI, P, R, RINFO, exposure, equity, settings):
        """
        For 4 lookback days and 3 markets, CLOSE is a numpy array looks like
        [[ 12798.   11537.5   9010. ]
         [ 12822.   11487.5   9020. ]
         [ 12774.   11462.5   8940. ]
         [ 12966.   11587.5   9220. ]]
        """

        # define helper function
        # use close price predict the trend of the next day
        def predict(CLOSE, gap):
            lookback = CLOSE.shape[0]
            X = np.concatenate([CLOSE[i:i + gap] for i in range(lookback - gap)], axis=1).T
            y = np.sign((CLOSE[gap:lookback] - CLOSE[gap - 1:lookback - 1]).T[0])
            y[y == 0] = 1
            # print(X.shape)
            clf = svm.SVC()
            clf.fit(X, y)

            y_pred = clf.predict(CLOSE[-gap:].T)

            return y_pred

        nMarkets = len(settings['markets'])
        gap = settings['gap']

        pos = np.zeros((1, nMarkets), dtype='float')
        for i in range(nMarkets):
            try:
                pos[0, i] = predict(CLOSE[:, i].reshape(-1, 1),
                                    gap, )



            # for NaN data set position to 0
            except ValueError:
                pos[0, i] = 0.

        return pos, settings

    def mySettings(self):
        """ Define your trading system settings here """

        settings = {}

        # Futures Contracts
        settings['markets'] = ['CASH', 'VLO', 'ZTS', 'GOOG', 'AMZN', 'AAPL', 'BA', 'FB', 'NFLX', 'THC']

        # settings['markets'] = ['F_CL', 'F_NG', 'CASH']

        settings['lookback'] = 20
        settings['budget'] = 10 ** 6
        settings['slippage'] = 0.05

        settings['gap'] = 10

        return settings

result = quantiacsToolbox.runts(myStrategy)
print(result['stats'])