import numpy


def myTradingSystem(DATE, OPEN, HIGH, LOW, CLOSE, VOL, exposure, equity, settings):
    nMarkets = CLOSE.shape[1]
    pos = numpy.zeros(nMarkets)

    # periodLonger = 5120
    # # best scores
    # periodShorter = 2520
    # TWO_MONTHS = 252

    periodLonger = 5040
    periodShorter = 2520
    TWO_MONTHS = 42

    for markets in range(nMarkets):
        fiftytwoweek = CLOSE[-periodShorter, markets]
        movingaverage = numpy.sum(CLOSE[-periodLonger:, markets]) / periodLonger
        # If markets are  not in bear market
        if settings['periodOfDecline'][markets] != 1:
            if CLOSE[-1, markets] > movingaverage:
                if CLOSE[-1, markets] > fiftytwoweek:
                    settings['relativePos'][markets] = 1
                    if CLOSE[-1, markets] > settings['high'][markets]:
                        settings['high'][markets] = CLOSE[-1, markets]
            if settings['high'][markets] > 0:
                # If market is off the high by 15% exit position
                if (((CLOSE[-1, markets] - settings['high'][markets]) / settings['high'][markets])
                        <= - .1):
                    settings['relativePos'][markets] = 0
                    settings['high'][markets] = 0
                    settings['periodOfDecline'][markets] = 1
        elif (CLOSE[-1, markets] > CLOSE[-TWO_MONTHS, markets]) and CLOSE[-1, markets] > settings['high'][markets]:
            settings['periodOfDecline'][markets] = 0
            settings['high'][markets] = 0
        elif CLOSE[-1, markets] < movingaverage:
            settings['periodOfDecline'][markets] = 0
            settings['high'][markets] = 0

    # Give less weight to more volatile asset(Ten year note)
    for markets in range(nMarkets):
        if markets == 0:
            pos[0] = 1
        elif markets != 2 and settings['relativePos'][markets] > 0:
            pos[markets] = .3
        elif markets == 2 and settings['relativePos'][markets] > 0:
            pos[markets] = .9

    return pos, settings


def mySettings():
    settings = {}
    # settings['markets'] = ['CASH','KSU','VLO','ZTS','GOOG','AMZN','AAPL','BA','FB','NFLX','THC','F_US','F_TU','F_ES'] # 1.664
    # settings['markets'] = ['CASH','F_US','F_TU','F_ES','F_TY','F_AX','F_DZ','ZTS','GOOG','AMZN','AAPL',	'F_UB','BA','FB','MO'] #1.30
    settings['markets'] = ['CASH', 'F_CL', 'F_NG', 'F_HO', 'F_SB', 'F_RB', 'F_GC', 'F_C', 'F_W', 'F_S', 'F_HG',
                           'F_BO', 'F_SI', 'F_CC', 'F_TY', 'F_FV', 'F_US', 'F_JY', 'F_TU', 'F_ES', 'F_ED', 'VLO', 'ZTS',
                           'GOOG', 'AMZN', 'AAPL', 'BA', 'FB', 'NFLX', 'THC']
    markets_size = len(settings['markets'])

    settings['beginInSample'] = '20080102'
    settings['endInSample'] = '20180601'
    # best
    # settings['lookback'] = 2520
    settings['lookback'] = 2520

    settings['budget'] = 10 ** 6
    settings['slippage'] = 0.05
    settings['relativePos'] = numpy.zeros(markets_size)
    settings['high'] = numpy.zeros(markets_size)
    settings['low'] = numpy.zeros(markets_size)
    settings['periodOfDecline'] = numpy.zeros(markets_size)
    settings['periodOfIncrease'] = numpy.zeros(markets_size)

    return settings


# Evaluate trading system defined in current file.
if __name__ == '__main__':
    import quantiacsToolbox

    results = quantiacsToolbox.runts(__file__)
    print(results['stats'])
