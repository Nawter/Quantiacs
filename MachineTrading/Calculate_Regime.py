from pandas_datareader import data as web
import numpy as np
import pandas as pd
from sklearn import mixture as mix
import seaborn as sns
import matplotlib.pyplot as plt
import fix_yahoo_finance as yf

#
# yf.pdr_override() # <== that's all it takes :-)
# stocks = ['AMZN','MSFT','BA']
#
# panel_data= web.get_data_yahoo(stocks,start= '2000-01-01', end='2017-01-01')
# close = panel_data['Close']

df = web.get_data_yahoo('SPY', start='1990-01-01', end='2017-01-01')
df = df[['Open', 'High', 'Low', 'Close']]
df['open'] = df['Open'].shift(1)
df['high'] = df['High'].shift(1)
df['low'] = df['Low'].shift(1)
df['close'] = df['Close'].shift(1)

df = df[['open', 'high', 'low', 'close']]

df = df.dropna()

unsup = mix.GaussianMixture(n_components=4,
                            covariance_type="spherical",
                            n_init=100,
                            random_state=42)

unsup.fit(np.reshape(df, (-1, df.shape[1])))
regime = unsup.predict(np.reshape(df, (-1, df.shape[1])))


df['Return']= np.log(df['close']/df['close'].shift(1))

Regimes = pd.DataFrame(regime, columns=['Regime'], index=df.index).join(df, how='inner').assign(
    market_cu_return=df.Return.cumsum()).reset_index(drop=False).rename(columns={'index': 'Date'})

order = [0, 1, 2, 3]
fig = sns.FacetGrid(data=Regimes, hue='Regime', hue_order=order, aspect=2, size=4)
fig.map(plt.scatter, 'Date', 'market_cu_return', s=4).add_legend()
plt.show()

for i in order:
    print('Mean for regime %i: ' % i, unsup.means_[i][0])
    print('Co-Variancefor regime %i: ' % i, (unsup.covariances_[i]))
