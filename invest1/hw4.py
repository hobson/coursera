Bollinger_val = (price - rolling_mean) / (rolling_std)
sum(below5diff)
below5diff.sum()
below5diff.sum().sum()
history
reload(sim)
s = sim.normalize_symbols('sp5002012')
len(s)
Bollinger_val = (close - close.rolling_mean(window=20)) / (close.rolling_std(window=20))
import pandas as pd
Bollinger_val = (close - pd.rolling_mean(close, window=20)) / (pd.rolling_std(close, window=20))
Bollinger_val[0]
Bollinger_val['AAPL']
Bollinger_val['AAPL'].plot()
