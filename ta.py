import talib as ta
import get_data
import numpy as np
from numpy import genfromtxt

example_candlestick =[
    [
        1499040000000,      # Open time
        "0.01634790",       # Open
        "0.80000000",       # High
        "0.01575800",       # Low
        "0.01577100",       # Close
        "148976.11427815",  # Volume
        1499644799999,      # Close time
        "2434.19055334",    # Quote asset volume
        308,                # Number of trades
        "1756.87402397",    # Taker buy base asset volume
        "28.46694368",      # Taker buy quote asset volume
        "17928899.62484339" # Can be ignored
    ]
]

data = genfromtxt("data/BTCUSDT_6h_26-04-2018_25-04-2021.csv", delimiter=",")
rsi = ta.RSI(data[:, 4])
overbought = data[np.where(rsi<30)]
for index, ob in enumerate(overbought):
    print(get_data.timeStampConverter(ob), rsi[np.where(rsi<30)][index])