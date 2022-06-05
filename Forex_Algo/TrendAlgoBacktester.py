import pandas as pd
import pandas_ta as ta
import numpy as np
import matplotlib.pyplot as plt
import math
import time
import tpqoa
from datetime import datetime, timedelta
# from itertools import product
plt.style.use("seaborn")


class TrendAlgoBacktester(tpqoa.tpqoa):
    """ Class for the vectorized backtesting trading strategies.
    """

    def __init__(self, conf_file, instrument, bar_length, start, end, begin_cash, risk):
        """
        Parameters
        ----------
        conf_file: str
            path to and filename of the configuration file,
            e.g. "/home/me/oanda.cfg"
        instrument: str
            ticker symbol (instrument) to be backtested
        bar_length: str
            bar granularity, a string like "S5", "M1" or "D"
        start: str
            start date for data import
        end: str
            end date for data import
        begin_cash: double
            beginning account balance 
        risk: double
            risk ratio parameter to trigger stop losses expressed in decimals   
        """
        super().__init__(conf_file)
        self.instrument = instrument
        self.bar_length = pd.to_timedelta(bar_length)
        self.start = start
        self.end = end
        self.tick_data = pd.DataFrame()
        self.raw_data = None
        self.data = None 
        self.last_bar = None
        self.begin_cash = begin_cash
        # self.balance = begin_cash
        self.risk = risk
        self.results = None
        # self.position = 0
        # self.profits = []
        self.get_data()

    def get_data(self):
        """ Retrieves and prepares the data.

        """
        dfm = self.get_history(instrument = self.instrument, start = self.start, end = self.end,
                               granularity = self.bar_length, price = "M", localize = False)[["o", "h", "l", "c"]].dropna().to_frame()
        dfa = self.get_history(instrument = self.instrument, start = self.start, end = self.end,
                               granularity = self.bar_length, price = "A", localize = False).c.dropna().to_frame()
        dfa.rename(columns={"c": "ask"}, inplace=True)
        dfb = self.get_history(instrument = self.instrument, start = self.start, end = self.end,
                               granularity = self.bar_length, price = "B", localize = False).c.dropna().to_frame()
        dfb.rename(columns={"c": "bid"}, inplace=True)
        df = pd.concat(dfm, dfa, dfb, axis=1)
        df = df.resample(self.bar_length, label = "right").last().dropna()
        df["returns"] = np.log(df["c"] / df["c"].shift(1))
        df["cash"] = self.begin_cash
        df["units"] = 0
        df["at_risk"] = 0
        df["position"] = 0
        df["strategy"] = 0
        self.raw_data = df.copy()

    def ema_crossover(self, short, longg):
        ''' Returns EMA Crossover Signal based on input parameters.

        Parameters
        ==========
        short: int
            Short EMA window
        longg: int
            Long EMA window
        '''
        df = self.raw_data.copy().dropna()
        df["ema_{}".format(short)] = df["c"].ewm(span=short, min_periods=short).mean()
        df["ema_{}".format(longg)] = df["c"].ewm(span=longg, min_periods=longg).mean()
        df["ema_crossover_{}_{}".format(short, longg)] = df["ema_{}".format(short)] - df["ema_{}".format(longg)]
        df["ema_crossover_{}_{}_signal".format(short, longg)] = np.where(df["ema_crossover_{}_{}".format(short, longg)] > 0, 1,
                                                                          np.where(df["ema_crossover_{}_{}".format(short, longg)] < 0, -1, 0)
                                                                          )
        self.data = df.copy()

    def macd_crossover(self, short, longg, signal):
        ''' Returns MACD Crossover Signal based on input parameters.

        Parameters
        ==========
        short: int
            Short EMA window
        longg: int
            Long EMA window
        signal: int
            MACD EMA smoothing window
        '''
        df = self.raw_data.copy().dropna()
        df.ta.macd(close='c', fast=8, slow=21, signal=5, append=True)
        df["trend_macd_crossover_signal"] = np.where(df["MACDh_{}_{}_{}".format(short, longg, signal)] > 0, 1,
                                               np.where(df["MACDh_{}_{}_{}".format(short, longg, signal)] < 0, -1, 0)
                                               )
        self.data = df.copy()

    def rsi(self, window):
        ''' Returns RSI Signal based on input window.

        Parameters
        ==========
        window: int
            Window parameter for RSI calculation
        '''
        df = self.data.copy().dropna()
        df.ta.rsi(close='c', length=window, append=True)
        df["trend_rsi_{}_signal".format(window)] = np.where(df["RSI_{}".format(window)] > 50, 1,
                                                      np.where(df["RSI_{}".format(window)] < 50, -1, 0)
                                                      )
        self.data = df.copy()

    def consensus_trend(self):
        ''' Returns whether MACD Crossover and RSI Trends are consistent.

        '''
        df = self.data.copy().dropna()
        df["consensus_signal"] = 0
        df["consensus_signal"] = np.where(df["consensus_signal"].shift(1) != 0, 0,
                                          np.where(df.filter(regex=("trend")).sum(axis=1)/df.filter(regex=("trend")).count(axis=1) >= 1, 1,
                                                   np.where(df.filter(regex=("trend")).sum(axis=1)/df.filter(regex=("trend")).count(axis=1) <= -1, -1, 0)
                                                   )
                                          )
        self.data = df.copy()

    def ma_crossover(self, short, longg):
        ''' Returns MACD Crossover Signal based on input parameters.

        Parameters
        ==========
        short: tuple
            short moving averagae parameter tuple (int, string) with window size and moving average type ("EMA", "SMA") 
        longg: tuple
            long moving averagae parameter tuple (int, string) with window size and moving average type ("EMA", "SMA") 
        '''
        df = self.data.copy().dropna()
        if short[1] == "EMA":
            df["mac_short"] = df["c"].ewm(span=short, min_periods=short).mean()
        if short[1] == "SMA":
            df["mac_short"] = df["c"].rolling(short, min_periods=short).mean()
        if longg[1] == "EMA":
            df["mac_long"] = df["c"].ewm(span=longg, min_periods=longg).mean()
        if longg[1] == "SMA":
            df["mac_long"] = df["c"].rolling(longg, min_periods=longg).mean()
        df["MAC_{}{}_{}{}".format(short[1], short[0], longg[1], longg[0])] = df["mac_short"] - df["mac_long"]
        df["MAC_{}{}_{}{}_signal".format(short[1], short[0], longg[1], longg[0])] = np.where(df["MAC_{}{}_{}{}".format(short[1], short[0], longg[1], longg[0])] > 0, 1,
                                                                                             np.where(df["MAC_{}{}_{}{}".format(short[1], short[0], longg[1], longg[0])] < 0, -1, 0)
                                                                                             )
        self.data = df.copy()

    def volatility_osc(self, length):
        ''' Returns volatilitiy oscillator based on input length window.

        Parameters
        ==========
        length: int
            window parameter for volatility standard deivation calculation
        '''        
        df = self.data.copy().dropna()
        df["spike"] = df["c"] - df["o"]
        df["upper"] = df["spike"].rolling(length, min_periods=length).std()
        df["lower"] = df["spike"].rolling(length, min_periods=length).std() * -1
        df["vol_signal_{}".format(length)] = np.where((df["spike"] > 0) & (df["spike"] - df["upper"] > 0), 1,
                                                      np.where((df["spike"] < 0) & (df["spike"] - df["lower"] < 0), -1, 0)
                                                      )
        self.data = df.copy()

    def swing(self, left, right):
        ''' Returns swign high and lows based on left and right window parameters.

        Parameters
        ==========
        left: int
            window parameter for historical periods 
        right: int
            window parameter for future periods
        '''        
        df = self.data.copy().dropna()
        df["d_{}".format(left+1)] = df.h.shift(0)
        for x in range(1, left+1):
            df["d_{}".format(x)] = df.h.shift(-x)
        for x in range(1, right+1):
            df["d_{}".format(left+1+x)] = df.h.shift(x)
        df["maxPH"] = df.filter(regex=("d_")).max(axis=1)
        df["PH"] = np.where(df["maxPH"] == df["d_{}".format(left+1)], df["d_{}".format(left+1)], np.nan)

        df["d_{}".format(left+1)] = df.l.shift(0)
        for x in range(1, left+1):
            df["d_{}".format(x)] = df.l.shift(-x)
        for x in range(1, right+1):
            df["d_{}".format(left+1+x)] = df.l.shift(x)
        df["minPL"] = df.filter(regex=("d_")).min(axis=1)
        df["PL"] = np.where(df["minPL"] == df["d_{}".format(left+1)], df["d_{}".format(left+1)], np.nan)

        df["recentPL"] = df.PL.shift(2).astype(float).fillna(method="ffill")
        df["recentPH"] = df.PH.shift(2).astype(float).fillna(method="ffill")

        self.data = self.data.join(df[["recentPL", "recentPH"]])

    def update_position(self):
        ''' Updates position based on open and close signals given by trading rules.

        '''        
        df = self.data.copy().dropna()
        df["position"] = 0
        df["strategy"] = 0
        df["position"] = np.where((df["open"] != 0) & (df["position"].shift(1) != df["open"]), df["open"], 
                                  np.where((df["close"] != 0) & (df["position"].shift(1) != 0), 0, df["position"].shift(1))
                                  )

        # determine when a trade takes place
        df["trades"] = df.position.diff().fillna(0)

        df["strategy"] = df["position"].shift(1) * df["returns"]
        df["trade_return"] = np.where(df["strategy"] == 0, 0, 
                                      np.where(df["trades"].shift(1) != 0, df["strategy"], 
                                               (df["trade_return"].shift(1) + df["strategy"]).apply(np.exp)
                                               )
                                      )
        self.data = df.copy()

    def test_strategy(self, risk_stop = True, swing_stop = False):
        """ Backtests the trading strategy.

        """ 
        df = self.data.copy().dropna()

        # determine initial strategy open and close signals
        df["open"] = np.where(df.filter(regex=("signal")).mean().abs() == 1.0, df.filter(regex=("signal")).mean().astype(int), 0)
        # df["close"] = np.where(df["trend_macd_crossover_signal"] != df["trend_macd_crossover_signal"].shift(1), 1, 0)
        df["close"] = np.where(df["trend_macd_crossover_signal"] != df['open'].replace(to_replace=0, method='ffill'), 1, 0)
        self.update_position()

        # adjust close signal to exclude negative returns
        df["close"] = np.where((df["close"] == 1) & (df["trade_return"] < 0), 0, df["close"])
        self.update_position()

        # adjust close signal for stop loss based on risk tolerance parameter
        if risk_stop == True:
            df["close"] = np.where(df["trade_return"] < -self.risk, 1, df["close"])
            self.update_position()

        # adjust close signal for stop loss based on recent swing high and low
        if swing_stop == True:
            df["close"] = np.where(df["position"].shift(1) == 0, df["close"], 
                                   np.where(df["position"].shift(1) == 1, np.where(df["c"] < df["recentPL"], 1, df["close"]),
                                            np.where(df["c"] > df["recentPH"], 1, df["close"])
                                            )
                                   )
            self.update_position()


        df["units"] = np.where(df["trades"] == 0, df["units"].shift(1),
                               np.where(df["position"] == 1, 
                                        np.where(df["trades"] == 2, math.floor((df["units"].shift(1)*df["bid"] + df["cash"].shift(1))/df["ask"]), 
                                                 math.floor(df["cash"].shift(1)/df["ask"])
                                                 ),
                                        np.where(df["position"] == -1, 
                                                 np.where(df["trades"] == -2, math.floor((df["units"].shift(1)*df["ask"] + df["cash"].shift(1))/df["bid"]), 
                                                          math.floor(df["cash"].shift(1)/df["bid"])
                                                          ),
                                                 0
                                                 )
                                        )
                                 )

        df["at_risk"] = np.where(df["position"] == 1, df["units"]*df["bid"], 
                                 np.where(df["position"] == -1, df["units"]*df["ask"], 0)
                                 )

        df["cash"] = np.where(df["trades"] == 0, df["cash"].shift(1),
                              np.where(df["position"] == 1, 
                                       np.where(df["trades"] == 2, df["units"].shift(1)*df["bid"] + df["cash"].shift(1) - df["units"]*df["ask"],
                                                df["cash"].shift(1) - df["units"]*df["ask"]
                                                ),
                                       np.where(df["position"] == -1, 
                                                np.where(df["trades"] == -2, df["units"].shift(1)*df["ask"] + df["cash"].shift(1) - df["units"]*df["bid"],
                                                         df["cash"].shift(1) - df["units"]*df["bid"]
                                                         ),
                                                np.where(df["trades"] == -1, df["cash"].shift(1) + df["units"].shift(1)*df["bid"],
                                                         df["cash"].shift(1) + df["units"].shift(1)*df["ask"]
                                                         )
                                                )
                                       )
                              )

        df["balance"] = df["at_risk"] + df["cash"]

        # calculate cumulative strategy returns
        df["creturns"] = df["returns"].cumsum().apply(np.exp)
        df["cstrategy"] = df["strategy"].cumsum().apply(np.exp)
        df["breturns"] = df["balance"]/self.begin_cash - 1
        self.results = df
       
        perf = df["cstrategy"].iloc[-1] # absolute performance of the strategy
        r_perf = df["breturns"].iloc[-1] # real performance of the strategy
        outperf = perf - df["creturns"].iloc[-1] # out-/underperformance of strategy
        r_outperf = r_perf - df["creturns"].iloc[-1] # real out-/underperformance of strategy
        return round(perf, 6), round(outperf, 6), round(r_perf, 6), round(r_outperf, 6)

    def plot_results(self):
        """ Plots the cumulative performance of the trading strategy
        compared to buy and hold.
        """
        if self.results is None:
            print("No results to plot yet. Run a strategy.")
        else:
            title = "{} Returns Comparison".format(self.instrument)
            self.results[["creturns", "cstrategy", "breturns"]].plot(title=title, figsize=(12, 8))


if __name__ == "__main__":
        
    trader = TrendAlgoBacktester(r"C:\Users\wangj\Documents\codingproj\Project Retire\Forex_Algo\oanda.cfg", "EUR_USD", "15min", "2022-01-01", "2022-05-13", 50000.0, 0.02)
    trader.ema_crossover(short = 50, longg = 200)
    trader.macd_crossover(short = 8, longg = 21, signal = 5)
    trader.rsi(window = 13)
    trader.rsi(window = 5)
    trader.consensus_trend()
    trader.ma_crossover(short = (5, "EMA"), longg = (11, "EMA"))
    trader.ma_crossover(short = (13, "EMA"), longg = (36, "SMA"))
    trader.volatility_osc(length = 100)
    trader.swing(left = 5, right = 5 )
    trader.test_strategy()
    no_swing_stop = trader.results
    trader.plot_results()
    trader.test_strategy(swing_stop = True)
    swing_stop = trader.results
    trader.plot_results()