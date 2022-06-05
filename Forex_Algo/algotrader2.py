import pandas as pd
import pandas_ta as ta
import numpy as np
# import math
import matplotlib.pyplot as plt
import time
import tpqoa
from datetime import datetime, timedelta
# from itertools import product
plt.style.use("seaborn")


class TrendAlgoBacktester(tpqoa.tpqoa):
    """ Class for the vectorized backtesting trading strategies.
    """

    def __init__(self, conf_file, instrument, bar_length, start, end, begin_cash, risk, leverage):
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
        self.bar_length = bar_length
        self.bar_number = int(bar_length[1:])
        self.start = start
        self.end = end
        self.tick_data = pd.DataFrame()
        self.raw_data = None
        self.data = None 
        self.last_bar = None
        self.begin_cash = begin_cash
        self.risk = risk
        self.leverage = leverage
        self.results = None
        # self.profits = []
        self.get_data()

    def get_data(self):
        """ Retrieves and prepares the data.

        """
        dfm = self.get_history(instrument = self.instrument, start = self.start, end = self.end,
                               granularity = self.bar_length, price = "M", localize = False)[["o", "h", "l", "c"]].dropna()
        dfa = self.get_history(instrument = self.instrument, start = self.start, end = self.end,
                               granularity = self.bar_length, price = "A", localize = False).c.dropna().to_frame()
        dfa.rename(columns={"c": "ask"}, inplace=True)
        dfb = self.get_history(instrument = self.instrument, start = self.start, end = self.end,
                               granularity = self.bar_length, price = "B", localize = False).c.dropna().to_frame()
        dfb.rename(columns={"c": "bid"}, inplace=True)
        df = pd.concat((dfm, dfa, dfb), axis=1)
        # df = df.resample(self.bar_length, label = "right").last().dropna()
        df["returns"] = np.log(df["c"] / df["c"].shift(1))*self.leverage
        self.raw_data = df.copy()
        self.data = self.raw_data.copy()

    def ema_crossover(self, short, longg):
        """ Returns EMA Crossover Signal based on input parameters.

        Parameters
        ==========
        short: int
            Short EMA window
        longg: int
            Long EMA window
        """
        df = self.data.copy()
        df["ema_{}".format(short)] = df["c"].ewm(span=short, min_periods=short).mean()
        df["ema_{}".format(longg)] = df["c"].ewm(span=longg, min_periods=longg).mean()
        df["ema_crossover_{}_{}".format(short, longg)] = df["ema_{}".format(short)] - df["ema_{}".format(longg)]
        df["ema_crossover_{}_{}_signal".format(short, longg)] = np.where(df["ema_crossover_{}_{}".format(short, longg)] > 0, 1,
                                                                          np.where(df["ema_crossover_{}_{}".format(short, longg)] < 0, -1, 0)
                                                                          )
        self.data = df.copy()

    def macd_crossover(self, short, longg, signal):
        """ Returns MACD Crossover Signal based on input parameters.

        Parameters
        ==========
        short: int
            Short EMA window
        longg: int
            Long EMA window
        signal: int
            MACD EMA smoothing window
        """
        df = self.data.copy()
        df.ta.macd(close="c", fast=8, slow=21, signal=5, append=True)
        df["trend_macd_crossover_signal"] = np.where(df["MACDh_{}_{}_{}".format(short, longg, signal)] > 0, 1,
                                               np.where(df["MACDh_{}_{}_{}".format(short, longg, signal)] < 0, -1, 0)
                                               )
        self.data = df.copy()

    def rsi(self, window):
        """ Returns RSI Signal based on input window.

        Parameters
        ==========
        window: int
            Window parameter for RSI calculation
        """
        df = self.data.copy()
        df.ta.rsi(close="c", length=window, append=True)
        df["trend_rsi_{}_signal".format(window)] = np.where(df["RSI_{}".format(window)] > 50, 1,
                                                      np.where(df["RSI_{}".format(window)] < 50, -1, 0)
                                                      )
        self.data = df.copy()

    def consensus_trend(self):
        """ Returns whether MACD Crossover and RSI Trends are consistent.

        """
        df = self.data.copy()
        df["consensus_signal"] = 0
        df["consensus_signal"] = np.where(df.filter(regex=("trend")).sum(axis=1)/df.filter(regex=("trend")).count(axis=1) >= 1, 1,
                                          np.where(df.filter(regex=("trend")).sum(axis=1)/df.filter(regex=("trend")).count(axis=1) <= -1, -1, 0)
                                          )
        df["consensus_signal"] = np.where(df["consensus_signal"].shift(1) != 0, 0, df["consensus_signal"])
        self.data = df.copy()

    def ma_crossover(self, short, longg):
        """ Returns MACD Crossover Signal based on input parameters.

        Parameters
        ==========
        short: tuple
            short moving averagae parameter tuple (int, string) with window size and moving average type ("EMA", "SMA") 
        longg: tuple
            long moving averagae parameter tuple (int, string) with window size and moving average type ("EMA", "SMA") 
        """
        df = self.data.copy()
        if short[1] == "EMA":
            df["mac_short"] = df["c"].ewm(span=short[0], min_periods=short[0]).mean()
        if short[1] == "SMA":
            df["mac_short"] = df["c"].rolling(short[0], min_periods=short[0]).mean()
        if longg[1] == "EMA":
            df["mac_long"] = df["c"].ewm(span=longg[0], min_periods=longg[0]).mean()
        if longg[1] == "SMA":
            df["mac_long"] = df["c"].rolling(longg[0], min_periods=longg[0]).mean()
        df["MAC_{}{}_{}{}".format(short[1], short[0], longg[1], longg[0])] = df["mac_short"] - df["mac_long"]
        df["MAC_{}{}_{}{}_signal".format(short[1], short[0], longg[1], longg[0])] = np.where(df["MAC_{}{}_{}{}".format(short[1], short[0], longg[1], longg[0])] > 0, 1,
                                                                                             np.where(df["MAC_{}{}_{}{}".format(short[1], short[0], longg[1], longg[0])] < 0, -1, 0)
                                                                                             )
        self.data = df.copy()

    def volatility_osc(self, length):
        """ Returns volatilitiy oscillator based on input length window.

        Parameters
        ==========
        length: int
            window parameter for volatility standard deivation calculation
        """        
        df = self.data.copy()
        df["spike"] = df["c"] - df["o"]
        df["upper"] = df["spike"].rolling(length, min_periods=length).std()
        df["lower"] = df["spike"].rolling(length, min_periods=length).std() * -1
        df["vol_signal_{}".format(length)] = np.where((df["spike"] > 0) & (df["spike"] - df["upper"] > 0), 1,
                                                      np.where((df["spike"] < 0) & (df["spike"] - df["lower"] < 0), -1, 0)
                                                      )
        self.data = df.copy()

    def swing(self, left, right):
        """ Returns swign high and lows based on left and right window parameters.

        Parameters
        ==========
        left: int
            window parameter for historical periods 
        right: int
            window parameter for future periods
        """        
        df = self.data.copy()
        df["swing_d_{}".format(right+1)] = df["h"].shift(0)
        for x in range(1, right+1):
            df["swing_d_{}".format(x)] = df["h"].shift(-x)
        for x in range(1, left+1):
            df["swing_d_{}".format(right+1+x)] = df["h"].shift(x)
        df["maxPH"] = df.filter(regex=("swing_d_")).max(axis=1)
        df["PH"] = np.where(df["maxPH"] == df["swing_d_{}".format(right+1)], df["swing_d_{}".format(right+1)], np.nan)
        df["recentPH"] = df["PH"].shift(right).astype(float).fillna(method="ffill")

        df["swing_d_{}".format(right+1)] = df["l"].shift(0)
        for x in range(1, right+1):
            df["swing_d_{}".format(x)] = df["l"].shift(-x)
        for x in range(1, left+1):
            df["swing_d_{}".format(right+1+x)] = df["l"].shift(x)
        df["minPL"] = df.filter(regex=("swing_d_")).min(axis=1)
        df["PL"] = np.where(df["minPL"] == df["swing_d_{}".format(right+1)], df["swing_d_{}".format(right+1)], np.nan)
        df["recentPL"] = df["PL"].shift(right).astype(float).fillna(method="ffill")
        self.data = self.data.join(df[["recentPL", "recentPH"]])

    def update_position(self):
        """ Updates position based on open and close signals given by trading rules.

        """        

        # reset trade positions
        self.data = self.data.drop(["position"], axis=1)
        df = self.data.copy()

        # update trade positions
        df["position"] = np.where((df["open"] != 0) & (df["open"].shift(1) != df["open"]), df["open"], np.nan)
        df["position"] = np.where(df["close"] != 0, 0, df["position"])
        df["position"] = df["position"].fillna(method="ffill")
        df["position"] = df["position"].fillna(0)
        self.data = df.copy()

    def update_returns(self):
        """ Updates trade returns based on open and close signals given by trading rules.

        """

        # reset trades, strategy, and returns
        self.data = self.data.drop(["trades", "strategy", "trade_return", "trade_return_cum"], axis=1)
        df = self.data.copy().reset_index()

        # determine when a trade takes place
        df["trades"] = df["position"].diff().fillna(0)
        df["strategy"] = df["position"].shift(1) * df["returns"]
        df["strategy"] = df["strategy"].fillna(0)

        for row in df.iterrows():
            if row[0] == 0:
                df.loc[row[0],"trade_return"] = 0
            elif df.loc[row[0],"strategy"] == 0:
                df.loc[row[0], "trade_return"] = 0
            elif df.loc[row[0]-1,"trades"] != 0:
                df.loc[row[0], "trade_return"] = df.loc[row[0], "strategy"]
            else:
                df.loc[row[0], "trade_return"] = df.loc[row[0]-1, "trade_return"] + df.loc[row[0], "strategy"]
        df["trade_return_cum"] = df["trade_return"].apply(np.exp) - 1
        self.data = self.data.join(df.set_index("time")[["trades", "strategy", "trade_return", "trade_return_cum"]])

    def update_portfolio(self):
        """ Updates portfolio holdings and cash position.

        """
        self.data = self.data.drop(["units", "trade_investment", "cash"], axis=1)
        df = self.data.copy().reset_index()
        df["units"] = np.nan
        df["trade_investment"] = np.nan
        df["margin_used"] = np.nan
        df["at_risk"] = np.nan
        df["pl"] = np.nan
        df["cash"] = np.nan
        df["balance"] = np.nan
        df["margin_available"] = np.nan
        
        for row in df.iterrows():
            if row[0] == 0:
                df.loc[row[0],"units"] = 0
                df.loc[row[0],"trade_investment"] = 0
                df.loc[row[0],"margin_used"] = 0
                df.loc[row[0],"at_risk"] = 0
                df.loc[row[0],"pl"] = 0
                df.loc[row[0],"cash"] = self.begin_cash
                df.loc[row[0],"balance"] = df.loc[row[0],"cash"]
                df.loc[row[0],"margin_available"] = df.loc[row[0],"balance"] - df.loc[row[0],"margin_used"]
            elif df.loc[row[0],"trades"] != 0:
                if df.loc[row[0], "position"] == 1:
                    df.loc[row[0],"units"] = np.floor(df.loc[row[0]-1, "balance"]*self.leverage/df.loc[row[0],"ask"])
                    df.loc[row[0],"trade_investment"] = df.loc[row[0],"units"]*df.loc[row[0],"ask"]
                    df.loc[row[0],"margin_used"] = df.loc[row[0],"trade_investment"]/self.leverage
                    df.loc[row[0],"at_risk"] = df.loc[row[0],"units"]*df.loc[row[0],"bid"]/self.leverage
                    df.loc[row[0],"pl"] = df.loc[row[0]-1,"units"]*df.loc[row[0],"bid"] - df.loc[row[0],"trade_investment"]
                elif df.loc[row[0], "position"] == -1:
                    df.loc[row[0],"units"] = np.floor(df.loc[row[0]-1, "balance"]*self.leverage/df.loc[row[0],"bid"])
                    df.loc[row[0],"trade_investment"] = df.loc[row[0],"units"]*df.loc[row[0],"bid"]
                    df.loc[row[0],"margin_used"] = df.loc[row[0],"trade_investment"]/self.leverage
                    df.loc[row[0],"at_risk"] = df.loc[row[0],"units"]*df.loc[row[0],"ask"]/self.leverage
                    df.loc[row[0],"pl"] = df.loc[row[0]-1,"units"]*df.loc[row[0],"ask"] - df.loc[row[0],"trade_investment"]
                else:
                    df.loc[row[0],"units"] = 0
                    df.loc[row[0],"trade_investment"] = 0
                    df.loc[row[0],"margin_used"] = 0
                    df.loc[row[0],"at_risk"] = 0
            else:
                df.loc[row[0],"units"] = df.loc[row[0]-1,"units"]
                df.loc[row[0],"trade_investment"] = df.loc[row[0]-1,"trade_investment"]
                df.loc[row[0],"margin_used"] = df.loc[row[0]-1,"margin_used"]
                df.loc[row[0],"at_risk"] = df.loc[row[0]-1,"at_risk"]

            df.loc[row[0],"cash"] = (df.loc[row[0]-1,"balance"]*self.leverage - df.loc[row[0],"trade_investment"])/self.leverage
            df.loc[row[0],"balance"] = df.loc[row[0]-1,"balance"] + df.loc[row[0],"pl"]
            df.loc[row[0],"margin_available"] = df.loc[row[0],"balance"] - df.loc[row[0],"margin_used"]
        
        df["units"] = df["units"].fillna(method="ffill")
        df["cash"] = df["cash"].fillna(method="ffill")
        self.data = self.data.join(df.set_index("time")[["units", "cash"]])

    def test_strategy(self, risk_stop = True, swing_stop = False):
        """ Backtests the trading strategy.

        Parameters
        ==========
        risk_stop: boolean
            determine whether to update stop loss close signals based on max loss risk parameter
        swing_stop: boolean
            determine whether to update stop loss close signals based on recent swing highs and lows
        """      
        df = self.data.copy().dropna()
        df["units"] = np.nan
        df["open_value"] = np.nan
        df["margin_used"] = np.nan
        df["portfolio_value"] = np.nan
        df["pl"] = np.nan
        df["cash"] = np.nan
        df["balance"] = np.nan
        df["open_balance"] = np.nan
        df["margin_available"] = np.nan

        # determine initial strategy positions based on initial open and close signals
        df["open"] = np.where(df.filter(regex=("signal")).mean(axis=1).abs() == 1.0, df.filter(regex=("signal")).mean(axis=1), 0)
        df["position"] = df["open"].replace(to_replace=0, method="ffill")
        df["close"] = np.where(df["trend_macd_crossover_signal"] != df["position"], 1, 0)
        self.data = df.copy()
        self.update_position()
        self.update_returns()

        # adjust close signal to exclude negative returns
        self.data["close"] = np.where(self.data["trade_return_cum"] < 0, 0, self.data["close"])
        self.update_position()
        self.update_returns()

        # adjust close signal for stop loss based on risk tolerance parameter
        if risk_stop == True:
            self.data["close"] = np.where(self.data["trade_return_cum"] < -self.risk, 1, self.data["close"])
            self.update_position()
            self.update_returns()

        # adjust close signal for stop loss based on recent swing high and low
        if swing_stop == True:
            self.data["close"] = np.where(self.data["position"].shift(1) == 0, self.data["close"], 
                                          np.where(self.data["position"].shift(1) == 1, np.where(self.data["c"] < df["recentPL"], 1, self.data["close"]),
                                                   np.where(self.data["c"] > self.data["recentPH"], 1, self.data["close"])
                                                   )
                                          )
            self.update_position()
            self.update_returns()

        # calculate portfolio holdings and cash balance
        df = self.data.copy().reset_index()
        df["units"] = np.nan
        df["cash"] = np.nan
        df["at_risk"] = np.nan
        df["balance"] = np.nan
        
        for row in df.iterrows():
            if row[0] == 0:
                df.loc[row[0],"units"] = 0
                df.loc[row[0],"cash"] = self.begin_cash
                df.loc[row[0],"at_risk"] = 0
            elif df.loc[row[0],"trades"] != 0:
                if df.loc[row[0], "position"] == 1:
                    if df.loc[row[0], "trades"] != 2:
                        df.loc[row[0],"units"] = np.floor(df.loc[row[0]-1, "cash"]*self.leverage/df.loc[row[0],"ask"])
                        df.loc[row[0],"cash"] = (df.loc[row[0]-1,"cash"]*self.leverage - df.loc[row[0],"units"]*df.loc[row[0],"ask"])/self.leverage
                    else:
                        df.loc[row[0],"units"] = np.floor((df.loc[row[0]-1, "units"]*df.loc[row[0], "bid"] + df.loc[row[0]-1, "cash"])*self.leverage/df.loc[row[0],"ask"])
                        df.loc[row[0],"cash"] = (df.loc[row[0]-1,"units"]*df.loc[row[0],"bid"] + df.loc[row[0]-1,"cash"]*self.leverage - df.loc[row[0],"units"]*df.loc[row[0],"ask"])/self.leverage
                elif df.loc[row[0], "position"] == -1:
                    if df.loc[row[0], "trades"] != -2:
                        df.loc[row[0],"units"] = np.floor(df.loc[row[0]-1, "cash"]*self.leverage/df.loc[row[0], "bid"])
                        df.loc[row[0],"cash"] = (df.loc[row[0]-1,"cash"]*self.leverage - df.loc[row[0],"units"]*df.loc[row[0],"bid"])/self.leverage
                    else:
                        df.loc[row[0],"units"] = np.floor((df.loc[row[0]-1, "units"]*df.loc[row[0], "ask"] + df.loc[row[0]-1, "cash"])*self.leverage/df.loc[row[0], "bid"])
                        df.loc[row[0],"cash"] = (df.loc[row[0]-1,"units"]*df.loc[row[0],"ask"] + df.loc[row[0]-1,"cash"]*self.leverage - df.loc[row[0],"units"]*df.loc[row[0],"bid"])/self.leverage
                elif df.loc[row[0], "trades"] == -1:
                    df.loc[row[0],"units"] = 0
                    df.loc[row[0],"cash"] = df.loc[row[0]-1,"cash"] + df.loc[row[0]-1,"units"]/self.leverage*df.loc[row[0],"bid"]
                else:
                    df.loc[row[0],"units"] = 0
                    df.loc[row[0],"cash"] = df.loc[row[0]-1,"cash"] + df.loc[row[0]-1,"units"]/self.leverage*df.loc[row[0],"ask"]
            else:
                df.loc[row[0],"units"] = df.loc[row[0]-1, "units"]
                df.loc[row[0],"cash"] = df.loc[row[0]-1, "cash"]
        
        df["units"] = df["units"].fillna(method="ffill")
        df["cash"] = df["cash"].fillna(method="ffill")
        self.data = self.data.join(df.set_index("time")[["units", "cash"]])

        # calculate at risk position value and total account balance
        df = self.data.copy()      
        df["at_risk"] = np.where(df["position"] == 1, df["units"]*df["bid"]/self.leverage, 
                                 np.where(df["position"] == -1, df["units"]*df["ask"]/self.leverage, 0)
                                 )
        df["balance"] = df["at_risk"] + df["cash"]

        # calculate strategy returns accounting for trading costs
        df["spread"] = df["ask"] - df["bid"]
        df["strategy_tc"] = np.where(df["trades"].shift(1) !=0, df["strategy"] - df["spread"].shift(1)/2*df["trades"].shift(1).abs()/df["c"].shift(1)*self.leverage, df["strategy"])

        # calculate cumulative strategy returns
        df["creturns"] = df["returns"].cumsum().apply(np.exp)
        df["cstrategy"] = df["strategy"].cumsum().apply(np.exp)
        df["cstrategy_tc"] = df["strategy_tc"].cumsum().apply(np.exp)
        df["breturns"] = df["balance"]/self.begin_cash
        self.results = df.copy()
        self.data = df.copy()
       
        hold = df["creturns"].iloc[-1]
        perf = df["cstrategy"].iloc[-1] # absolute performance of the strategy
        outperf = perf - hold # out-/underperformance of strategy
        perf_tc = df["cstrategy_tc"].iloc[-1] # absolute performance of the strategy accounting for trading costs
        outperf_tc = perf_tc - hold # out-/underperformance of strategy accounting for trading costs
        perf_r = df["breturns"].iloc[-1] # real performance of the strategy
        outperf_r = perf_r - hold # real out-/underperformance of strategy
        return round(hold, 6), round(perf, 6), round(outperf, 6), round(perf_tc, 6), round(outperf_tc, 6), round(perf_r, 6), round(outperf_r, 6)

    def plot_results(self):
        """ Plots the cumulative performance of the trading strategy
        compared to buy and hold.
        """
        if self.results is None:
            print("No results to plot yet. Run a strategy.")
        else:
            title = "{} Returns Comparison".format(self.instrument)
            self.results[["creturns", "cstrategy", "breturns", "cstrategy_tc"]].plot(title=title, figsize=(12, 8))


if __name__ == "__main__":
        
    trader = TrendAlgoBacktester(r"C:\Users\wangj\Documents\codingproj\Project Retire\Forex_Algo\oanda.cfg", "EUR_USD", "M15", "2022-01-01", "2022-05-13", 50000.0, 0.02, 30)
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