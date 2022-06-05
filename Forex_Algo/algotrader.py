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
    def __init__(self, conf_file, instrument, bar_length, start, end, begin_cash, risk, leverage, ratio):
        """
        Parameters
        ==========
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
        begin_cash: float
            beginning account balance 
        risk: float
            risk ratio parameter to trigger stop losses expressed in decimals
        leverage: float
            leverate ratio available, or 1/margin rate
        ratio: float
            ratio of total available margin willing to risk
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
        self.ratio = ratio
        self.results = None
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
        df["returns"] = np.log(df["c"] / df["c"].shift(1))
        self.raw_data = df.copy()
        df["spread"] = df["ask"] - df["bid"]
        df["trading_cost"] = (df["spread"]/2) / df["c"]
        self.data = df.copy()

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

    def update_strategy(self):
        """ Updates position, trades, and strategy returns based on open and close signals given by trading rules.
        """        

        # reset trade positions, trades, and strategy returns
        self.data = self.data.drop(["position", "trades", "strategy", "strategy_tc"], axis=1)
        df = self.data.copy()

        # update trade positions
        df["position"] = np.where((df["open"] != 0) & (df["open"].shift(1) != df["open"]), df["open"], np.nan)
        df["position"] = np.where(df["close"] != 0, 0, df["position"])
        df["position"] = df["position"].fillna(method="ffill")
        df["position"] = df["position"].fillna(0)

        # determine when a trade takes place and calculate strategy returns
        df["trades"] = df["position"].diff().fillna(0)
        df["strategy"] = df["position"].shift(1) * df["returns"]
        df["strategy"] = df["strategy"].fillna(0)
        df["strategy_tc"] = df["strategy"] - df["trading_cost"] * df["trades"].abs()
        df["strategy"] = df["strategy"] * self.leverage
        df["strategy_tc"] = df["strategy_tc"] * self.leverage
        self.data = df.copy()

    def update_portfolio(self):
        """ Updates portfolio balance and margin 
        """        
        self.data = self.data.drop(["units", "portfolio_cost", "margin_used", "portfolio_value", "profit_loss", "real_return", "balance", "margin_available", "margin_closeout_ratio"], axis=1)
        df = self.data.copy().reset_index()
        open_price = {1: "ask", -1: "bid"}
        close_price = {1: "bid", -1: "ask"}
        
        for row in df.iterrows():
            if row[0] == 0:
                df.loc[row[0],"units"] = 0
                df.loc[row[0],"portfolio_cost"] = 0
                df.loc[row[0],"margin_used"] = 0
                df.loc[row[0],"portfolio_value"] = 0
                df.loc[row[0],"profit_loss"] = 0
                df.loc[row[0],"real_return"] = 0
                df.loc[row[0],"balance"] = self.begin_cash

            # when in an active position
            elif df.loc[row[0],"position"] != 0: 

                # when there is no trade and portfolio maintains previous active position
                if df.loc[row[0],"trades"] == 0: 
                    df.loc[row[0],"units"] = df.loc[row[0]-1,"units"]
                    df.loc[row[0],"portfolio_cost"] = df.loc[row[0]-1,"portfolio_cost"]
                    df.loc[row[0],"margin_used"] = df.loc[row[0]-1,"margin_used"]

                # when a trade is executed and portfolio opens an active position
                else:

                    # when only 1 trade occurs and portfolio opens active position
                    if abs(df.loc[row[0],"trades"]) == 1:
                        df.loc[row[0],"units"] = np.floor(df.loc[row[0]-1,"balance"]*self.ratio*self.leverage/df.loc[row[0], open_price[df.loc[row[0],"position"]]])
                    
                    # when 2 trades are executed and portfolio closes previous position and opens a new active position
                    else:
                        df.loc[row[0],"units"] = np.floor((df.loc[row[0]-1,"margin_used"] + ((df.loc[row[0]-1,"units"]*df.loc[row[0], close_price[df.loc[row[0]-1,"position"]]]) - df.loc[row[0]-1,"portfolio_cost"]))*self.ratio*self.leverage/df.loc[row[0], open_price[df.loc[row[0],"position"]]])

                    df.loc[row[0],"portfolio_cost"] = df.loc[row[0],"units"]*df.loc[row[0], open_price[df.loc[row[0],"position"]]]
                    df.loc[row[0],"margin_used"] = df.loc[row[0],"portfolio_cost"]/self.leverage                    

                # when in an active position irregardless if any trades occur
                df.loc[row[0],"portfolio_value"] = df.loc[row[0],"units"]*df.loc[row[0], close_price[df.loc[row[0],"position"]]]
                df.loc[row[0],"profit_loss"] = (df.loc[row[0],"portfolio_value"] - df.loc[row[0],"portfolio_cost"]) * df.loc[row[0],"position"]
                df.loc[row[0],"real_return"] = df.loc[row[0],"profit_loss"] / df.loc[row[0],"margin_used"] 
                df.loc[row[0],"balance"] = df.loc[row[0],"margin_used"] + df.loc[row[0],"profit_loss"]

            # when not in an active position 
            else:
                df.loc[row[0],"units"] = 0
                df.loc[row[0],"portfolio_cost"] = 0
                df.loc[row[0],"margin_used"] = 0
                df.loc[row[0],"portfolio_value"] = 0

                # when a trade is executed and the portfolio closes previous position
                if df.loc[row[0],"trades"] != 0:
                    df.loc[row[0],"profit_loss"] = (df.loc[row[0]-1,"units"]*df.loc[row[0], close_price[df.loc[row[0]-1,"position"]]] - df.loc[row[0]-1,"portfolio_cost"]) * df.loc[row[0]-1,"position"]
                    df.loc[row[0],"real_return"] = df.loc[row[0],"profit_loss"] / df.loc[row[0]-1,"margin_used"]
                    df.loc[row[0],"balance"] = df.loc[row[0]-1,"margin_used"] + df.loc[row[0],"profit_loss"]
                
                # when no trades are executed and the portfolio is not in an active position
                else:
                    df.loc[row[0],"profit_loss"] = 0
                    df.loc[row[0],"real_return"] = 0
                    df.loc[row[0],"balance"] = df.loc[row[0]-1,"balance"]

        df = df.fillna(method="ffill")
        df["margin_available"] = np.maximum(0, df["balance"] - df["margin_used"])
        df["margin_closeout_ratio"] = np.where(df["margin_used"] == 0, 1.0, df["balance"] / df["margin_used"])
        self.data = self.data.join(df.set_index("time")[["units", "portfolio_cost", "margin_used", "portfolio_value", "profit_loss", "real_return", "balance", "margin_available", "margin_closeout_ratio"]])

    def update_returns(self, strategy_type):
        """ Updates cumulative open trade returns based on selected strategy returns.
        """

        if strategy_type = "real_return":
            self.data["trade_return"] = self.data["real_return"]

        else:

            # reset cumulative trade returns
            self.data = self.data.drop(["trade_return"], axis=1)
            df = self.data.copy().reset_index()
            df["trade_return_add"] = np.nan

            # calculate open cumulative trade returns
            for row in df.iterrows():
                if row[0] == 0:
                    df.loc[row[0],"trade_return_add"] = 0
                elif df.loc[row[0], strategy_type] == 0:
                    df.loc[row[0], "trade_return_add"] = 0
                elif df.loc[row[0]-1,"trades"] != 0:
                    df.loc[row[0], "trade_return_add"] = df.loc[row[0], strategy_type]
                else:
                    df.loc[row[0], "trade_return_add"] = df.loc[row[0]-1, "trade_return_add"] + df.loc[row[0], strategy_type]
            df["trade_return"] = df["trade_return_add"].apply(np.exp) - 1
            self.data = self.data.join(df.set_index("time")[["trade_return"]])

    def test_strategy(self, strategy_type, risk_stop = True, swing_stop = False):
        """ Backtests the trading strategy.
        Parameters
        ==========
        risk_stop: boolean
            determine whether to update stop loss close signals based on max loss risk parameter
        swing_stop: boolean
            determine whether to update stop loss close signals based on recent swing highs and lows
        """      
        df = self.data.copy().dropna()
        df["position"] = np.nan
        df["trades"] = np.nan
        df["strategy"] = np.nan
        df["strategy_tc"] = np.nan
        df["trade_return"] = np.nan
        df["units"] = np.nan
        df["portfolio_cost"] = np.nan
        df["margin_used"] = np.nan
        df["portfolio_value"] = np.nan
        df["profit_loss"] = np.nan
        df["real_return"] = np.nan
        df["balance"] = np.nan
        df["margin_available"] = np.nan
        df["margin_closeout_ratio"] = np.nan

        # determine initial strategy positions based on initial open and close signals
        df["open"] = np.where(df.filter(regex=("signal")).mean(axis=1).abs() == 1.0, df.filter(regex=("signal")).mean(axis=1), 0)
        df["position"] = df["open"].replace(to_replace=0, method="ffill")
        df["close"] = np.where(df["trend_macd_crossover_signal"] != df["position"], 1, 0)
        self.data = df.copy()
        self.update_strategy()
        self.update_portfolio()
        self.update_returns(strategy_type=strategy_type)

        # adjust close signal to exclude negative returns
        self.data["close"] = np.where(self.data["trade_return"] < 0, 0, self.data["close"])
        self.update_strategy()
        self.update_portfolio()
        self.update_returns(strategy_type=strategy_type)

        # adjust close signal for stop loss based on risk tolerance parameter
        if risk_stop == True:
            self.data["close"] = np.where(self.data["trade_return"] < -self.risk, 1, self.data["close"])
            self.update_strategy()
            self.update_portfolio()
            self.update_returns(strategy_type=strategy_type)

        # adjust close signal for stop loss based on recent swing high and low
        if swing_stop == True:
            self.data["recentPHL"] = np.where(self.data["position"] == 0, 0, np.where(self.data["position"] == 1, self.data["recentPL"], self.data["recentPH"]))
            self.data["close"] = np.where(self.data["position"].shift(1) == 0, self.data["close"],
                                          np.where((self.data["c"] - self.data["recentPHL"]) * self.data["position"] < 0, 1, self.data["close"])
                                          )
            self.update_strategy()
            self.update_portfolio()
            self.update_returns(strategy_type=strategy_type)

        # adjust close signal for margin closeout ratio
        self.data["close"] = np.where(self.data["margin_closeout_ratio"] < 0.5, 1, self.data["close"])
        self.update_strategy()
        self.update_portfolio()
        self.update_returns(strategy_type=strategy_type)

        # calculate cumulative strategy returns
        df = self.data.copy()
        df["creturns"] = df["returns"].cumsum().apply(np.exp)
        df["cstrategy"] = df["strategy"].cumsum().apply(np.exp)
        df["cstrategy_tc"] = df["strategy_tc"].cumsum().apply(np.exp)
        df["breturns"] = df["balance"]/self.begin_cash
        self.results = df.copy()
        self.data = df.copy()
       
        hold = df["creturns"].iloc[-1] # absolute performance of buy and hold strategy
        perf = df["cstrategy"].iloc[-1] # absolute performance of the strategy
        outperf = perf - hold # out-/underperformance of strategy
        perf_tc = df["cstrategy_tc"].iloc[-1] # absolute performance of the strategy accounting for trading costs
        outperf_tc = perf_tc - hold # out-/underperformance of strategy accounting for trading costs
        perf_r = df["breturns"].iloc[-1] # real performance of the strategy
        outperf_r = perf_r - hold # real out-/underperformance of strategy
        return round(hold, 6), round(perf, 6), round(outperf, 6), round(perf_tc, 6), round(outperf_tc, 6), round(perf_r, 6), round(outperf_r, 6)

    def plot_results(self):
        """ Plots the cumulative performance of the trading strategy compared to buy and hold.
        """
        if self.results is None:
            print("No results to plot yet. Run a strategy.")
        else:
            title = "{} Returns Comparison".format(self.instrument)
            self.results[["creturns", "cstrategy", "cstrategy_tc", "breturns"]].plot(title=title, figsize=(12, 8))


if __name__ == "__main__":
        
    trader = TrendAlgoBacktester(r"C:\Users\wangj\Documents\codingproj\Project Retire\Forex_Algo\oanda.cfg", "EUR_USD", "M15", "2022-01-01", "2022-05-13", 50000.0, 0.02, 50, 1.0)
    trader.ema_crossover(short = 50, longg = 200)
    trader.macd_crossover(short = 8, longg = 21, signal = 5)
    trader.rsi(window = 13)
    trader.rsi(window = 5)
    trader.consensus_trend()
    trader.ma_crossover(short = (5, "EMA"), longg = (11, "EMA"))
    trader.ma_crossover(short = (13, "EMA"), longg = (36, "SMA"))
    trader.volatility_osc(length = 100)
    trader.swing(left = 5, right = 5 )
    # trader.test_strategy(strategy_type = "strategy")
    # no_swing_stop = trader.results
    # trader.plot_results()
    trader.test_strategy(strategy_type = "strategy_tc", swing_stop = True)
    swing_stop = trader.results
    trader.plot_results()
    trader.test_strategy(strategy_type = "real_return", swing_stop = True)
    swing_stop = trader.results
    trader.plot_results()