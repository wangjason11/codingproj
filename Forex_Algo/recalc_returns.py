def recalc_portfolio(self):
        """ Updates portfolio balance and margin 
        """        
        self.data = self.data.drop(["portfolio_value", "profit_loss", "real_return", "balance", "margin_available", "margin_closeout_ratio"], axis=1)
        df = self.data.copy().reset_index()
        open_price = {1: "ask", -1: "bid"}
        close_price = {1: "bid", -1: "ask"}

        df["price_diff"] = np.where(df["returns"] != df["returns_control"], )
        
        for row in df.iterrows():

            # when not in an active position 
            if df.loc[row[0],"position"] == 0:  

                # when a trade is executed and the portfolio closes previous position
                if abs(df.loc[row[0],"trades"]) == 1:
                    df.loc[row[0],"profit_loss"] = (df.loc[row[0]-1,"units"] * (df.loc[row[0], close_price[df.loc[row[0]-1,"position"]]] + df["price_diff"]) - df.loc[row[0]-1,"portfolio_cost"]) * df.loc[row[0]-1,"position"]
                    df.loc[row[0],"real_return"] = df.loc[row[0],"profit_loss"] / df.loc[row[0]-1,"margin_used"]
                    df.loc[row[0],"balance"] = df.loc[row[0]-1,"margin_used"] + df.loc[row[0],"profit_loss"]

                if abs(df.loc[row[0],"trades"]) == 2:
                    df.loc[row[0],"units"] = np.floor((df.loc[row[0]-1,"margin_used"] + ((df.loc[row[0]-1,"units"]*df.loc[row[0], close_price[df.loc[row[0]-1,"position"]]]) - df.loc[row[0]-1,"portfolio_cost"]))*self.ratio*self.leverage/df.loc[row[0], open_price[df.loc[row[0],"position"]]])
                    df.loc[row[0],"profit_loss"] = (df.loc[row[0]-1,"units"] * (df.loc[row[0], close_price[df.loc[row[0]-1,"position"]]] + df["price_diff"]) - df.loc[row[0]-1,"portfolio_cost"]) * df.loc[row[0]-1,"position"]
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

    def recalc_returns(self, strategy_type):
        """ Updates cumulative open trade returns based on selected strategy returns.
        """

        # reset cumulative trade returns
        self.data = self.data.drop(["trade_return"], axis=1)
        df = self.data.copy().reset_index()

        df["open_price"] = np.where((df["trades"] !=0) & (df["position"] != 0), df["c"], np.nan)
        df["open_price"] = df["open_price"].fillna(method="ffill")

        df["returns_control"] = np.where(df["trade_return"] < -0.02, -0.02, df["trade_return"])
        df["price_diff"] = np.where(df["returns_control"] != df["trade_return"], (df["returns_control"] - df["trade_return"]) * df["open_price"].shift(1) * df["position"].shift(1), 0) 


        df["strategy"] = df["position"].shift(1) * df["returns_control"]
        df["strategy"] = df["strategy"].fillna(0)
        df["strategy_tc"] = df["strategy"] - df["trading_cost"] * df["trades"].abs()
        df["strategy"] = df["strategy"] * self.leverage
        df["strategy_tc"] = df["strategy_tc"] * self.leverage


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

