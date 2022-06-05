    def update_portfolio(self):
        """ Updates portfolio balance and margin 
        """        
        self.data = self.data.drop(["units", "portfolio_cost", "margin_used", "cash", "portfolio_value", "profit_loss", "real_return", "balance", "margin_available", "margin_closeout_ratio"], axis=1)
        df = self.data.copy().reset_index()
        open_price = {1: "ask", -1: "bid"}
        close_price = {1: "bid", -1: "ask"}
        
        for row in df.iterrows():
            if row[0] == 0:
                df.loc[row[0],"units"] = 0
                df.loc[row[0],"portfolio_cost"] = 0
                df.loc[row[0],"margin_used"] = 0
                df.loc[row[0],"cash"] = self.begin_cash
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
                    df.loc[row[0],"cash"] = df.loc[row[0]-1,"cash"]

                # when a trade is executed and portfolio opens an active position
                else:

                    # when only 1 trade occurs and portfolio opens active position
                    if abs(df.loc[row[0],"trades"]) == 1:
                        df.loc[row[0],"units"] = np.floor(df.loc[row[0]-1,"balance"]*self.ratio*self.leverage/df.loc[row[0], open_price[df.loc[row[0],"position"]]])
                        df.loc[row[0],"portfolio_cost"] = df.loc[row[0],"units"]*df.loc[row[0], open_price[df.loc[row[0],"position"]]]
                        df.loc[row[0],"margin_used"] = df.loc[row[0],"portfolio_cost"]/self.leverage
                        df.loc[row[0],"cash"] = df.loc[row[0]-1,"cash"] - df.loc[row[0],"margin_used"]
                    
                    # when 2 trades are executed and portfolio closes previous position and opens a new active position
                    else:
                        df.loc[row[0],"units"] = np.floor(df.loc[row[0]-1,"cash"] + (df.loc[row[0]-1,"margin_used"] + ((df.loc[row[0]-1,"units"]*df.loc[row[0], close_price[df.loc[row[0]-1,"position"]]]) - df.loc[row[0]-1,"portfolio_cost"]))*self.ratio*self.leverage/df.loc[row[0], open_price[df.loc[row[0],"position"]]])
                        df.loc[row[0],"portfolio_cost"] = df.loc[row[0],"units"]*df.loc[row[0], open_price[df.loc[row[0],"position"]]]
                        df.loc[row[0],"margin_used"] = df.loc[row[0],"portfolio_cost"]/self.leverage
                        df.loc[row[0],"cash"] = df.loc[row[0]-1,"cash"] + (df.loc[row[0]-1,"margin_used"] + ((df.loc[row[0]-1,"units"]*df.loc[row[0], close_price[df.loc[row[0]-1,"position"]]]) - df.loc[row[0]-1,"portfolio_cost"])) - df.loc[row[0],"margin_used"]

                # when in an active position irregardless if any trades occur
                df.loc[row[0],"portfolio_value"] = df.loc[row[0],"units"]*df.loc[row[0], close_price[df.loc[row[0],"position"]]]
                df.loc[row[0],"profit_loss"] = (df.loc[row[0],"portfolio_value"] - df.loc[row[0],"portfolio_cost"]) * df.loc[row[0],"position"]
                df.loc[row[0],"real_return"] = df.loc[row[0],"profit_loss"] / df.loc[row[0],"margin_used"] 
                df.loc[row[0],"balance"] = df.loc[row[0],"margin_used"] + df.loc[row[0],"profit_loss"] + df.loc[row[0],"cash"]

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
                    df.loc[row[0],"balance"] = df.loc[row[0]-1,"margin_used"] + df.loc[row[0],"profit_loss"] + df.loc[row[0]-1,"cash"]
                
                # when no trades are executed and the portfolio is not in an active position
                else:
                    df.loc[row[0],"profit_loss"] = 0
                    df.loc[row[0],"real_return"] = 0
                    df.loc[row[0],"balance"] = df.loc[row[0]-1,"balance"]

                df.loc[row[0],"cash"] = df.loc[row[0],"balance"]

        df = df.fillna(method="ffill")
        df["margin_available"] = np.maximum(0, df["balance"] - df["margin_used"])
        df["margin_closeout_ratio"] = np.where(df["margin_used"] == 0, 1.0, df["balance"] / df["margin_used"])
        self.data = self.data.join(df.set_index("time")[["units", "portfolio_cost", "margin_used", "cash", "portfolio_value", "profit_loss", "real_return", "balance", "margin_available", "margin_closeout_ratio"]])
