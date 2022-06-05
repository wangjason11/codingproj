

    def update_portfolio(self):
        """ Updates portfolio portfolio_values and cash position.

        """
        self.data = self.data.drop(["units", "open_value", "cash"], axis=1)
        df = self.data.copy().reset_index()
        df["units"] = np.nan
        df["open_value"] = np.nan
        df["margin_used"] = np.nan
        df["portfolio_value"] = np.nan
        df["pl"] = np.nan
        df["cash"] = np.nan
        df["balance"] = np.nan
        df["margin_available"] = np.nan
        trade_price = {1: "ask", 2: "bid"}
        close_price = {1: "bid", 2: "ask"}
        
        for row in df.iterrows():
            if row[0] == 0:
                df.loc[row[0],"units"] = 0
                df.loc[row[0],"open_value"] = 0
                df.loc[row[0],"margin_used"] = 0
                df.loc[row[0],"portfolio_value"] = 0
                df.loc[row[0],"pl"] = 0
                df.loc[row[0],"cash"] = self.begin_cash
                df.loc[row[0],"balance"] = self.begin_cash
                df.loc[row[0],"margin_available"] = df.loc[row[0],"balance"] - df.loc[row[0],"margin_used"]
            elif df.loc[row[0],"position"] != 0:
                if df.loc[row[0],"trades"] != 0:
                    df.loc[row[0],"units"] = np.floor(df.loc[row[0]-1, "balance"]*self.leverage/df.loc[row[0], trade_price[df.loc[row[0],"position"]]])
                    df.loc[row[0],"open_value"] = df.loc[row[0],"units"]*df.loc[row[0], trade_price[df.loc[row[0],"position"]]]
                else:
                    df.loc[row[0],"units"] = df.loc[row[0]-1,"units"]
                    df.loc[row[0],"open_value"] = df.loc[row[0]-1,"open_value"]
                df.loc[row[0],"portfolio_value"] = df.loc[row[0],"units"]*df.loc[row[0], close_price[df.loc[row[0],"position"]]]
                df.loc[row[0],"pl"] = df.loc[row[0],"portfolio_value"] - df.loc[row[0],"open_value"]
            else:
                df.loc[row[0],"units"] = 0
                df.loc[row[0],"open_value"] = 0
                df.loc[row[0],"portfolio_value"] = 0
                if df.loc[row[0],"trades"] != 0:
                    df.loc[row[0],"pl"] = df.loc[row[0]-1,"units"]*df.loc[row[0], close_price[df.loc[row[0]-1,"position"]]] - df.loc[row[0]-1,"open_value"]
                else:
                    df.loc[row[0],"pl"] = 0
            
            df.loc[row[0],"balance"] = df.loc[row[0]-1,"balance"] + df.loc[row[0],"pl"]
            df.loc[row[0],"margin_used"] = df.loc[row[0],"open_value"]/self.leverage
            df.loc[row[0],"margin_available"] = df.loc[row[0],"balance"] - df.loc[row[0],"margin_used"]\



            df.loc[row[0],"cash"] = (df.loc[row[0]-1,"balance"]*self.leverage - df.loc[row[0],"open_value"])/self.leverage

            df.loc[row[0],"cash"] = (df.loc[row[0]-1,"balance"]*self.leverage - df.loc[row[0],"open_value"])/self.leverage






                    dfu.loc[row[0],"cash"] = dfu.loc[row[0]-1,"cash"] - dfu.loc[row[0],"units"]/self.leverage*dfu.loc[row[0],"ask"]
                    dfu.loc[row[0],"cash"] = dfu.loc[row[0]-1,"units"]/self.leverage*dfu.loc[row[0],"bid"] + dfu.loc[row[0]-1,"cash"] - dfu.loc[row[0],"units"]/self.leverage*dfu.loc[row[0],"ask"]


                


                df.loc[row[0],"units"] = np.floor(df.loc[row[0]-1, "balance"]*self.leverage/df.loc[row[0],"ask"])
                df.loc[row[0],"open_value"] = df.loc[row[0],"units"]*df.loc[row[0],"ask"]
                df.loc[row[0],"margin_used"] = df.loc[row[0],"open_value"]/self.leverage
                df.loc[row[0],"portfolio_value"] = df.loc[row[0],"units"]*df.loc[row[0],"bid"]/self.leverage
                df.loc[row[0],"pl"] = df.loc[row[0]-1,"units"]*df.loc[row[0],"bid"] - df.loc[row[0],"open_value"]






                else:
                    df.loc[row[0],"units"] = 0
                    df.loc[row[0],"open_value"] = 0
                    df.loc[row[0],"margin_used"] = 0
                    df.loc[row[0],"portfolio_value"] = 0
            else:






            elif df.loc[row[0], "position"] == 1:
                if df.loc[row[0],"trades"] != 0:
                    df.loc[row[0],"units"] = np.floor(df.loc[row[0]-1, "balance"]*self.leverage/df.loc[row[0],"ask"])
                    df.loc[row[0],"open_value"] = df.loc[row[0],"units"]*df.loc[row[0],"ask"]
                    df.loc[row[0],"margin_used"] = df.loc[row[0],"open_value"]/self.leverage
                    df.loc[row[0],"portfolio_value"] = df.loc[row[0],"units"]*df.loc[row[0],"bid"]/self.leverage
                df.loc[row[0],"pl"] = df.loc[row[0]-1,"units"]*df.loc[row[0],"bid"] - df.loc[row[0],"open_value"]
            elif df.loc[row[0], "position"] == -1:
                if df.loc[row[0],"trades"] != 0:
                    df.loc[row[0],"units"] = np.floor(df.loc[row[0]-1, "balance"]*self.leverage/df.loc[row[0],"bid"])
                    df.loc[row[0],"open_value"] = df.loc[row[0],"units"]*df.loc[row[0],"bid"]
                    df.loc[row[0],"margin_used"] = df.loc[row[0],"open_value"]/self.leverage
                    df.loc[row[0],"portfolio_value"] = df.loc[row[0],"units"]*df.loc[row[0],"ask"]/self.leverage
                df.loc[row[0],"pl"] = df.loc[row[0]-1,"units"]*df.loc[row[0],"ask"] - df.loc[row[0],"open_value"]
            elif df.loc[row[0], "position"] == 0 and df.loc[row[0],"trades"] != 0:
                df.loc[row[0],"units"] = 0
                df.loc[row[0],"open_value"] = 0
                df.loc[row[0],"margin_used"] = 0
                df.loc[row[0],"portfolio_value"] = 0
            else:
                df.loc[row[0],"units"] = df.loc[row[0]-1,"units"]
                df.loc[row[0],"open_value"] = df.loc[row[0]-1,"open_value"]
                df.loc[row[0],"margin_used"] = df.loc[row[0]-1,"margin_used"]
                df.loc[row[0],"portfolio_value"] = df.loc[row[0]-1,"portfolio_value"]


        
        df["units"] = df["units"].fillna(method="ffill")
        df["cash"] = df["cash"].fillna(method="ffill")
        self.data = self.data.join(df.set_index("time")[["units", "cash"]])

