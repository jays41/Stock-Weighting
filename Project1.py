#Importing Libraries
import pandas as pd
import numpy as np
import cvxpy as cp

#TASK 1:INPUTS
np.random.seed(20) #seeded to get same results each time, can change when needed

stock_prices = pd.read_csv("stock_prices.csv", parse_dates = ["date"])
#clarification, parse_dates added for DateTime casting
sxp = pd.read_csv("s&p_data.csv", parse_dates = ["Date"])
end_date = sxp["Date"].max() #Recent datapoints
start_date = end_date - pd.Timedelta(days=365) #last year
filtered_data = stock_prices[(stock_prices["date"] >= start_date) & (stock_prices["date"] <= end_date)].copy()


days_traded_stock = filtered_data.groupby("ticker")["date"].nunique()
qualified_tickers = days_traded_stock[days_traded_stock >= 150].index #List of STOCK TICKERS
#150 days chosen but can be adjusted if more qualified stock tickers are needed

randomised_tickers = np.random.choice(qualified_tickers, size = min(20, len(qualified_tickers))) #can adjust size!!
#can implement random.normal if needed with an array of tickers but not necessary... I think???

latest_data = filtered_data.sort_values("date").groupby("ticker").tail(1).set_index("ticker") #Oldest to newest with last row(recent closing price)
latest_price = latest_data["close"]
sectors = latest_data["sector"]

#fake mock uplifts
mock_price_increase_values = np.random.uniform(0.20, 0.40, len(randomised_tickers)) #can change with actual input values when received
mock_price_increase = pd.Series(mock_price_increase_values, index= randomised_tickers)

#random months till it will hit target
target_horizon = pd.Series(np.random.choice([3,6,9,12], size = len(randomised_tickers)), index = randomised_tickers) #can change depending on what target_horizon is desired
target_price = latest_price.reindex(randomised_tickers) * (1+ mock_price_increase) #reindex due to array length mismatch!!

#Fake betas
betas = pd.Series(np.random.uniform(0.7, 1.3, size = len(randomised_tickers)), index = randomised_tickers) #betas are randomised but can be set to 1 if needed.....
#chosen uniform, any better dists???

#Making the DataFrame from inputs

inputs_df = pd.DataFrame({
    "ticker_name": randomised_tickers,
    "latest_price": latest_price.reindex(randomised_tickers).values,
    "target_price": target_price.values,
    "target_horizon": target_horizon.values,
    "beta": betas.values,
    "sector": sectors.reindex(randomised_tickers).values

})

#made use of .reindex() to fix the mismatch of array lengths

#Task 2 Expected Returns



inputs_df["expected_return"] = (((inputs_df["target_price"]/ inputs_df["latest_price"])) ** (12/inputs_df["target_horizon"]) -1) #Expected return formula 


#Task 3 Covariance variance Matrix

price_set = filtered_data[filtered_data["ticker"].isin(randomised_tickers)].copy() # selecting the prices from the price dataset with respect to the chosen qualified tickers
price_set = price_set.sort_values(["ticker", "date"]) #reformatting columns
price_set["daily_return"] = price_set.groupby("ticker")["close"].pct_change() #percentage change of each ticker within each date
new_returns = price_set.pivot(index = "date", columns = "ticker", values = "daily_return") #just for helping with visualisation- ticker vs date col, row
covariance_matrix = new_returns.cov() #ticker vs ticker matrix



#Task 4 -Put returns + beta here, conditions: low risk, fully-invested, in line with S&P benchmark


returns = inputs_df["expected_return"].values
betas = inputs_df["beta"].values
new_covariance_matrix = covariance_matrix.reindex(index= randomised_tickers, columns = randomised_tickers).values #incase tickers order is switched and in specific place(got an error without this)
n = len(randomised_tickers)
weights_vector = cp.Variable(n) #vector of [w1,w2.....wn]
target_task = cp.Minimize(cp.quad_form(weights_vector,new_covariance_matrix)) #minimize the portfolio variance
#tried cp.sum_squares() but non-canonical!
conditions = [cp.sum(weights_vector) == 1, weights_vector >= 0, betas @ weights_vector == 1] #waiting for caps??? applied the other conditions
problem = cp.Problem(target_task, conditions)
problem.solve()
#solution is found in the weights_vector where correct weightings of each stock ticker are found

#Task 5

inputs_df["optimal_weights"] = np.array(weights_vector.value)
expected_portfolio_return = returns @ np.array(weights_vector.value) #dot product of return and weights 


portfolio_volatility = np.sqrt(np.array(weights_vector.value).T @ new_covariance_matrix @ np.array(weights_vector.value))
portfolio_beta = betas @ np.array(weights_vector.value)
sharpe_ratio = expected_portfolio_return / portfolio_volatility

#print("Expected return: {}".format(expected_portfolio_return))
#print("Expected volatility: {}".format(portfolio_volatility))
#print("Beta: {}".format(portfolio_beta))
#print("Sharpe ratio: {}".format(sharpe_ratio))


sector_exposure = inputs_df.groupby("sector")["optimal_weights"].sum().sort_values()
#print(sector_exposure)




