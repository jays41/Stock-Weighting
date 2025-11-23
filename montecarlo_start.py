#NOTE: Monte Carlo must generate CORRELATED STOCK RETURNS

number_of_simulations = 5000 
horizon_days = 252 #simulating 1 year of trading
starting_portfolio_value = 1.0 #normalised henceforth
annual_returns = inputs_df["expected_return"].values
daily_mean_returns = (1+annual_returns) ** (1/252)-1 # converts annual to daily_returns = (1+annual_returns)^(1/252) - 1, known as daily drift
#I think this is more accurate than dividing by 252 as the above takes compounding growth into effect

cholesky_covariance = np.linalg.cholesky(new_covariance_matrix) #cholesky decomposition needed needed to create correlated shocks here!!

number_of_stocks = len(randomised_tickers)
matrix = np.random.normal(size = (horizon_days, number_of_simulations, number_of_stocks))#generating random INDEPENDENT shocks
#creates 3 dimenstional tensor of days * simulations * stocks
#genereates actual pure shocks

correlated_daily_returns = np.empty_like(matrix)
for i in range(horizon_days):
    correlated_daily_returns[i] = matrix[i] @ cholesky_covariance.T + daily_mean_returns #each shock is now a correlated return and the drift is needed to shift returns
#independent shock + drift

#convert stock returns to portfolio returns
#shape: (days, simulations)
porfolio_daily_returns = correlated_daily_returns @ optimal_weights #single time series per simulation produced

porfolio_values = np.empty_like(portfolio_daily_returns)
#set day 0 manually due to recursive formula used V_t = V_(t-1) * (1+r_t)
porfolio_values[0] = starting_portfolio_value * (1 + portfolio_daily_returns[0])

for i in range(1, horizon_days):
    portfolio_values[i] = portfolio_values[i-1] * (1+ portfolio_daily_returns[i])

end_val = portfolio_values[-1] #last portfolio value 
final_returns = end_val -1
