#NOTE: Monte Carlo must generate CORRELATED STOCK RETURNS

number_of_simulations = 5000 
horizon_days = 252 #simulating 1 year of trading
starting_portfolio_value = 1.0 #normalised henceforth
annual_returns = inputs_df["expected_return"].values
daily_mean_returns = (1+annual_returns) ** (1/252)-1 # converts annual to daily_returns = (1+annual_returns)^(1/252) - 1, known as daily drift
#I think this is more accurate than dividing by 252 as the above takes compounding growth into effect

#ERROR: got matrix is not positive definite so solution is the two statementss
eps = 1e-6 #small number
covariance_pd = new_covariance_matrix + np.eye(len(new_covariance_matrix))*eps #creates identitfy matrix same size as covariance matrix
#process is jittering to ensure positive definite
cholesky_covariance = np.linalg.cholesky(covariance_pd) #cholesky decomposition needed needed to create correlated shocks here!!

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
optimal_weights = np.array(weights_vector.value).flatten()
portfolio_daily_returns = correlated_daily_returns @ optimal_weights #single time series per simulation produced

portfolio_values = np.empty_like(portfolio_daily_returns)
#set day 0 manually due to recursive formula used V_t = V_(t-1) * (1+r_t)
portfolio_values[0] = starting_portfolio_value * (1 + portfolio_daily_returns[0])

for i in range(1, horizon_days):
    portfolio_values[i] = portfolio_values[i-1] * (1+ portfolio_daily_returns[i])

end_val = portfolio_values[-1] #last portfolio value 
final_returns = end_val -1

#confidence_level
p_value = 0.05 #95% value at risk
value_at_risk = np.quantile(final_returns, p_value)
conditional_value_at_risk = final_returns[final_returns <= value_at_risk].mean()
mean_return = np.mean(final_returns)
median_return = np.median(final_returns)
std_return = np.std(final_returns)
plt.hist(final_returns, bins = 100, color="blue", edgecolor = "k", alpha = 0.9)
plt.axvline(value_at_risk, color = "red", linestyle = "--", linewidth = 3, label = "VaR =" + format(value_at_risk, ".2%"))
plt.axvline(conditional_value_at_risk, color = "darkred", linestyle = ":", linewidth = 3, label = "CVaR =" + format(conditional_value_at_risk, ".2%"))
plt.axvline(mean_return, color = "green", linestyle = "--", linewidth = 1.5, label = "Mean =" + format(mean_return, ".2%"))
plt.axvline(value_at_risk, color = "blue", linestyle = "--", linewidth = 1.5, label = "Median =" + format(value_at_risk, ".2%"))

plt.title("A One Year Simulated Portfolio Returns with VaR / CVaR")
plt.xlabel("Portfolio Return")
plt.ylabel("Frequency (Number of Simulations)")
plt.legend()
plt.grid(alpha = 0.6)
plt.tight_layout()
plt.show()

print("Mean return :" + format(mean_return, ".2%"))
print("Median return:" + format(median_return, ".2%"))
print("Standard deviation:" + format(std_return, ".2%"))
print("Value at risk:" + format(value_at_risk, ".2%"))
print("Conditional value at risk:" + format(conditional_value_at_risk, ".2%"))
