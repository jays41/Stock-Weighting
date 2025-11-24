#Importing Libraries
import pandas as pd
import numpy as np
import cvxpy as cp
from betas import calculate_beta
from monteCarlo import monte_carlo_sim

MAX_POSITION_SIZE = 0.25  # Maximum weight for any single stock
MAX_SECTOR_ALLOCATION = 0.40  # Maximum allocation to any single sector

#TASK 1:INPUTS
np.random.seed(20) #seeded to get same results each time, can change when needed

stock_prices = pd.read_csv(r"data\stock_prices.csv", parse_dates = ["date"])
#clarification, parse_dates added for DateTime casting
sxp = pd.read_csv(r"data\s&p_data.csv", parse_dates = ["Date"])
end_date = sxp["Date"].max() #Recent datapoints
start_date = end_date - pd.Timedelta(days=365) #last year
filtered_data = stock_prices[(stock_prices["date"] >= start_date) & (stock_prices["date"] <= end_date)].copy()


days_traded_stock = filtered_data.groupby("ticker")["date"].nunique()
qualified_tickers = days_traded_stock[days_traded_stock >= 150].index #List of STOCK TICKERS
#150 days chosen but can be adjusted if more qualified stock tickers are needed

# Keep randomised_tickers as pandas Index to maintain order
randomised_tickers = pd.Index(np.random.choice(qualified_tickers, size = min(20, len(qualified_tickers)))) #can adjust size!!
#can implement random.normal if needed with an array of tickers but not necessary... I think???

latest_data = filtered_data.sort_values("date").groupby("ticker").tail(1).set_index("ticker") #Oldest to newest with last row(recent closing price)
latest_price = latest_data["close"]
sectors = latest_data["sector"]

latest_prices_selected = latest_price.reindex(randomised_tickers)
sectors_selected = sectors.reindex(randomised_tickers)

#fake mock uplifts
mock_price_increase_values = np.random.uniform(0.20, 0.40, len(randomised_tickers)) #can change with actual input values when received
mock_price_increase = pd.Series(mock_price_increase_values, index= randomised_tickers)

#random months till it will hit target
target_horizon = pd.Series(np.random.choice([3,6,9,12], size = len(randomised_tickers)), index = randomised_tickers) #can change depending on what target_horizon is desired
target_price = latest_prices_selected * (1+ mock_price_increase)

#Calculate real betas using historical data
market_data_path = r'data\s&p_data.csv'
stock_data_path = r"data\stock_prices.csv"

betas = calculate_beta(randomised_tickers, start_date, end_date, market_data_path, stock_data_path)

# Drop any stocks with insufficient data (NaN betas)
valid_betas = betas.dropna()
dropped_tickers = betas[betas.isna()].index.tolist()

if len(dropped_tickers) > 0:
    print(f"Dropped {len(dropped_tickers)} tickers due to insufficient data: {dropped_tickers}")
    print(f"Tickers dropped: {dropped_tickers}")

randomised_tickers = valid_betas.index
betas = valid_betas

print(betas)

#Making the DataFrame from inputs

inputs_df = pd.DataFrame({
    "ticker_name": randomised_tickers,
    "latest_price": latest_prices_selected.values,
    "target_price": target_price.values,
    "target_horizon": target_horizon.values,
    "beta": betas.values,
    "sector": sectors_selected.values

})

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
sectors = inputs_df["sector"].values
new_covariance_matrix = covariance_matrix.reindex(index= randomised_tickers, columns = randomised_tickers).values
n = len(randomised_tickers)
weights_vector = cp.Variable(n)
target_task = cp.Minimize(cp.quad_form(weights_vector,new_covariance_matrix))

unique_sectors = list(set(sectors))
sector_matrix = np.zeros((len(unique_sectors), n))
for i, sector in enumerate(unique_sectors):
    sector_mask = sectors == sector
    sector_matrix[i] = sector_mask

base_conditions = [
    cp.sum(weights_vector) == 1,  # Fully invested
    weights_vector >= 0,  # Long only
    weights_vector <= MAX_POSITION_SIZE  # Position size limit
]

sector_conditions = []
for i in range(len(unique_sectors)):
    sector_conditions.append(sector_matrix[i] @ weights_vector <= MAX_SECTOR_ALLOCATION)

conditions = base_conditions + sector_conditions + [betas @ weights_vector == 1]
problem = cp.Problem(target_task, conditions)
problem.solve()

if problem.status != cp.OPTIMAL:
    print(f"Exact beta constraint failed with status: {problem.status}")
    
    # Try with relaxed beta constraint
    conditions_relaxed = base_conditions + sector_conditions + [
        betas @ weights_vector >= 0.95,
        betas @ weights_vector <= 1.05
    ]
    problem_relaxed = cp.Problem(target_task, conditions_relaxed)
    problem_relaxed.solve()
    
    if problem_relaxed.status != cp.OPTIMAL:
        print(f"Relaxed beta constraint (5%) failed with status: {problem_relaxed.status}")
        
        # Last resort: no beta constraint
        conditions_basic = base_conditions + sector_conditions
        problem_basic = cp.Problem(target_task, conditions_basic)
        problem_basic.solve()
        
        if problem_basic.status != cp.OPTIMAL:
            print(f"Optimisation failed completely with status: {problem_basic.status}")
            exit()
        else:
            print("Optimisation succeeded without beta constraint")
            problem = problem_basic
    else:
        print("Optimisation succeeded with relaxed beta constraint")
        problem = problem_relaxed
else:
    print("Optimisation succeeded with exact beta constraint")

#solution is found in the weights_vector where correct weightings of each stock ticker are found

#Task 5

optimal_weights = np.array(weights_vector.value)
optimal_weights = np.maximum(optimal_weights, 0)  # Set any tiny negative weights to 0
optimal_weights = optimal_weights / np.sum(optimal_weights)  # Renormalise so they sum to 1

inputs_df["optimal_weights"] = optimal_weights
expected_portfolio_return = returns @ optimal_weights #dot product of return and weights 

portfolio_volatility_daily = np.sqrt(optimal_weights.T @ new_covariance_matrix @ optimal_weights)
portfolio_volatility = portfolio_volatility_daily * np.sqrt(252)  # Annualise volatility
portfolio_beta = betas @ optimal_weights
sharpe_ratio = expected_portfolio_return / portfolio_volatility

# Validate constraints
beta_deviation = abs(portfolio_beta - 1.0)
if beta_deviation > 0.1:
    print(f"WARNING: Portfolio beta ({portfolio_beta:.3f}) deviates significantly from target (1.0)")

# Check sector constraints
print("\nSector Allocations:")
for sector in unique_sectors:
    sector_weight = inputs_df[inputs_df["sector"] == sector]["optimal_weights"].sum()
    status = "OK" if sector_weight <= MAX_SECTOR_ALLOCATION else "WARNING"
    print(f"{status}: {sector}: {sector_weight:.1%} (limit: {MAX_SECTOR_ALLOCATION:.1%})")

print("Portfolio Metrics:")
print("Expected return: {:.4f}".format(expected_portfolio_return))
print("Expected volatility: {:.4f}".format(portfolio_volatility))
print("Beta: {:.4f}".format(portfolio_beta))
print("Sharpe ratio: {:.4f}".format(sharpe_ratio))

print("Sector Exposures:")
sector_exposure = inputs_df.groupby("sector")["optimal_weights"].sum().sort_values()
print(sector_exposure)

print("Optimal Weights:")
print(inputs_df[["ticker_name", "optimal_weights"]].sort_values("optimal_weights"))


results = monte_carlo_sim(
    weights=optimal_weights,
    expected_returns=returns,
    cov_matrix=new_covariance_matrix,
    num_simulations=10000,
    time_horizon=252  # 1 year
)

print(f"Expected Return: {results['mean_return']:.2%}")
print(f"VaR (95%): {results['var_95']:.2%}")