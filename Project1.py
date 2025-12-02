#Importing Libraries
import pandas as pd
import numpy as np
import cvxpy as cp
from betas import calculate_beta
from monteCarlo import monte_carlo_sim
import warnings
warnings.filterwarnings('ignore', category=RuntimeWarning)

def optimise_portfolio(data_path):
    target_data = {}
    
    with open(data_path, "r") as file:
        for line in file:
            parsed_line = line.strip().split(", ")
            if len(parsed_line) >= 3:
                ticker = parsed_line[0].strip().upper()
                target_price = float(parsed_line[1].strip())
                target_horizon = int(parsed_line[2].strip())
                target_data[ticker] = {
                    'target_price': target_price,
                    'target_horizon': target_horizon
                }
    
    input_tickers = list(target_data.keys())

    MAX_POSITION_SIZE = 1 # 0.25  # Maximum weight for any single stock
    MAX_SECTOR_ALLOCATION = 1 # 0.40  # Maximum allocation to any single sector

    #TASK 1:INPUTS

    stock_prices = pd.read_csv(r"data\stock_prices.csv", parse_dates = ["date"])
    #clarification, parse_dates added for DateTime casting
    sxp = pd.read_csv(r"data\s&p_data.csv", parse_dates = ["Date"])
    end_date = sxp["Date"].max() #Recent datapoints
    start_date = end_date - pd.Timedelta(days=365) #last year
    filtered_data = stock_prices[(stock_prices["date"] >= start_date) & (stock_prices["date"] <= end_date)].copy()


    days_traded_stock = filtered_data.groupby("ticker")["date"].nunique()
    all_qualified_tickers = days_traded_stock[days_traded_stock >= 150].index #List of STOCK TICKERS
    #150 days chosen but can be adjusted if more qualified stock tickers are needed

    # Filter to only include tickers that are in our input file AND have sufficient trading data
    qualified_input_tickers = [ticker for ticker in input_tickers if ticker in all_qualified_tickers]
    
    if len(qualified_input_tickers) == 0:
        print("ERROR: No input tickers have sufficient trading data (>= 150 days)")
        return
    
    if len(qualified_input_tickers) < len(input_tickers):
        missing_tickers = set(input_tickers) - set(qualified_input_tickers)
        print(f"WARNING: {len(missing_tickers)} tickers from input file lack sufficient trading data: {missing_tickers}")
    
    randomised_tickers = pd.Index(qualified_input_tickers)

    latest_data = filtered_data.sort_values("date").groupby("ticker").tail(1).set_index("ticker")
    latest_price = latest_data["close"].reindex(randomised_tickers)
    sectors = latest_data["sector"].reindex(randomised_tickers)

    target_prices = [target_data[ticker]['target_price'] for ticker in randomised_tickers]
    target_horizons = [target_data[ticker]['target_horizon'] for ticker in randomised_tickers]
    
    market_data_path = r'data\s&p_data.csv'
    stock_data_path = r"data\stock_prices.csv"
    betas = calculate_beta(randomised_tickers, start_date, end_date, market_data_path, stock_data_path).dropna()
    
    # only keep tickers with valid betas
    final_tickers = betas.index
    if len(final_tickers) != len(randomised_tickers):
        print(f"Dropped {len(randomised_tickers) - len(final_tickers)} tickers due to insufficient beta data")
    
    latest_price = latest_price.reindex(final_tickers)
    sectors = sectors.reindex(final_tickers)
    target_prices = [target_data[ticker]['target_price'] for ticker in final_tickers]
    target_horizons = [target_data[ticker]['target_horizon'] for ticker in final_tickers]
    betas = betas.reindex(final_tickers)

    #Making the DataFrame from inputs

    inputs_df = pd.DataFrame({
        "ticker_name": final_tickers,
        "latest_price": latest_price.values,
        "target_price": target_prices,
        "target_horizon": target_horizons,
        "beta": betas.values,
        "sector": sectors.values
    })

    def expected_return(target_price, current_price, horizon_months):
        ratio = target_price / current_price
        annual_factor = 12.0 / horizon_months
        expected_return = (ratio ** annual_factor) - 1
        return min(expected_return, 10.0)  # Cap at 1000%
    
    inputs_df["expected_return"] = [
        expected_return(row["target_price"], row["latest_price"], row["target_horizon"])
        for _, row in inputs_df.iterrows()
    ]

    price_set = filtered_data[filtered_data["ticker"].isin(final_tickers)].copy()
    price_set = price_set.sort_values(["ticker", "date"])
    price_set["daily_return"] = price_set.groupby("ticker")["close"].pct_change()
    price_set["daily_return"] = price_set["daily_return"].clip(-0.5, 0.5)  # Cap extreme returns
    
    new_returns = price_set.pivot(index="date", columns="ticker", values="daily_return").fillna(0)
    covariance_matrix = new_returns.cov()


    returns = inputs_df["expected_return"].values
    betas = inputs_df["beta"].values
    sectors = inputs_df["sector"].values
    new_covariance_matrix = covariance_matrix.reindex(index=final_tickers, columns=final_tickers).values
    n = len(final_tickers)
    weights_vector = cp.Variable(n)
    target_task = cp.Minimize(cp.quad_form(weights_vector, new_covariance_matrix))

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
    optimal_weights = np.maximum(optimal_weights, 0)
    optimal_weights = optimal_weights / np.sum(optimal_weights)

    inputs_df["optimal_weights"] = optimal_weights
    expected_portfolio_return = returns @ optimal_weights
    portfolio_volatility = np.sqrt(optimal_weights.T @ new_covariance_matrix @ optimal_weights) * np.sqrt(252)
    portfolio_beta = betas @ optimal_weights
    sharpe_ratio = expected_portfolio_return / portfolio_volatility if portfolio_volatility > 0 else 0

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

if __name__ == "__main__":
    optimise_portfolio("test.csv")