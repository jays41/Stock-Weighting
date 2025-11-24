import numpy as np
import matplotlib.pyplot as plt

def monte_carlo_sim(weights, expected_returns, cov_matrix, num_simulations=10000, time_horizon=252):
    portfolio_return_daily = (expected_returns @ weights) / 252
    portfolio_vol_daily = np.sqrt(weights.T @ cov_matrix @ weights)
    
    simulated_returns = np.random.normal(
        loc=portfolio_return_daily,
        scale=portfolio_vol_daily,
        size=(num_simulations, time_horizon)
    )
    
    cumulative_returns = np.cumprod(1 + simulated_returns, axis=1)
    final_values = cumulative_returns[:, -1]
    
    results = {
        'final_values': final_values,
        'paths': cumulative_returns,
        'mean_return': np.mean(final_values) - 1,
        'median_return': np.median(final_values) - 1,
        'percentile_5': np.percentile(final_values, 5) - 1,
        'percentile_95': np.percentile(final_values, 95) - 1,
        'var_95': 1 - np.percentile(final_values, 5),
    }

    portfolio_return_daily = (expected_returns @ weights) / 252
    portfolio_vol_daily = np.sqrt(weights.T @ cov_matrix @ weights)

    all_paths = []
    for _ in range(num_simulations):
        daily_returns = np.random.normal(portfolio_return_daily, portfolio_vol_daily, time_horizon)
        cumulative = np.cumprod(1 + daily_returns)
        all_paths.append(cumulative)

    all_paths = np.array(all_paths)

    # Plot 100 random paths
    sample_indices = np.random.choice(num_simulations, size=100, replace=False)
    for i in sample_indices:
        plt.plot(all_paths[i], color='lightblue', alpha=0.3, linewidth=0.5)

    mean_path = np.mean(all_paths, axis=0)
    median_path = np.median(all_paths, axis=0)
    p5 = np.percentile(all_paths, 5, axis=0)
    p95 = np.percentile(all_paths, 95, axis=0)

    plt.plot(mean_path, color='darkblue', linewidth=2.5, label='Mean', zorder=5)
    plt.plot(median_path, color='purple', linewidth=2, label='Median', linestyle=':', zorder=5)
    plt.plot(p5, color='red', linewidth=2, label='5th Percentile (VaR)', linestyle='--', zorder=5)
    plt.plot(p95, color='green', linewidth=2, label='95th Percentile', linestyle='--', zorder=5)

    plt.axhline(y=1, color='black', linewidth=1, linestyle='-', alpha=0.5)
    plt.xlabel('Trading Days')
    plt.ylabel('Portfolio Value (Starting at $1)')
    plt.title(f'Monte Carlo Simulation ({num_simulations:,} Simulations, 100 Shown)')
    plt.legend(loc='best')
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.show()

    final_values = all_paths[:, -1]
    print(f"\nMonte Carlo Results (1 Year):")
    print(f"Mean Return: {np.mean(final_values) - 1:.2%}")
    print(f"Median Return: {np.median(final_values) - 1:.2%}")
    print(f"5th Percentile: {np.percentile(final_values, 5) - 1:.2%}")
    print(f"95th Percentile: {np.percentile(final_values, 95) - 1:.2%}")
    print(f"Probability of Loss: {np.mean(final_values < 1):.2%}")
    
    return results