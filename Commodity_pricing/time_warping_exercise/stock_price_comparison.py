# Compare stock price patterns
import numpy as np
from dtaidistance import dtw
from dtaidistance import dtw_visualisation as dtwviz

def compare_stock_patterns(prices1, prices2):
    # Normalize prices
    prices1_norm = (prices1 - np.mean(prices1)) / np.std(prices1)
    prices2_norm = (prices2 - np.mean(prices2)) / np.std(prices2)
    
    # Calculate DTW distance
    distance = dtw.distance(prices1_norm, prices2_norm)
    return distance

# Example usage
apple_prices = np.array([150, 152, 148, 155, 160])
google_prices = np.array([2800, 2850, 2790, 2900, 2950, 3000])

similarity = compare_stock_patterns(apple_prices, google_prices)
print(f"Stock pattern similarity: {similarity}")

# Stock pattern similarity: 0.6325293529421651