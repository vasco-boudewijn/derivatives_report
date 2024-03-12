import numpy as np
import pandas as pd

def get_hurst_exponent(time_series, max_lag=20):
    """Returns the Hurst Exponent of the time series"""

    lags = range(2, max_lag)

    # variances of the lagged differences
    tau = [np.std(np.subtract(time_series[lag:], time_series[:-lag])) for lag in lags]

    # Check for zero standard deviation to avoid singular matrix problem
    tau = np.array(tau)
    nonzero_indices = np.where(tau != 0)[0]
    lags = np.array(list(lags[i] for i in nonzero_indices))
    tau = np.array(list(tau[i] for i in nonzero_indices))

    # calculate the slope of the log plot -> the Hurst Exponent
    reg = np.polyfit(np.log(lags), np.log(tau), 1)

    return reg[0]

# Load data
spot_px = pd.read_excel('/Users/vascoboudewijn/Documents/data analytics research/wheat spot px.xlsx')
new_df = pd.DataFrame(spot_px.iloc[:, 1].values, index=spot_px.iloc[:, 0], columns=['Last Px'])
cleaned_df_pd = new_df[pd.notnull(new_df['Last Px'])]

# Define lags
lags = [20, 100, 300, 500, 1000, 2000 ,4000, 5800]

# Calculate mean for each lag and Hurst exponent
results = []
for lag in lags:
    # Filter out NaN values
    filtered_data = cleaned_df_pd['Last Px'].dropna()
    # Check if enough data points are available for the lag
    if len(filtered_data) >= lag:
        # Calculate mean
        mean = filtered_data.tail(lag).mean()
        # Calculate Hurst exponent
        hurst_exp = get_hurst_exponent(filtered_data.values, lag)
        # Append results
        results.append({'Lag': lag, 'Mean': mean, 'Hurst Exponent': hurst_exp})

