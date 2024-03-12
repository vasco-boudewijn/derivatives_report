import pandas as pd
import numpy as np
import yfinance as yf
import statsmodels.api as sm

spot_px = pd.read_excel('/Users/vascoboudewijn/Documents/data analytics research/wheat spot px.xlsx')
new_df = pd.DataFrame(spot_px.iloc[:, 1].values, index=spot_px.iloc[:, 0], columns=['Last Px'])
cleaned_df_pd = new_df[pd.notnull(new_df['Last Px'])]

df = pd.read_excel('/Users/vascoboudewijn/Documents/data analytics research/uk export data.xlsx', index_col=None)
melted_df = pd.melt(df, id_vars=['Country of Destination'], var_name='Date', value_name='Export Quantity')

df_ordered = melted_df.sort_values(by='Date')
export_to_eu = df_ordered[df_ordered['Country of Destination'] == 'EU']


def get_exchange_rates(base_currency, target_currency, start_date, end_date):
    exchange_rates = yf.download(f"{base_currency}{target_currency}=X", start=start_date, end=end_date)
    return exchange_rates['Close'].to_frame()

start_date_gdp = '2000-01-03'
end_date_gdp = pd.Timestamp.now().strftime('%Y-%m-%d')
gdp_in_euro = get_exchange_rates('GBP', 'EUR', start_date_gdp, end_date_gdp)
gdp_euro = gdp_in_euro['Close'].resample('MS').mean()

# Convert gdp_dzd Series to DataFrame
gdp_euro_df = gdp_euro.to_frame()

# Merge the two DataFrames on the common dates
merged_df = pd.merge(gdp_euro_df, export_to_eu, how='inner', left_index=True, right_on='Date')

combined_df = pd.merge(gdp_in_euro, cleaned_df_pd, how= 'inner', left_index=True, right_index=True )

# Prepare data for regression
X = sm.add_constant(merged_df[gdp_euro_df.columns[0]])  # Independent variable (GDP)
y = merged_df['Export Quantity']  # Dependent variable (export value to Algeria)

# Fit regression model
model = sm.OLS(y, X).fit()

common_index = combined_df.index.intersection(gdp_in_euro.index)

# Filter the gdp_in_euro DataFrame to keep only the rows with common index values
gdp_in_euro_filtered = gdp_in_euro.loc[common_index]

y_log = np.log(combined_df.loc[common_index, 'Last Px'])
X_log = np.log(gdp_in_euro_filtered)

X_log.rename(columns={X_log.columns[0]: 'Exchange rate'}, inplace=True)

y_log_df = y_log.to_frame()
y_log_df.rename(columns={y_log_df.columns[0]: 'Spot Px of Wheat'}, inplace=True)

X_log_mean = X_log.mean()

# Subtract the mean from the independent variable to mean-center it
X_log_centered = X_log - X_log_mean

# Perform the log-log regression with mean-centered independent variable
model_log_log_centered = sm.OLS(y_log, sm.add_constant(X_log_centered)).fit()

# Perform the log-log regression
model_log_log = sm.OLS(y_log, X_log).fit()
print(model_log_log_centered.summary())

with open('/Users/vascoboudewijn/Documents/data analytics research/regression_summary.txt', 'w') as f:
    f.write(model_log_log_centered.summary().as_text())
