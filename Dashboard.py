import os
import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import yfinance as yf
from datetime import datetime, date, timedelta
import pandas_datareader as web

# Set page layout to wide
st.set_page_config(layout='wide')

# Get the absolute path to the directory where the script is located
script_dir = os.path.dirname(os.path.abspath(__file__))

# Change the working directory to the script directory
os.chdir(script_dir)

# Load raw transactions data
transactions_raw = pd.read_csv('transactions.csv', skiprows=2, skip_blank_lines=True)

# Drop empty rows
transactions_raw = transactions_raw.dropna(how='all')

# Set settlement date as date in datetime format
transactions_raw['date'] = pd.to_datetime(transactions_raw['Settlement Date'])

# Replace ' ' with '_' and make all columns lowercase
transactions_raw.columns= transactions_raw.columns.str.replace(' ','_').str.lower()

# Subset transactions data to only include 'date', 'description', 'net_amount, 'security_id', 'quantity', 'price', 'commission/fees'
transactions = transactions_raw[['date', 'description', 'net_amount', 'security_id', 'quantity', 'price', 'commission/fees']]

# Rename columns
transactions = transactions.rename(columns={'security_id':'symbol', 'commission/fees':'fees'})

# Define function to categorize transactions
def categorize_transactions(row):
    if 'buy' in row['description'].lower():
        return 'buy'
    elif 'sell' in row['description'].lower():
        return 'sell'
    elif 'dividend' in row['description'].lower():
        return 'dividend'
    elif 'deposit' in row['description'].lower():
        return 'deposit'
    elif 'withdrawal' in row['description'].lower():
        return 'withdrawal'
    elif 'fee' in row['description'].lower():
        return 'fee'
    else:
        return 'other'

# Apply function to categorize transactions
transactions.loc[:, 'type'] = transactions.apply(categorize_transactions, axis=1)

# Separate transactions into symbol transactions and non-symbol transactions
symbol_transactions = transactions[transactions['symbol'].notna()].copy()
non_symbol_transactions = transactions[transactions['symbol'].isna()].copy()

# Convert 'net_amount' to numeric
symbol_transactions['net_amount'] = pd.to_numeric(symbol_transactions['net_amount'].str.replace(',', ''), errors='coerce')
non_symbol_transactions['net_amount'] = pd.to_numeric(non_symbol_transactions['net_amount'].str.replace(',', ''), errors='coerce')

# Set date range and create a dataframe with daily data for each symbol
start_date = transactions.date.min()
end_date = datetime.now().date()
date_range = pd.date_range(start=start_date, end=end_date)
symbols = symbol_transactions.symbol.unique()

idx = pd.MultiIndex.from_product([symbols, date_range], names=['symbol', 'date'])
daily_data = pd.DataFrame(index=idx).reset_index()

# Merge daily data with symbol transactions
symbol_transactions = pd.merge(daily_data, symbol_transactions, on=['symbol', 'date'], how='left')
symbol_transactions['symbol'] = symbol_transactions['symbol'].ffill()

# Create masks for symbol transactions
buy_mask = symbol_transactions['type'] == 'buy'
sell_mask = symbol_transactions['type'] == 'sell'
dividend_mask = symbol_transactions['type'] == 'dividend'

# Make quantity negative for sell transactions
symbol_transactions.loc[sell_mask, 'quantity'] = -symbol_transactions.loc[sell_mask, 'quantity']

# Ensure transactions are sorted by date
transactions = transactions.sort_values('date')

# Fill NaN with 0 for quantity
symbol_transactions['quantity'] = symbol_transactions['quantity'].fillna(0)

# Calculate cumulative quantity for each symbol, forward fill and fill NaN with 0
symbol_transactions['cumulative_quantity'] = symbol_transactions.groupby('symbol')['quantity'].cumsum().fillna(0)

# Calculate dividends received for each symbol
symbol_transactions.loc[dividend_mask, 'dividends'] = symbol_transactions.loc[dividend_mask, 'net_amount'].groupby(symbol_transactions['symbol']).cumsum()

# Forward fill dividends by symbol and fill NaN with 0
symbol_transactions['dividends'] = symbol_transactions.groupby('symbol')['dividends'].ffill().fillna(0)

# Calculate average buy price (excluding fees) per share for each symbol
symbol_transactions.loc[buy_mask, 'avg_buy_price'] = (symbol_transactions.loc[buy_mask, 'quantity'] * symbol_transactions.loc[buy_mask, 'price']).groupby(symbol_transactions['symbol']).cumsum() / symbol_transactions.loc[buy_mask].groupby('symbol')['quantity'].cumsum()

# Forward fill average buy price (excluding fees) per share by symbol
symbol_transactions['avg_buy_price'] = symbol_transactions.groupby('symbol')['avg_buy_price'].ffill()

# Calculate average sell price (excluding fees) for each symbol
symbol_transactions.loc[sell_mask, 'avg_sell_price'] = (symbol_transactions.loc[sell_mask, 'price'] * - symbol_transactions.loc[sell_mask, 'quantity']) / symbol_transactions.loc[sell_mask, 'quantity'].abs()

# Forward fill average sell price (excluding fees) by symbol
symbol_transactions['avg_sell_price'] = symbol_transactions.groupby('symbol')['avg_sell_price'].ffill()

# Calculate total cost for each buy transaction
symbol_transactions.loc[buy_mask, 'total_cost'] = symbol_transactions.loc[buy_mask, 'price'] * symbol_transactions.loc[buy_mask, 'quantity'] - symbol_transactions.loc[buy_mask, 'fees']

# Calculate cumulative total cost for each symbol, forward fill and fill NaN with 0
symbol_transactions['cumulative_total_cost'] = symbol_transactions.groupby('symbol')['total_cost'].cumsum().ffill().fillna(0)

# Calculate average cost basis per share for each symbol
symbol_transactions['avg_cost_basis_per_share'] = symbol_transactions['cumulative_total_cost'] / symbol_transactions['cumulative_quantity']

# Adjust average cost basis per share for sell transactions to use the cumulative quantity and cumulative total cost from the previous day
prev_day_cumulative_quantity = symbol_transactions.groupby('symbol')['cumulative_quantity'].shift(1)
prev_day_cumulative_total_cost = symbol_transactions.groupby('symbol')['cumulative_total_cost'].shift(1)
sell_mask_prev_day_quantity = sell_mask & (prev_day_cumulative_quantity > 0)

symbol_transactions.loc[sell_mask_prev_day_quantity, 'avg_cost_basis_per_share'] = prev_day_cumulative_total_cost / prev_day_cumulative_quantity

# Forward fill average cost basis per share by symbol
symbol_transactions['avg_cost_basis_per_share'] = symbol_transactions.groupby('symbol')['avg_cost_basis_per_share'].ffill()

# Calculate sales proceeds net of fees for sell transactions
symbol_transactions.loc[sell_mask, 'sales_proceeds'] = (symbol_transactions.loc[sell_mask, 'price'] * - symbol_transactions.loc[sell_mask, 'quantity']) + symbol_transactions.loc[sell_mask, 'fees']

# For each sell transaction, calculate cost basis of sold shares based on average cost basis per share at the time of sale
symbol_transactions.loc[sell_mask, 'sell_cost_basis'] = symbol_transactions.loc[sell_mask, 'avg_cost_basis_per_share'] * -symbol_transactions.loc[sell_mask, 'quantity']

# Subtract cost basis of sold shares from cumulative total cost for each symbol
symbol_transactions.loc[sell_mask, 'cumulative_total_cost'] -= symbol_transactions.loc[sell_mask, 'sell_cost_basis']

# Initialize realized gains and realized returns to NaN
symbol_transactions['realized_gains'] = np.nan
symbol_transactions['realized_returns'] = np.nan

# Calculate realized gains in dollars and realized returns (net of fees, including dividends if sold) 
# only when both sell cost basis and sales proceeds are not null
sell_mask_realized_return = (sell_mask) & (symbol_transactions['sell_cost_basis'].notnull()) & (symbol_transactions['sales_proceeds'].notnull())
symbol_transactions.loc[sell_mask_realized_return, 'realized_gains'] = symbol_transactions.loc[sell_mask_realized_return, 'sales_proceeds'] + symbol_transactions.loc[sell_mask_realized_return, 'dividends']
symbol_transactions.loc[sell_mask_realized_return, 'realized_returns'] = symbol_transactions.loc[sell_mask_realized_return, 'realized_gains'] / symbol_transactions.loc[sell_mask_realized_return, 'sell_cost_basis'] - 1

# Forward fill realized gains and realized returns by symbol
symbol_transactions['realized_gains'] = symbol_transactions.groupby('symbol')['realized_gains'].ffill()
symbol_transactions['realized_returns'] = symbol_transactions.groupby('symbol')['realized_returns'].ffill()

# Group non-symbol transactions by type and calculate cumulative total
non_symbol_transactions['cumulative_total'] = non_symbol_transactions.groupby('type')['net_amount'].cumsum()

# Merge symbol and non-symbol transactions back together
transactions = pd.concat([symbol_transactions, non_symbol_transactions], ignore_index=True)

# Calculate cash flow for each transaction
def calculate_cash(row):
    if row['type'] == 'buy':
        return -row['total_cost']
    elif row['type'] == 'sell':
        return row['sales_proceeds']
    elif row['type'] == 'dividend':
        return row['net_amount']
    elif row['type'] == 'deposit':
        return row['net_amount']
    elif row['type'] == 'withdrawal':
        return row['net_amount']
    elif row['type'] == 'fee':
        return row['net_amount']
    else:
        return 0

transactions['cash'] = transactions.apply(calculate_cash, axis=1)

# Ensure the transactions are sorted by date
transactions = transactions.sort_values('date')

# Compute cumulative cash value
transactions['cumulative_cash'] = transactions['cash'].cumsum()

# Load daily price data for each symbol in the portfolio
symbols_list = symbols.tolist()
prices = yf.download(symbols_list, start=start_date, end=end_date)['Adj Close']

# Reset the index to create a column for the dates
prices.reset_index(inplace=True)

# Rename the date column to match the transactions dataframe
prices = prices.rename(columns={'Date': 'date'}) # type: ignore

# Melt the prices dataframe
prices = prices.melt(id_vars='date', var_name='symbol', value_name='adj_close')

# Merge the transactions and prices dataframes
transactions = pd.merge(transactions, prices, how='left', on=['symbol', 'date'])

# Identify the latest date with available closing price
last_valid_date = transactions['date'].loc[transactions['adj_close'].notnull()].iloc[-1]

# Remove the time component from dates
transactions['date'] = transactions['date'].dt.date

# Forward fill the adjusted closing prices
transactions['adj_close'] = transactions.groupby('symbol')['adj_close'].ffill()

# Calculate daily values of each holding
transactions['daily_value'] = transactions['cumulative_quantity'] * transactions['adj_close']

# Calculate daily portfolio value including cash
portfolio_value = transactions.groupby('date').apply(lambda x: x['daily_value'].sum() + x['cumulative_cash'].iloc[-1])

# Change column name to 'Portfolio Value'
portfolio_value = portfolio_value.rename('Portfolio Value')

# Convert date index to datetime
portfolio_value.index = pd.to_datetime(portfolio_value.index)

# Calculate daily portfolio returns
portfolio_returns = portfolio_value.pct_change().dropna()

# Change column name to 'Portfolio Return'
portfolio_returns = portfolio_returns.rename('Portfolio Return')

# Create a summary dataframe
summary = transactions.groupby('symbol').last()

# Rename columns
summary = summary.rename(columns={'symbol': 'Symbol', 'cumulative_quantity': 'Quantity', 'adj_close': 'Current Price', 'daily_value': 'Current Value',
                        'realized_returns': 'Realized Return', 'dividends': 'Dividends Received'})

# Sort the holdings by quantity
summary = summary.sort_values('Quantity', ascending=False)

# # Add the latest cash value to the summary dataframe
# summary.loc['Cash', 'Current Value'] = transactions['cumulative_cash'].iloc[-1]

# Create a dataframe for cash
cash_df = pd.DataFrame(index=['Cash'], columns=['Current Value'])
cash_df.loc['Cash', 'Current Value'] = transactions['cumulative_cash'].iloc[-1]

# Concatenate cash_df with the rest of the summary dataframe
summary = pd.concat([cash_df, summary])

# Show only the columns we're interested in
summary = summary[['Quantity', 'Current Price', 'Current Value', 'Realized Return', 'Dividends Received']]

# Convert realized returns to percentages and round to two decimal places
summary['Realized Return'] = summary['Realized Return'].apply(lambda x: '{:.2%}'.format(x) if pd.notnull(x) else '-')

# Round quantity to integer
summary['Quantity'] = summary['Quantity'].apply(lambda x: '{:,.0f}'.format(x) if pd.notnull(x) else '-')

# Round the other summary values to two decimal places and add thousands separators
summary[['Current Price', 'Current Value', 'Dividends Received']] = summary[['Current Price', 'Current Value', 'Dividends Received']].applymap(
    lambda x: '{:,.2f}'.format(x) if pd.notnull(x) else '-')

# Calculate total return using the first and latest portfolio value (does not work if more cash has been deposited)
total_return = portfolio_value.iloc[-1] / portfolio_value.iloc[0] - 1

# Load daily price data for S&P 500 to use as a benchmark
benchmark_daily = yf.download('^GSPC', start=start_date, end=end_date)['Adj Close']

# Change column name to 'S&P 500'
benchmark_daily = benchmark_daily.rename('S&P 500')

# Calculate daily benchmark returns
benchmark_returns_daily = benchmark_daily.pct_change().dropna()

# Merge with portfolio returns to create a dataframe with both
portfolio_benchmark_returns_daily = pd.merge(portfolio_returns, benchmark_returns_daily, how='left', left_index=True, right_index=True)

# Fill NaN with zero for benchmark returns
portfolio_benchmark_returns_daily['S&P 500'] = portfolio_benchmark_returns_daily['S&P 500'].fillna(0)

# Load daily 3-month Treasury Bill data to use as the risk-free rate
rf = web.DataReader('TB3MS', 'fred',start=start_date, end=end_date)

# Convert to monthly returns
rf = (1 + (rf / 100)) ** (1 / 12) - 1 
rf = rf.resample('M').last()
# st.write(rf)

# Calculate monthly benchmark returns
benchmark_returns_monthly = benchmark_daily.resample('M').last().pct_change().dropna()
# st.write(benchmark_returns_monthly)

# Calculate monthly portfolio returns
portfolio_returns_monthly = portfolio_value.resample('M').last().pct_change().dropna()

# Change column name to 'Portfolio Return'
portfolio_returns_monthly = portfolio_returns_monthly.rename('Portfolio Return')
# st.write(portfolio_returns_monthly)

# Merge monthly risk-free rate, portfolio returns, benchmark returns
monthly_returns = rf.join(portfolio_returns_monthly, how='left').join(benchmark_returns_monthly, how='left')

# Calculate excess returns over risk-free rate
monthly_returns['excess_return'] = monthly_returns['Portfolio Return'] - monthly_returns['TB3MS']
st.write(monthly_returns)

# Calculate standard deviation of excess returns
std_monthly_excess_return = monthly_returns['excess_return'].std()

# Calculate realized Sharpe ratio for the period
sharpe_ratio_monthly = monthly_returns['excess_return'].mean() / std_monthly_excess_return

# Annualize Sharpe ratio
sharpe_ratio = sharpe_ratio_monthly * np.sqrt(12)

# Create streamlit display elements
st.title('Portfolio Dashboard')

st.caption(f'Data provided as of market close on {last_valid_date: %m-%d-%Y}')

col1, col2 = st.columns(2)

with col1:
    st.subheader('Summary')
    st.dataframe(summary)

with col2:
    metriccol1, metriccol2, metriccol3 = st.columns(3)
    with metriccol1:
        st.metric(label=f'Portfolio Value', value='${:,.0f}'.format(portfolio_value.iloc[-1]), delta='${:,.0f}'.format(portfolio_value.iloc[-1] - portfolio_value.iloc[0]))
    with metriccol2:
        st.metric(label=f'Total Return', value='{:.2%}'.format(total_return))
    with metriccol3:
        st.metric(label=f'Sharpe Ratio', value='{:.2f}'.format(sharpe_ratio))
    st.area_chart(data = portfolio_value)

# st.dataframe(portfolio_value)

# st.dataframe(portfolio_returns)