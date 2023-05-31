import os
import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import yfinance as yf
from datetime import datetime, date, timedelta

# Get the absolute path to the directory where the script is located
script_dir = os.path.dirname(os.path.abspath(__file__))

# Change the working directory to the script directory
os.chdir(script_dir)

# Load raw transactions data
transactions_raw = pd.read_csv('transactions.csv', skiprows=2, skip_blank_lines=True)

# Drop empty rows
transactions_raw = transactions_raw.dropna(how='all')

# Set settlement date as 'date' in datetime format
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

# Make 'quantity' negative for 'sell' transactions
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

# Adjust 'avg_cost_basis_per_share' for sell transactions to use the 'cumulative_quantity' and 'cumulative_total_cost' from the previous day
prev_day_cumulative_quantity = symbol_transactions.groupby('symbol')['cumulative_quantity'].shift(1)
prev_day_cumulative_total_cost = symbol_transactions.groupby('symbol')['cumulative_total_cost'].shift(1)
sell_mask_prev_day_quantity = sell_mask & (prev_day_cumulative_quantity > 0)

symbol_transactions.loc[sell_mask_prev_day_quantity, 'avg_cost_basis_per_share'] = prev_day_cumulative_total_cost / prev_day_cumulative_quantity

# Forward fill avg_cost_basis_per_share by symbol
symbol_transactions['avg_cost_basis_per_share'] = symbol_transactions.groupby('symbol')['avg_cost_basis_per_share'].ffill()

# Calculate sales proceeds net of fees for sell transactions
symbol_transactions.loc[sell_mask, 'sales_proceeds'] = (symbol_transactions.loc[sell_mask, 'price'] * - symbol_transactions.loc[sell_mask, 'quantity']) + symbol_transactions.loc[sell_mask, 'fees']

# For each sell transaction, calculate cost basis of sold shares based on average cost basis per share at the time of sale
symbol_transactions.loc[sell_mask, 'sell_cost_basis'] = symbol_transactions.loc[sell_mask, 'avg_cost_basis_per_share'] * -symbol_transactions.loc[sell_mask, 'quantity']

# Subtract cost basis of sold shares from cumulative total cost for each symbol
symbol_transactions.loc[sell_mask, 'cumulative_total_cost'] -= symbol_transactions.loc[sell_mask, 'sell_cost_basis']

# Initialize realized returns to NaN
symbol_transactions['realized_returns'] = np.nan

# Calculate realized returns (net of fees, including dividends if sold) only when both 'sell_cost_basis' and 'sales_proceeds' are not null
sell_mask_realized_return = (sell_mask) & (symbol_transactions['sell_cost_basis'].notnull()) & (symbol_transactions['sales_proceeds'].notnull())
symbol_transactions.loc[sell_mask_realized_return, 'realized_returns'] = (symbol_transactions.loc[sell_mask_realized_return, 'sales_proceeds'] + symbol_transactions.loc[sell_mask_realized_return, 'dividends']) / symbol_transactions.loc[sell_mask_realized_return, 'sell_cost_basis'] - 1

# Forward fill realized returns by symbol
symbol_transactions['realized_returns'] = symbol_transactions.groupby('symbol')['realized_returns'].ffill()

# Calculate dollar amount realized returns
symbol_transactions['dollar_realized_returns'] = symbol_transactions['realized_returns'] * symbol_transactions['total_cost']

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

# Rename the 'Date' column to 'date' to match the transactions DataFrame
prices.rename(columns={'Date': 'date'}, inplace=True)

# Melt the prices DataFrame
prices = prices.melt(id_vars='date', var_name='symbol', value_name='adj_close')

# Merge the transactions and prices dataframes
transactions = pd.merge(transactions, prices, how='left', on=['symbol', 'date'])

# Forward fill the adjusted closing prices
transactions['adj_close'] = transactions.groupby('symbol')['adj_close'].ffill()

# Calculate daily values of each holding
transactions['daily_value'] = transactions['cumulative_quantity'] * transactions['adj_close']

# # Calculate daily portfolio value including cash
# portfolio_value = transactions.groupby('date')['daily_value'].sum() + transactions.set_index('date')['cumulative_cash']

# Calculate daily portfolio value including cash
portfolio_value = transactions.groupby('date').apply(lambda x: x['daily_value'].sum() + x['cumulative_cash'].iloc[-1])

# Calculate daily portfolio returns
portfolio_returns = portfolio_value.pct_change().dropna()

# Change column name to 'Portfolio Value ($)'
portfolio_value = portfolio_value.rename('Portfolio Value ($)')

# Create a summary DataFrame
summary = transactions.groupby('symbol').last()

# Rename columns
summary = summary.rename(columns={'symbol': 'Symbol', 'cumulative_quantity': 'Quantity', 'adj_close': 'Current Price', 'daily_value': 'Current Value',
                        'realized_returns': 'Realized Return', 'dividends': 'Dividends Received'})

# Add the latest cash value to the summary DataFrame
summary.loc['Cash', 'Current Value'] = transactions['cumulative_cash'].iloc[-1]

# Show only the columns we're interested in
summary = summary[['Quantity', 'Current Price', 'Current Value', 'Realized Return', 'Dividends Received']]

summary = summary.round(2)

# Create streamlit display elements
st.title('Portfolio Dashboard')
st.metric(label='Portfolio Value', value=portfolio_value.iloc[-1].round(2))
st.area_chart(data = portfolio_value)

st.dataframe(summary)