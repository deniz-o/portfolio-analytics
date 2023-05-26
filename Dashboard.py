import os
import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

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
transactions.rename(columns={'security_id':'symbol', 'commission/fees':'fees'}, inplace=True)

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
transactions['type'] = transactions.apply(categorize_transactions, axis=1)

# Separate transactions into symbol transactions and non-symbol transactions
symbol_transactions = transactions[transactions['symbol'].notna()].copy()
non_symbol_transactions = transactions[transactions['symbol'].isna()].copy()

# Convert 'net_amount' to numeric
symbol_transactions['net_amount'] = pd.to_numeric(symbol_transactions['net_amount'], errors='coerce')
non_symbol_transactions['net_amount'] = pd.to_numeric(non_symbol_transactions['net_amount'], errors='coerce')

# Set date range and create a dataframe with daily data for each symbol
date_range = pd.date_range(start=symbol_transactions.date.min(), end=symbol_transactions.date.max())
symbols = symbol_transactions.symbol.unique()

idx = pd.MultiIndex.from_product([symbols, date_range], names=['symbol', 'date'])
daily_data = pd.DataFrame(index=idx).reset_index()

# Merge daily data with symbol transactions
symbol_transactions = pd.merge(daily_data, symbol_transactions, on=['symbol', 'date'], how='left')
symbol_transactions['symbol'] = symbol_transactions['symbol'].ffill()

# Create masks for transaction types
buy_mask = symbol_transactions['type'] == 'buy'
sell_mask = symbol_transactions['type'] == 'sell'
dividend_mask = symbol_transactions['type'] == 'dividend'
deposit_mask = symbol_transactions['type'] == 'deposit'
withdrawal_mask = symbol_transactions['type'] == 'withdrawal'

# Make 'quantity' negative for 'sell' transactions
symbol_transactions.loc[sell_mask, 'quantity'] = -symbol_transactions.loc[sell_mask, 'quantity']

# Calculate cumulative quantity for each symbol, forward fill and fill NaN with 0
symbol_transactions['cumulative_quantity'] = symbol_transactions.groupby('symbol')['quantity'].cumsum().ffill().fillna(0)

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
symbol_transactions.loc[buy_mask, 'total_cost'] = symbol_transactions.loc[buy_mask, 'price'] * symbol_transactions.loc[buy_mask, 'quantity'] + symbol_transactions.loc[buy_mask, 'fees']

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
symbol_transactions.loc[sell_mask, 'sales_proceeds'] = (symbol_transactions.loc[sell_mask, 'price'] * - symbol_transactions.loc[sell_mask, 'quantity']) - symbol_transactions.loc[sell_mask, 'fees']

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

# Create a summary DataFrame
summary = transactions.groupby('symbol').last()

# Show only the columns we're interested in
summary = summary[['cumulative_quantity', 'realized_returns', 'dividends']]

# Display the summary DataFrame in Streamlit
st.dataframe(summary)