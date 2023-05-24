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

# Subset transactions data to only include 'date', 'Transaction Type', 'Symbol', 'Quantity', 'Price', 'Fees', 'Amount'
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

# Calculate cumulative quantity and dividends
symbol_transactions.loc[symbol_transactions['type'] == 'buy', 'cumulative_quantity'] = symbol_transactions.loc[symbol_transactions['type'] == 'buy'].groupby('symbol')['quantity'].cumsum()
symbol_transactions.loc[symbol_transactions['type'] == 'sell', 'cumulative_quantity'] = -symbol_transactions.loc[symbol_transactions['type'] == 'sell'].groupby('symbol')['quantity'].cumsum()
symbol_transactions['cumulative_quantity'] = symbol_transactions['cumulative_quantity'].fillna(0)
symbol_transactions['cumulative_quantity'] = symbol_transactions.groupby('symbol')['cumulative_quantity'].ffill()
symbol_transactions['cumulative_quantity'] = symbol_transactions.groupby('symbol')['cumulative_quantity'].bfill()

symbol_transactions.loc[symbol_transactions['type'] == 'dividend', 'dividends'] = symbol_transactions.loc[symbol_transactions['type'] == 'dividend', 'net_amount'].groupby(symbol_transactions['symbol']).cumsum()
symbol_transactions['dividends'] = symbol_transactions.groupby('symbol')['dividends'].ffill().fillna(0)

# Handle non-symbol transactions separately
non_symbol_transactions['cumulative_total'] = non_symbol_transactions.groupby('type')['net_amount'].cumsum()

# Merge symbol and non-symbol transactions back together
transactions = pd.concat([symbol_transactions, non_symbol_transactions], ignore_index=True)

# Placeholders for price data and returns calculation
transactions['buy_price'] = np.nan
transactions['sell_price'] = np.nan
transactions['returns'] = np.nan
transactions['total_return'] = np.nan

# Calculate buy price, sell price, realized returns
transactions.loc[transactions['type'] == 'buy', 'buy_price'] = transactions.loc[transactions['type'] == 'buy', 'price']

transactions.loc[transactions['type'] == 'sell', 'sell_price'] = transactions.loc[transactions['type'] == 'sell', 'price']

transactions['realized_returns_exc_div'] = transactions['sell_price'] / transactions['buy_price'] - 1
transactions['realized_returns'] = (transactions['sell_price'] + transactions['dividends'])/ transactions['buy_price'] - 1

print(transactions)


