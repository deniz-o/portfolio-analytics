import streamlit as st
import yfinance as yf

import os
import time
from datetime import datetime, date, timedelta
import pandas as pd
import pandas_datareader as web
import numpy as np
import plotly.graph_objects as go
import matplotlib.pyplot as plt

import transactions as tx

# Set page layout to wide
st.set_page_config(layout='wide')

# Hide streamlit elements
hide_streamlit_style = """
        <style>
        #MainMenu {visibility: hidden;}
        footer {visibility: hidden;}
        </style>
        """

st.markdown(hide_streamlit_style, unsafe_allow_html=True)

# # Display progress bar
# progress_text = "Processing transactions..."
# my_bar = st.progress(0, text=progress_text)

# for percent_complete in range(100):
#     time.sleep(0.2)
#     my_bar.progress(percent_complete + 1, text=progress_text)

# my_bar.empty()

# Get the absolute path to the directory where the script is located
script_dir = os.path.dirname(os.path.abspath(__file__))

# Change the working directory to the script directory
os.chdir(script_dir)

# Load and preprocess transactions data
transactions = tx.load_and_preprocess_transactions('transactions.csv')

# Apply function to categorize transactions
transactions.loc[:, 'type'] = transactions.apply(tx.categorize_transactions, axis=1)

# Separate transactions into symbol transactions and non-symbol transactions
symbol_transactions = transactions[transactions['symbol'].notna()].copy()
non_symbol_transactions = transactions[transactions['symbol'].isna()].copy()

# Create the list of symbols
symbols = symbol_transactions.symbol.unique().tolist()

# Set date range
start_date = transactions.date.min()
end_date = datetime.now().date()

# Process symbol transactions
symbol_transactions = tx.process_symbol_transactions(symbol_transactions, start_date, end_date)

# Group non-symbol transactions by type and calculate cumulative total
non_symbol_transactions['cumulative_total'] = non_symbol_transactions.groupby('type')['net_amount'].cumsum()

# Merge symbol and non-symbol transactions back together
transactions = pd.concat([symbol_transactions, non_symbol_transactions], ignore_index=True)

# Calculate cash for each transaction
transactions['cash'] = transactions.apply(tx.calculate_cash, axis=1)

# Ensure the transactions are sorted by date
transactions = transactions.sort_values('date')

# Compute cumulative cash value
transactions['cumulative_cash'] = transactions['cash'].cumsum()

# Load daily price data for each symbol
prices = tx.load_daily_prices(symbols, start_date, end_date)

# Merge the transactions and prices dataframes
transactions = pd.merge(transactions, prices, how='left', on=['symbol', 'date'])

# Identify the latest date with available closing price
last_valid_date = transactions['date'].loc[transactions['adj_close'].notnull()].iloc[-1]

# Forward fill the adjusted closing prices
transactions['adj_close'] = transactions.groupby('symbol')['adj_close'].ffill()

# For each holding, calculate unrealized and total gains/losses, unrealized and total returns
transactions = tx.calculate_holding_metrics(transactions)

# Calculate daily portfolio value (including cash) and portfolio returns
portfolio_value, portfolio_returns = tx.calculate_daily_portfolio_returns(transactions)

# Create a summary dataframe
summary = tx.create_summary_table(transactions)

# Show only the columns we're interested in
summary = summary[
    ['Quantity', 'Current Price', 'Current Value', 'Allocation', 'Realized Gain/Loss', 'Realized Return', 'Unrealized Gain/Loss',
     'Unrealized Return', 'Dividends Collected', 'Total Gain/Loss']
     ]

# Set '-' for unrealized returns of securities that are not currently held
summary.loc[summary['Quantity'] == 0, 'Unrealized Return'] = '-'

# Format the values for 'Allocation', 'Realized Return', and 'Unrealized Return'
def format_percentage(x):
    if pd.notnull(x) and isinstance(x, (int, float)):
        return '{:.2%}'.format(x)
    else:
        x='-'
    return x

# Convert allocation and realized returns to percentages and round to two decimal places
summary[['Allocation', 'Realized Return', 'Unrealized Return']] = summary[['Allocation', 'Realized Return', 'Unrealized Return']].applymap(format_percentage)

# Round quantity to integer
summary['Quantity'] = summary['Quantity'].apply(lambda x: '{:,.0f}'.format(x) if pd.notnull(x) else '-')

# Round the other summary values to two decimal places and add thousands separators
summary[['Current Price', 'Current Value', 'Realized Gain/Loss', 'Unrealized Gain/Loss', 'Dividends Collected', 'Total Gain/Loss']] = summary[
    ['Current Price', 'Current Value', 'Realized Gain/Loss', 'Unrealized Gain/Loss', 'Dividends Collected', 'Total Gain/Loss']].applymap(
    lambda x: '{:,.2f}'.format(x) if pd.notnull(x) else '-')

# Calculate total return using the first and latest portfolio value (not meaningful if more cash has been deposited)
total_return = portfolio_value.iloc[-1] / portfolio_value.iloc[0] - 1

# Load daily price data for S&P 500 (^GSPC) to use as a benchmark
benchmark_daily = tx.load_benchmark_data(start_date, end_date)

# Merge daily portfolio value with benchmark
portfolio_benchmark_daily = pd.merge(portfolio_value, benchmark_daily, how='left', left_index=True, right_index=True)

# Forward fill the benchmark daily values
portfolio_benchmark_daily['S&P 500'] = portfolio_benchmark_daily['S&P 500'].fillna(method='ffill')

# Calculate daily benchmark returns
benchmark_returns_daily = portfolio_benchmark_daily['S&P 500'].pct_change().dropna()

# Merge with portfolio returns to create a dataframe with both
portfolio_benchmark_returns_daily = pd.merge(portfolio_returns, benchmark_returns_daily, how='left', left_index=True, right_index=True)
portfolio_benchmark_returns_daily.columns = ['Portfolio Return', 'S&P 500 Return']

# Calculate cumulative returns
portfolio_cumulative_returns = (1 + portfolio_benchmark_returns_daily['Portfolio Return']).cumprod() - 1
benchmark_cumulative_returns = (1 + portfolio_benchmark_returns_daily['S&P 500 Return']).cumprod() - 1

# Normalize cumulative returns to start at 100
portfolio_cumulative_returns_normalized = 100 * (1 + portfolio_cumulative_returns)
benchmark_cumulative_returns_normalized = 100 * (1 + benchmark_cumulative_returns)

# Load daily 3-month Treasury Bill data to use as the risk-free rate
rf = web.DataReader('TB3MS', 'fred',start=start_date, end=end_date)

# Convert to monthly returns
rf = (1 + (rf / 100)) ** (1 / 12) - 1 
rf = rf.resample('M').last()

# Calculate monthly benchmark returns
benchmark_returns_monthly = benchmark_daily.resample('M').last().pct_change().dropna()

# Change column name to 'S&P500 Return'
benchmark_returns_monthly = benchmark_returns_monthly.rename('S&P 500 Return')

# Calculate monthly portfolio returns
portfolio_returns_monthly = portfolio_value.resample('M').last().pct_change().dropna()

# Change column name to 'Portfolio Return'
portfolio_returns_monthly = portfolio_returns_monthly.rename('Portfolio Return')

# Merge monthly risk-free rate, portfolio returns, benchmark returns
monthly_returns = rf.join(portfolio_returns_monthly, how='left').join(benchmark_returns_monthly, how='left')

# Calculate excess returns over risk-free rate
monthly_returns['excess_return'] = monthly_returns['Portfolio Return'] - monthly_returns['TB3MS']

# Calculate standard deviation of monthly excess returns
std_monthly_excess_return = monthly_returns['excess_return'].std()

# Calculate realized Sharpe ratio for the period
sharpe_ratio_monthly = monthly_returns['excess_return'].mean() / std_monthly_excess_return

# Annualize Sharpe ratio
sharpe_ratio = sharpe_ratio_monthly * np.sqrt(12)

# Create streamlit display elements
st.title('Portfolio Dashboard')

st.caption(f'Data provided as of market close on {last_valid_date: %m-%d-%Y}')

metriccol1, metriccol2, metriccol3 = st.columns(3)
with metriccol1:
    st.metric(
        label=f'Portfolio Value', 
        value='${:,.0f}'.format(portfolio_value.iloc[-1]), 
        delta='${:,.0f}'.format(portfolio_value.iloc[-1] - portfolio_value.iloc[0])
        )
with metriccol2:
    st.metric(
        label=f'Total Return',
        value='{:.2%}'.format(total_return)
        )
with metriccol3:
    st.metric(
        label=f'Sharpe Ratio', 
        value='{:.2f}'.format(sharpe_ratio)
        )

st.area_chart(data = portfolio_value)

st.subheader('Summary')
st.dataframe(summary)

# Create chart for portfolio and benchmark cumulative returns
st.subheader('Normalized Portfolio Cumulative Returns vs S&P 500')
fig = go.Figure()
fig.add_trace(go.Scatter(x=portfolio_cumulative_returns_normalized.index, y=portfolio_cumulative_returns_normalized,
                    mode='lines', name='Portfolio'))
fig.add_trace(go.Scatter(x=benchmark_cumulative_returns_normalized.index, y=benchmark_cumulative_returns_normalized,
                    mode='lines', name='S&P 500'))

# Update y-axis to start from 100
fig.update_yaxes(range=[80, max(max(portfolio_cumulative_returns_normalized), max(benchmark_cumulative_returns_normalized)+ 10)])

st.plotly_chart(fig, use_container_width=True)

# Create chart for portfolio and benchmark monthly returns
st.subheader('Portfolio and S&P 500 Monthly Returns')
fig = go.Figure()
fig.add_trace(go.Bar(x=monthly_returns.index, y=monthly_returns['Portfolio Return'],
                    name='Portfolio'))
fig.add_trace(go.Bar(x=monthly_returns.index, y=monthly_returns['S&P 500 Return'],
                    name='S&P 500'))

fig.update_layout(barmode='group')

st.plotly_chart(fig, use_container_width=True)