import pandas as pd
import numpy as np
import yfinance as yf

def load_and_preprocess_transactions(file_path):
    '''
    Loads transactions from a CSV file and preprocess for analysis
    
    Parameters:
        file_path (str): Path to the CSV file
    
    Returns:
        DataFrame: Processed transactions data
    
    Notes:
        Column names in the CSV file should include the following:
        - Date
        - Symbol
        - Description
        - Net Amount
        - Quantity
        - Price
        - Fees
        Case insensitive and spaces are ignored
    '''
    transactions_raw = pd.read_csv(file_path, skiprows=2, skip_blank_lines=True)
    transactions_raw = transactions_raw.dropna(how='all')
    transactions_raw.columns= transactions_raw.columns.str.replace(' ','_').str.lower()
    transactions_raw['date'] = pd.to_datetime(transactions_raw['date'])
    transactions_raw['net_amount'] = pd.to_numeric(
        transactions_raw['net_amount'].str.replace(',', ''), errors='coerce')
    transactions = transactions_raw[['date', 'symbol', 'description', 'net_amount',
                                     'quantity', 'price', 'fees']]
    return transactions

def categorize_transactions(row):
    '''Categorizes transactions based on keywords in the description'''
    transaction_keywords = {
        'buy': ['buy'],
        'sell': ['sell'],
        'dividend': ['dividend'],
        'deposit': ['deposit'],
        'withdrawal': ['withdrawal'],
        'fee': ['fee']
    }
    description = row['description'].lower()
    for transaction_type, keywords in transaction_keywords.items():
        if any(keyword in description for keyword in keywords):
            return transaction_type
    return 'other'

def process_symbol_transactions(symbol_transactions, start_date, end_date):
    """
    Processes symbol transactions to perform calculations such as
    cumulative quantity, cost basis, realized returns, dividends collected

    Parameters:
        symbol_transactions (DataFrame): Symbol transactions data
        start_date (datetime): Start date for the date range
        end_date (datetime): End date for the date range

    Returns:
        DataFrame: Processed symbol transactions data
    """
    # Set date range and create a dataframe with daily data for each symbol
    date_range = pd.date_range(start=start_date, end=end_date)
    symbols = symbol_transactions.symbol.unique()
    idx = pd.MultiIndex.from_product([symbols, date_range], names=['symbol', 'date'])
    daily_data = pd.DataFrame(index=idx).reset_index()
    
    # Merge daily data with symbol transactions 
    symbol_transactions = pd.merge(daily_data, symbol_transactions, on=['symbol', 'date'], how='left')
    symbol_transactions['symbol'] = symbol_transactions['symbol'].ffill()
    
    # Ensure transactions are sorted by date
    symbol_transactions = symbol_transactions.sort_values('date')

    # Create masks for symbol transactions
    buy_mask = symbol_transactions['type'] == 'buy'
    sell_mask = symbol_transactions['type'] == 'sell'
    dividend_mask = symbol_transactions['type'] == 'dividend'

    # Ensure quantity is negative for sell transactions
    symbol_transactions.loc[sell_mask, 'quantity'] = -abs(symbol_transactions.loc[sell_mask, 'quantity'])

    # Fill NaN with 0 for quantity
    symbol_transactions['quantity'] = symbol_transactions['quantity'].fillna(0)

    # Calculate cumulative quantity for each symbol, forward fill and fill NaN with 0
    symbol_transactions['cumulative_quantity'] = symbol_transactions.groupby('symbol')['quantity'].cumsum().fillna(0)

    # Calculate average buy price (excluding fees) per share for each symbol and forward fill
    symbol_transactions.loc[buy_mask, 'avg_buy_price'] = (symbol_transactions.loc[buy_mask, 'quantity'] * symbol_transactions.loc[buy_mask, 'price']).groupby(symbol_transactions['symbol']).cumsum() / symbol_transactions.loc[buy_mask].groupby('symbol')['quantity'].cumsum()
    symbol_transactions['avg_buy_price'] = symbol_transactions.groupby('symbol')['avg_buy_price'].ffill()

    # Calculate average sell price (excluding fees) for each symbol and forward fill
    symbol_transactions.loc[sell_mask, 'avg_sell_price'] = (symbol_transactions.loc[sell_mask, 'price'] * - symbol_transactions.loc[sell_mask, 'quantity']) / symbol_transactions.loc[sell_mask, 'quantity'].abs()
    symbol_transactions['avg_sell_price'] = symbol_transactions.groupby('symbol')['avg_sell_price'].ffill()

    # Calculate total cost for each buy transaction
    symbol_transactions.loc[buy_mask, 'total_cost'] = symbol_transactions.loc[buy_mask, 'price'] * symbol_transactions.loc[buy_mask, 'quantity'] - symbol_transactions.loc[buy_mask, 'fees']

    # Fill NaN with 0 for total cost
    symbol_transactions['total_cost'] = symbol_transactions['total_cost'].fillna(0)

    # Calculate cumulative total cost for each symbol
    symbol_transactions['cumulative_total_cost'] = symbol_transactions.groupby('symbol')['total_cost'].cumsum()

    # Calculate average cost basis per share for each symbol
    symbol_transactions['avg_cost_basis_per_share'] = symbol_transactions['cumulative_total_cost'] / symbol_transactions['cumulative_quantity']

    # Adjust average cost basis per share for sell transactions to use the cumulative quantity and cumulative total cost from the previous day
    prev_day_cumulative_quantity = symbol_transactions.groupby('symbol')['cumulative_quantity'].shift(1)
    prev_day_cumulative_total_cost = symbol_transactions.groupby('symbol')['cumulative_total_cost'].shift(1)
    sell_mask_prev_day_quantity = sell_mask & (prev_day_cumulative_quantity > 0)

    symbol_transactions.loc[sell_mask_prev_day_quantity, 'avg_cost_basis_per_share'] = prev_day_cumulative_total_cost / prev_day_cumulative_quantity

    # Forward fill average cost basis per share by symbol
    symbol_transactions['avg_cost_basis_per_share'] = symbol_transactions.groupby('symbol')['avg_cost_basis_per_share'].ffill()
    
    # For each sell transaction, calculate cost basis of sold shares based on average cost basis per share at the time of sale
    symbol_transactions.loc[sell_mask, 'sell_cost_basis'] = symbol_transactions.loc[sell_mask, 'avg_cost_basis_per_share'] * symbol_transactions.loc[sell_mask, 'quantity'].abs()

    # Subtract cost basis of sold shares from cumulative total cost for each symbol
    symbol_transactions.loc[sell_mask, 'total_cost'] = - symbol_transactions.loc[sell_mask, 'sell_cost_basis']

    # Update cumulative total cost for each symbol after sell transactions
    symbol_transactions['cumulative_total_cost'] = symbol_transactions.groupby('symbol')['total_cost'].cumsum()

    # Calculate sales proceeds net of fees for sell transactions
    symbol_transactions.loc[sell_mask, 'sales_proceeds'] = (symbol_transactions.loc[sell_mask, 'price'] * - symbol_transactions.loc[sell_mask, 'quantity']) + symbol_transactions.loc[sell_mask, 'fees']

    # Calculate dividends collected for each symbol
    symbol_transactions.loc[dividend_mask, 'dividends'] = symbol_transactions.loc[dividend_mask, 'net_amount'].groupby(symbol_transactions['symbol']).cumsum()

    # Forward fill dividends by symbol and fill NaN with 0
    symbol_transactions['dividends'] = symbol_transactions.groupby('symbol')['dividends'].ffill().fillna(0)

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

    return symbol_transactions

def calculate_cash(row):
    cash = 0
    transaction_type = row['type']
    if transaction_type == 'buy':
        cash = -row['total_cost']
    elif transaction_type in ['sell', 'dividend', 'deposit', 'withdrawal', 'fee']:
        cash = row['net_amount']
    return cash

def load_daily_prices(symbols, start_date, end_date):
    # Convert symbols to list if necessary
    symbols_list = symbols.tolist() if isinstance(symbols, pd.Series) else symbols

    # Load daily price data using yfinance
    prices = yf.download(symbols_list, start=start_date, end=end_date)['Adj Close']

    # Reset the index to create a column for the dates
    prices.reset_index(inplace=True)

    # Rename the date column to match the transactions dataframe
    prices = prices.rename(columns={'Date': 'date'})

    # Melt the prices dataframe
    prices = prices.melt(id_vars='date', var_name='symbol', value_name='adj_close')

    return prices

def calculate_holding_metrics(transactions):
    transactions['daily_value'] = transactions['cumulative_quantity'] * transactions['adj_close']
    transactions['unrealized_gains'] = transactions['daily_value'] - transactions['cumulative_total_cost']
    transactions['unrealized_returns'] = transactions['unrealized_gains'] / transactions['cumulative_total_cost']
    transactions['total_gains'] = transactions['realized_gains'] + transactions['unrealized_gains'] + transactions['dividends']
    transactions['total_returns'] = transactions['total_gains'] / (transactions['cumulative_total_cost'] + transactions.groupby('symbol')['sell_cost_basis'].cumsum())
    return transactions

def calculate_daily_portfolio_returns(transactions):
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

    return portfolio_value, portfolio_returns

def create_summary_table(transactions):
    summary = transactions.groupby('symbol').last()

    # Rename columns
    summary = summary.rename(columns={
        'symbol': 'Symbol',
        'cumulative_quantity': 'Quantity',
        'adj_close': 'Current Price',
        'daily_value': 'Current Value',
        'realized_gains': 'Realized Gain/Loss',
        'realized_returns': 'Realized Return',
        'unrealized_gains': 'Unrealized Gain/Loss',
        'unrealized_returns': 'Unrealized Return',
        'dividends': 'Dividends Collected'})
    
    summary = summary.sort_values('Quantity', ascending=False)
    
    # Create a dataframe for cash
    cash_df = pd.DataFrame(index=['Cash'], columns=['Current Value'])
    cash_df.loc['Cash', 'Current Value'] = transactions['cumulative_cash'].iloc[-1]

    # Concatenate cash_df with the rest of the summary dataframe
    summary = pd.concat([cash_df, summary])

    # Calculate allocations for each holding
    summary['Allocation'] = summary['Current Value'] / summary['Current Value'].sum()

    return summary

def load_benchmark_data(start_date, end_date):
    # Load daily price data for S&P 500 to use as a benchmark
    benchmark_daily = yf.download('^GSPC', start=start_date, end=end_date)['Adj Close']

    # Change column name to 'S&P 500'
    benchmark_daily = benchmark_daily.rename('S&P 500')

    # Convert index to datetime
    benchmark_daily.index = pd.to_datetime(benchmark_daily.index)

    return benchmark_daily



