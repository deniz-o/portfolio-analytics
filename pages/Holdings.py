import streamlit as st
from Dashboard import symbol_transactions, prices, summary

st.title('Individual Holdings')

symbol_transactions = symbol_transactions[symbol_transactions['type'].notna()]
prices = prices.set_index('date')

symbol_selection = st.selectbox('Symbol', symbol_transactions['symbol'].unique())

if symbol_selection:
    col1, col2, col3 = st.columns(3)
    with col1:
        st.metric('Total Gain/Loss', summary.loc[symbol_selection, 'Total Gain/Loss'])
    with col2:
        st.metric('Realized Return', summary.loc[symbol_selection, 'Realized Return'])
    with col3:
        st.metric('Unrealized Return', summary.loc[symbol_selection, 'Unrealized Return'])
    st.subheader('Performance')
    st.line_chart(prices.loc[prices['symbol']==symbol_selection, 'adj_close'])
    st.subheader('Transactions')
    st.dataframe(symbol_transactions[symbol_transactions['symbol'] == symbol_selection].set_index('date'))

else:
    st.write('Select a holding symbol')

    st.dataframe(summary)