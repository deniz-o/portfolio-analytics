import streamlit as st
from Dashboard import transactions

st.title('Transactions')
st.dataframe(transactions.loc[transactions['type'].notna()].set_index('date'))