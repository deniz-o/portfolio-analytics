import streamlit as st
from Dashboard import transactions

st.dataframe(transactions[transactions['type'].notna()].set_index('date'))