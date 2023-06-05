import streamlit as st
from Dashboard import transactions

st.dataframe(transactions.set_index('date'))