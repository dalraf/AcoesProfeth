import streamlit as st
from functions import executar

st.set_page_config(
    page_title="Ações Prophet",
    layout="wide",
    initial_sidebar_state="expanded",
)


if not "df" in st.session_state:
    st.session_state["df"] = executar()

if "df" in st.session_state:
    df = st.session_state["df"]
    st.dataframe(df, use_container_width=True)