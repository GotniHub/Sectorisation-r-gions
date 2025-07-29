import streamlit as st

st.set_page_config(
    page_title="Hello",
    page_icon="👋",
)

#logo
st.logo("LOGO.png", icon_image="Logom.png")

st.write("# Welcome, C'est Advent+ Africa! 👋")
st.image("LOGO.png", use_column_width=True)