import streamlit as st

# Set up the title of your web app
st.title("Steam App")

# Add a simple text description
st.write("HVL ML Assignment 2 Prosjekt")

# Add an interactive slider widget
age = st.slider("Select your age", 0, 100, 25)
st.write(f"You selected: {age}")


st.text_input("URL")
