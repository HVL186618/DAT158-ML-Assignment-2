import streamlit as st

# Set up the title of your web app
st.title("My First Streamlit App")

# Add a simple text description
st.write("Welcome! This app runs purely on Python.")

# Add an interactive slider widget
age = st.slider("Select your age", 0, 100, 25)
st.write(f"You selected: {age}")