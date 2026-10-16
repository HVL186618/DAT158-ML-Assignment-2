import streamlit as st


st.title("Steam App")


st.write("HVL ML Assignment 2 Prosjekt")


url = st.text_input("URL")

type = st.pills("Type", ["app", "bundle"])

pris = st.number_input("Price (In dollars $)", min_value=0, max_value=999, value=0, step=1)

#sjekker om du liker ender. Kan nekte å funke om denne ikke sjekkes av
likeducks = st.checkbox("I like ducks") 
if not likeducks:
    st.write("Please confirm you like ducks.")

