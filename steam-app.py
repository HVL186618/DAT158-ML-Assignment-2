import streamlit as st


st.title("Steam App")


st.write("HVL ML Assignment 2 Prosjekt")


url = st.text_input("URL")

type = st.pills("Type", ["app", "bundle"])

#sjekker om du liker ender. Kan nekte å funke om denne ikke sjekkes av
likeducks = st.checkbox("I like ducks") 
