import streamlit as st


st.title("Steam App")


st.write("HVL ML Assignment 2 Prosjekt")


url = st.text_input("URL")

type = st.pills("Type", ["app", "bundle"])

price = st.number_input("Price (In dollars $)", min_value=0, max_value=999, value=0, step=1)

#TODO Implementer resten av input

tags = st.text_input("Tags") #Kan være en annen input type?

genre = st.text_input("Genre")

#sjekker om du liker ender. Kan nekte å funke om denne ikke sjekkes av
likeducks = st.checkbox("I like ducks") 
if not likeducks:
    st.write("Please confirm you like ducks.")
