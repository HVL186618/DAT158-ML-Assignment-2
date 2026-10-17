import streamlit as st


st.title("Steam App")


st.write("HVL ML Assignment 2 Prosjekt")


#url = st.text_input("URL")

#type = st.pills("Type", ["app", "bundle"])

recentcount = st.number_input("Recent Reviews Count", min_value = 0, step = 1)

recentpositive = st.number_input("Recent Positive %", max_value = 1, step = 0.01)


releasedate = st.date_input("Release Date"); #utslippsdatoen


#TODO Implementer resten av input

tags = st.text_input("Tags") #Kan være en annen input type?

genre = st.text_input("Genre")

game_details = st.text_input("Game Details")

achievements = st.number_input("Achievements", min_value = 0, step=1) #step=1, gjør at verdien må være et int og ikke et desimaltall

price = st.number_input("Price (In dollars $)", min_value=0, max_value=999, value=0, step=1)

#sjekker om du liker ender. Kan nekte å funke om denne ikke sjekkes av
likeducks = st.checkbox("I like ducks") 
if not likeducks:
    st.write("Please confirm you like ducks.")
