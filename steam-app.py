import streamlit as st


st.title("Steam App")


st.write("HVL ML Assignment 2 Prosjekt")

#TODO Implementer resten av input. Fjern det som ikke trengs

#url = st.text_input("URL")

#type = st.pills("Type", ["app", "bundle"])

recentCount = st.number_input("Recent Reviews Count", min_value = 0, step = 1, key = "recent-count")

recentPositive = st.slider("Recent Positive Reviews (%)", min_value = 0.00, max_value = 1.00, step = 0.01, key = "recent-positive")

reviewsCount = st.number_input("Reviews Count", min_value = 0, step = 1, key = "reviews-count")

reviewsPositive = st.slider("Recent Positive Reviews(%)", min_value = 0.00, max_value = 1.00, step = 0.01, key = "reviews-positive")


releaseDate = st.date_input("Release Date", key = "release-date"); #utslippsdatoen


tags = st.text_input("Tags", key = "tags") #Kan være en annen input type?

gameDetails = st.text_input("Game Details", key = "game-details")

achievements = st.number_input("Achievements", min_value = 0, step = 1, key = "achievements") #step=1, gjør at verdien må være et int og ikke et desimaltall

genre = st.text_input("Genre", key = "genre")

price = st.number_input("Price (In dollars $)", min_value = 0, max_value = 999, value = 0, step = 1, key = "price")

#sjekker om du liker ender. Kan nekte å funke om denne ikke sjekkes av
likeducks = st.checkbox("I like ducks", key = "quack") 

if not likeducks:
    st.write("Please confirm you like ducks.")


submit = st.button("Submit", disabled = not likeducks)
