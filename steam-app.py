import streamlit as st


st.title("Steam App")


st.write("HVL ML Assignment 2 Prosjekt")

st.divider()

#TODO Implementer resten av input. Fjern det som ikke trengs

#url = st.text_input("URL")

#type = st.pills("Type", ["app", "bundle"])

recentCount = st.number_input("Recent Reviews Count", min_value = 0, step = 1, key = "recent-count")

#min og max verdi representerer fra 0.00 til 1.00 prosent av nylige positive reviews
#"min_value" og "max_value" må være av samme type, som her er double
recentPositive = st.slider("Recent Positive Reviews (%)", min_value = 0.00, max_value = 1.00, step = 0.01, key = "recent-positive", help = "From 0-100%, 0.00 = 0 and 1.00 = 100", label_visibility = "visible") #'help = "string"' viser en notat når du har musen over feltet om 'label_visibility = "visible"'

reviewsCount = st.number_input("Reviews Count", min_value = 0, step = 1, key = "reviews-count")

#min og max verdi representerer fra 0.00 til 1.00 av positive reviews
#"min_value" og "max_value" må være av samme type, som her er double
reviewsPositive = st.slider("Recent Positive Reviews (%)", min_value = 0.00, max_value = 1.00, step = 0.01, key = "reviews-positive", help = "From 0-100%, 0.00 = 0 and 1.00 = 100", label_visibility = "visible")


releaseDate = st.date_input("Release Date", key = "release-date"); #utslippsdatoen


tags = st.text_input("Tags", key = "tags", help = "Add tags seperated by ','", label_visibility =  "visible") #Kan være en annen input type?

gameDetails = st.text_input("Game Details", key = "game-details", help = "Add tags seperated by ','", label_visibility = "visible")

achievements = st.number_input("Achievements", min_value = 0, step = 1, key = "achievements") #step=1, gjør at verdien må være et int og ikke et desimaltall

genre = st.text_input("Genre", key = "genre", help = "Input a single genre.", label_visibility = "visible")

price = st.number_input("Price (In dollars $)", min_value = 0, max_value = 999, value = 0, step = 1, key = "price")

#sjekker om du liker ender. Kan nekte å funke om denne ikke sjekkes av
likeducks = st.checkbox("I like ducks", key = "quack") 

st.divider()

if not likeducks:
    st.write("Please confirm you like ducks.")


submit = st.button("Submit", key = "submit", disabled = not likeducks)

