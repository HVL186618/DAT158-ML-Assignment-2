import streamlit as st
import numpy as np
import joblib as jlib

#Laster lagret model og scaler
model = jlib.load("model/steam_price_model.pkl") #TODO: sett inn classifier model .pkl fil her
scaler = jlib.load("model/steam_price_scaler.duck") #TODO: Sett inn scaler .pkl fil her

st.title("Steam App")


st.write("HVL ML Assignment 2 Prosjekt")

st.divider()

#TODO Implementer resten av input. Fjern det som ikke trengs

#url = st.text_input("URL")

#type = st.pills("Type", ["app", "bundle"])

publisher = st.text_input("Publisher", key = "publisher", placeholder = "Annapurna Interactive")

developer = st.text_input("Developer", key = "developer", placeholder = "Mobius Digital")

recentCount = st.number_input("Recent Reviews Count", min_value = 0, step = 1, key = "recent-count")

#min og max verdi representerer fra 0.00 til 1.00 prosent av nylige positive reviews
#"min_value" og "max_value" må være av samme type, som her er double
recentPositive = st.slider("Recent Positive Reviews (%)", min_value = 0.00, max_value = 1.00, step = 0.01, key = "recent-positive", help = "From 0-100%, 0.00 = 0 and 1.00 = 100", label_visibility = "visible") #'help = "string"' viser en notat når du har musen over feltet om 'label_visibility = "visible"'

reviewsCount = st.number_input("Reviews Count", min_value = 0, step = 1, key = "reviews-count")

#min og max verdi representerer fra 0.00 til 1.00 av positive reviews
#"min_value" og "max_value" må være av samme type, som her er double
reviewsPositive = st.slider("Recent Positive Reviews (%)", min_value = 0.00, max_value = 1.00, step = 0.01, key = "reviews-positive", help = "From 0-100%, 0.00 = 0 and 1.00 = 100", label_visibility = "visible")


releaseDate = st.date_input("Release Date", key = "release-date"); #utslippsdatoen


popularTags = st.text_input("Popular Tags", key = "popular-tags", help = "Add tags seperated by ','", label_visibility =  "visible", placeholder = "Exploration, Space, Mystery") #Kan være en annen input type?

gameDetails = st.text_input("Game Details", key = "game-details", help = "Add tags seperated by ','", label_visibility = "visible", placeholder = "Single-player")

achievements = st.number_input("Achievements", min_value = 0, step = 1, key = "achievements") #step=1, gjør at verdien må være et int og ikke et desimaltall

genre = st.text_input("Genre", key = "genre", help = "Input a single genre.", label_visibility = "visible", placeholder = "Adventure")

price = st.number_input("Price (In dollars $)", min_value = 0, max_value = 999, value = 0, step = 1, key = "price")

#sjekker om du liker ender. Kan nekte å funke om denne ikke sjekkes av
likeducks = st.checkbox("I like ducks", key = "quack", help = "Mandatory.", label_visibility = "visible") 

st.divider()

if not likeducks:
    st.write("Please confirm you like ducks.")

#Forbedrer input
#TODO: fjern kommentar og input parametere inni features array
features = np.array([[publisher, developer, recentCount]]) #lager en array for parametere
#features_scaled = scaler.transform(features)


if st.button("Submit", key = "submit", disabled = not likeducks): #"disabled" sørger for at knappen skrus av når "likeducks" er i 'false' tilstand
    #prediction = model.predict("features_scaled")[0] #juster features
    #st.success(f"the predicted price is: **{prediction}**) #printer ut prediction pris verdi
    st.write("Test") #bare en test for å se om knappen funker. Kan fjernes etterpå
    


