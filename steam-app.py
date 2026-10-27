"""
>streamlit run steam-app.py
"""

import streamlit as st #må ha med for å bruke st funksjoner
from predict import MODEL_PATH, predict_price

st.title("Steam App")


st.write("HVL ML Assignment 2 Prosjekt")

st.divider()


#url = st.text_input("URL")

#type = st.pills("Type", ["app", "bundle"])

publisher = st.text_input("Publisher", key = "publisher", placeholder = "Annapurna Interactive")

developer = st.text_input("Developer", key = "developer", placeholder = "Mobius Digital")

#recentCount = st.number_input("Recent Reviews Count", min_value = 0, step = 1, key = "recent-count")

#min og max verdi representerer fra 0.00 til 1.00 prosent av nylige positive reviews
#"min_value" og "max_value" må være av samme type, som her er double
#recentPositive = st.slider("Recent Positive Reviews (%)", min_value = 0.00, max_value = 1.00, step = 0.01, key = "recent-positive", help = "From 0-100%, 0.00 = 0 and 1.00 = 100", label_visibility = "visible") #'help = "string"' viser en notat når du har musen over feltet om 'label_visibility = "visible"'

#reviewsCount = st.number_input("Reviews Count", min_value = 0, step = 1, key = "reviews-count")

#min og max verdi representerer fra 0.00 til 1.00 av positive reviews
#"min_value" og "max_value" må være av samme type, som her er double
#reviewsPositive = st.slider("Recent Positive Reviews (%)", min_value = 0.00, max_value = 1.00, step = 0.01, key = "reviews-positive", help = "From 0-100%, 0.00 = 0 and 1.00 = 100", label_visibility = "visible")


releaseDate = st.number_input("Release Date", key = "release-date", min_value = 2003, step = 1) #utslippsdatoen


popularTags = st.text_input("Popular Tags", key = "popular-tags", help = "Add tags seperated by ','", label_visibility =  "visible", placeholder = "Exploration, Space, Mystery") #Kan være en annen input type?

gameDetails = st.text_input("Game Details", key = "game-details", help = "Add tags seperated by ','", label_visibility = "visible", placeholder = "Single-player")

achievements = st.number_input("Achievements", min_value = 0, step = 1, key = "achievements") #step=1, gjør at verdien må være et int og ikke et desimaltall

genre = st.text_input("Genre", key = "genre", help = "Input a single genre.", label_visibility = "visible", placeholder = "Adventure")

languages = st.text_input("Languages", key = "languages", help = "Add languages seperated by ','", label_visibility = "visible")

#price = st.number_input("Price (In dollars $)", min_value = 0, max_value = 999, value = 0, step = 1, key = "price")

#sjekker om du liker ender. Kan nekte å funke om denne ikke sjekkes av
likeducks = st.checkbox("I like ducks", key = "quack", help = "Mandatory.", label_visibility = "visible") 

st.divider()



if st.button("Submit", key = "submit", disabled = not likeducks): #"disabled" sørger for at knappen skrus av når "likeducks" er i 'false' tilstand

    if not likeducks:#hvis du ikke like ender...
        st.write("Please confirm you like ducks.") #skriver ut melding.

    else: #ellers om likeducks er 'true'...

        game = {
            "publisher": publisher,
            "developer": developer,
            "popular_tags": popularTags,
            "genre": genre,
            "game_details": gameDetails,
            "achievements": int(achievements),
            "release_year": int(releaseDate),
            "languages": languages
        } #konverterer 'achievements' til int type
        #lager en 'game' liste som passer parametere til input som 'predict_price' funksjon forventer.
       
        try: #prøver å utføre følgende kode blokk...
            prediction = round(predict_price(game), 2) #lager en 'prediction' objekt som blir da desimaltall verdi returnert av predict_price(game)
        except(OSError, KeyError, TypeError, ValueError) as exc: #om du har OSError, KeyError, TypeError eller ValueError blir det satt til 'exc' (exception type) og blokken kjøres
            st.error(f"Couldn't make a prediction: **{exc}**") #printer ut error melding  at man ikke kunne gjøre anslag og sender 'exc' objekt
        else: #ellers om det ikke blir feil kjøres denne kode blokken. VIKTIG: Kjøres bare om det ikke er feil.
            st.success(f"the predicted price is: **{prediction}**") #printer ut prediction pris verdi
    


