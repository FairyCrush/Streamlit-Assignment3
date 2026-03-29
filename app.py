import streamlit as st
import pandas as pd
import joblib

model = joblib.load("models/model.pkl")

st.title("ASG 04 MD - Bevlin Logen - Spaceship Titanic Model Deployment")
st.header("Passenger Information")

HomePlanet = st.selectbox("HomePlanet", ["Earth", "Europa", "Mars"])
CryoSleep = st.selectbox("CryoSleep", [True, False])
Destination = st.selectbox("Destination", ["TRAPPIST-1e", "55 Cancri e", "PSO J318.5-22"])
Age = st.number_input("Age", 0, 100, 30)
VIP = st.selectbox("VIP", [True, False])

RoomService = st.number_input("RoomService", value=0.0)
FoodCourt = st.number_input("FoodCourt", value=0.0)
ShoppingMall = st.number_input("ShoppingMall", value=0.0)
Spa = st.number_input("Spa", value=0.0)
VRDeck = st.number_input("VRDeck", value=0.0)

Cabin = st.text_input("Cabin", "C/123/S")
PassengerId = st.text_input("PassengerId", "0001_01")
Name = st.text_input("Name", "John Doe")

if st.button("Predict"):
    
   input_data = pd.DataFrame([{
       "HomePlanet": HomePlanet,
       "CryoSleep": CryoSleep,
       "Destination": Destination,
       "Age": Age,
       "VIP": VIP,
       "RoomService": RoomService,
       "FoodCourt": FoodCourt,
       "ShoppingMall": ShoppingMall,
       "Spa": Spa,
       "VRDeck": VRDeck,
       "Cabin": Cabin,
       "PassengerId": PassengerId,
       "Name": Name
    }])
   prediction = model.predict(input_data)
    
   if prediction[0] == True:
       st.success("Passenger Was Transported")
   else:
       st.error("Passenger Was Not Transported")