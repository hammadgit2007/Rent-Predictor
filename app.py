import streamlit as st
import numpy as np
import pandas as pd
from sklearn.linear_model import LinearRegression

# Title of the web app
st.set_page_config(page_title="Rent Predictor", page_icon="🏠", layout="centered")
st.title("🏠 Pakistani House Rent Predictor")
st.write("Adjust the features below to estimate the monthly rent of a house in Pakistan.")

# We use @st.cache_resource so the model trains ONLY ONCE when the app starts, not on every click!
@st.cache_resource
def get_trained_model():
    np.random.seed(42)
    n_houses = 500
    
    # Generate the exact same dataset we designed
    rooms = np.random.randint(2, 7, size=n_houses)
    bathrooms = np.random.randint(1, 4, size=n_houses)
    kitchens = np.random.randint(1, 3, size=n_houses)
    parking_spaces = np.random.randint(0, 3, size=n_houses)
    
    base_price = 50000
    noise = np.random.normal(0, 15000, size=n_houses)
    
    prices_pkr = (
        base_price 
        + (rooms * 3000) 
        + (bathrooms * 1000) 
        + (kitchens * 2000) 
        + (parking_spaces * 2000) 
        + noise
    )
    
    df = pd.DataFrame({
        'Rooms': rooms,
        'Bathrooms': bathrooms,
        'Kitchens': kitchens,
        'Parking_Spaces': parking_spaces,
        'Price_PKR': prices_pkr.round(0)
    })
    
    X = df[['Rooms', 'Bathrooms', 'Kitchens', 'Parking_Spaces']]
    y = df['Price_PKR']
    
    model = LinearRegression()
    model.fit(X, y)
    return model

# Train/load the model
model = get_trained_model()

# --- User Interface Components ---
st.write("---")
st.header("Customize House Specifications")

# Create sliders for the user input
rooms_input = st.slider("Number of Rooms", min_value=2, max_value=6, value=3, step=1)
bathrooms_input = st.slider("Number of Bathrooms", min_value=1, max_value=3, value=1, step=1)
kitchens_input = st.slider("Number of Kitchens", min_value=1, max_value=2, value=1, step=1)
parking_input = st.slider("Parking Spaces", min_value=0, max_value=2, value=1, step=1)

# Prediction Logic
if st.button("Calculate Estimated Rent", type="primary"):
    # Format the input for scikit-learn
    user_house = np.array([[rooms_input, bathrooms_input, kitchens_input, parking_input]])
    
    # Predict
    predicted_rent = model.predict(user_house)[0]
    
    st.write("---")
    st.subheader("Results:")
    st.success(f"Estimated Monthly Rent: **PKR {predicted_rent:,.0f}**")