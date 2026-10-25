import streamlit as st
import pandas as pd
import joblib
import os

st.title("Tourism Package Purchase Prediction")
st.write("Enter details below to predict if the customer will purchase the Wellness Tourism Package:")

# Input fields
age = st.number_input("Age", min_value=18, max_value=100, value=30)
type_of_contact = st.selectbox("Type of Contact", ["Company Invited", "Self Inquiry"])
city_tier = st.selectbox("City Tier", [1, 2, 3], index=0)
occupation = st.selectbox("Occupation", ["Salaried", "Small Business", "Large Business", "Free Lancer"])
gender = st.selectbox("Gender", ["Male", "Female"])
num_person_visiting = st.number_input("Number of Persons Visiting", min_value=1, max_value=10, value=2)
preferred_property_star = st.selectbox("Preferred Property Star", [3.0, 4.0, 5.0])
marital_status = st.selectbox("Marital Status", ["Single", "Married", "Divorced", "Unmarried"])
num_trips = st.number_input("Number of Trips", min_value=1, max_value=20, value=3)
passport = st.selectbox("Passport", [0, 1])
own_car = st.selectbox("Own Car", [0, 1])
num_children_visiting = st.number_input("Number of Children Visiting", min_value=0, max_value=10, value=0)
designation = st.selectbox("Designation", ["Executive", "Manager", "Senior Manager", "AVP", "VP"])
monthly_income = st.number_input("Monthly Income", min_value=1000, value=20000)
pitch_satisfaction_score = st.slider("Pitch Satisfaction Score", 1, 5, 3)
product_pitched = st.selectbox("Product Pitched", ["Basic", "Standard", "Deluxe", "Super Deluxe", "King"])
num_followups = st.slider("Number of Follow-ups", 1, 10, 3)
duration_of_pitch = st.number_input("Duration of Pitch", min_value=1, value=15)

# Create DataFrame
input_data = pd.DataFrame([{
    'Age': age,
    'TypeofContact': type_of_contact,
    'CityTier': city_tier,
    'Occupation': occupation,
    'Gender': gender,
    'NumberOfPersonVisiting': num_person_visiting,
    'PreferredPropertyStar': preferred_property_star,
    'MaritalStatus': marital_status,
    'NumberOfTrips': num_trips,
    'Passport': passport,
    'OwnCar': own_car,
    'NumberOfChildrenVisiting': num_children_visiting,
    'Designation': designation,
    'MonthlyIncome': monthly_income,
    'PitchSatisfactionScore': pitch_satisfaction_score,
    'ProductPitched': product_pitched,
    'NumberOfFollowups': num_followups,
    'DurationOfPitch': duration_of_pitch
}])

# Load model
model_path = os.path.join(os.path.dirname(__file__), "model.pkl")
if os.path.exists(model_path):
    model = joblib.load(model_path)
    if st.button("Predict"):
        prediction = model.predict(input_data)[0]
        if prediction == 1:
            st.success("Prediction: Yes, the customer is likely to purchase the Wellness Tourism Package!")
        else:
            st.warning("Prediction: No, the customer is unlikely to purchase the Wellness Tourism Package.")
else:
    st.error(f"Model file not found at {model_path}. Please train and save the model first.")
