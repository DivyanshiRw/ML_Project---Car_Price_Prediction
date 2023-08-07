import streamlit as st
import joblib
import pandas as pd

model = joblib.load('DR_carPrice.joblib')
# Load the preprocessor if you used one during training
preprocessor = joblib.load('preprocessor.joblib')

def predict_car_price(car_features):
    # Assuming car_features is a dictionary containing the user inputs
    car_features_df = pd.DataFrame(car_features, index=[0])
    if preprocessor:
        car_features_transformed = preprocessor.transform(car_features_df)
    else:
        car_features_transformed = car_features_df
    return model.predict(car_features_transformed)

def main():
    st.title('Car Price Prediction')

    car_name = st.text_input('Enter the car name:')
    year = st.number_input('Enter the car year:', min_value=2010, max_value=2023)
    distance = st.number_input('Enter the car distance:')
    owner = st.number_input('Enter the number of previous owners:')
    fuel = st.selectbox('Select the fuel type:', ['PETROL', 'DIESEL', 'CNG', 'LPG'])
    drive = st.selectbox('Select the drive type:', ['Manual', 'Automatic'])
    car_type = st.selectbox('Select the car type:', ['HatchBack', 'Sedan', 'SUV', 'Lux_SUV', 'Lux_sedan'])

    car_features = {
        'Car Name': car_name,
        'Year': year,
        'Distance': distance,
        'Owner': owner,
        'Fuel': fuel,
        'Drive': drive,
        'Type': car_type
    }

    if st.button('Predict Price'):
        predicted_price = predict_car_price(car_features)
        st.success(f'Predicted Car Price: {predicted_price[0]:.2f}')

if __name__ == '__main__':
    main()



