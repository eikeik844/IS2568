import streamlit as st
import pandas as pd
import numpy as np
import joblib
import tensorflow as tf

# ==========================================
# 1. SETUP & ASSET LOADING
# ==========================================
st.set_page_config(page_title="Air Quality Predictor", layout="wide")

# Load models and scaler (cached so it only loads once)
@st.cache_resource
def load_assets():
    scaler = joblib.load('scaler.pkl')
    ensemble_model = joblib.load('ensemble_model.pkl')
    nn_model = tf.keras.models.load_model('neural_network_model.keras')
    return scaler, ensemble_model, nn_model

scaler, ensemble_model, nn_model = load_assets()

# The features our models were trained on
feature_names = ['PT08.S1(CO)', 'PT08.S2(NMHC)', 'NOx(GT)', 'NO2(GT)', 'T', 'RH', 'AH']

# ==========================================
# 2. SIDEBAR NAVIGATION
# ==========================================
st.sidebar.title("Navigation")
page = st.sidebar.radio("Go to:", [
    "1. Explain: Ensemble Model", 
    "2. Explain: Neural Network", 
    "3. Test: Ensemble Model", 
    "4. Test: Neural Network"
])

# ==========================================
# 3. PAGE 1: ENSEMBLE EXPLANATION
# ==========================================
if page == "1. Explain: Ensemble Model":
    st.title("Model 1: Ensemble Machine Learning")
    st.write("### 1. Data Preparation")
    st.write("The dataset used is the AirQualityUCI dataset. It contained severe imperfections, such as garbage values (-200) and missing data. We cleaned this by dropping corrupted columns, dropping empty rows, and imputing missing values with the median. We also merged baseline AQI data from a Global Air Pollution dataset.")
    
    st.write("### 2. Algorithm Theory")
    st.write("This model uses a **Voting Regressor**, which is an ensemble technique. It combines the predictions of three distinct algorithms to create a stronger, more accurate final prediction:")
    st.markdown("""
    * **Random Forest:** A bagging method that builds multiple decision trees.
    * **Gradient Boosting:** A boosting method that builds trees sequentially to correct prior errors.
    * **Support Vector Regressor (SVR):** Finds the hyperplane that best fits the continuous data.
    """)
    
    st.write("### 3. Development Process")
    st.write("The features were standardized using a `StandardScaler` so that all sensors had equal weight. The Voting Regressor was then trained on 80% of the dataset to predict the target variable: Carbon Monoxide (CO).")

# ==========================================
# 4. PAGE 2: NEURAL NETWORK EXPLANATION
# ==========================================
elif page == "2. Explain: Neural Network":
    st.title("Model 2: Deep Neural Network")
    st.write("### 1. Data Preparation")
    st.write("Similar to the Ensemble model, the Neural Network requires strictly clean numerical data. Deep learning models are highly sensitive to unscaled data, so applying the `StandardScaler` was a critical step in the preparation pipeline.")
    
    st.write("### 2. Algorithm Theory")
    st.write("A Neural Network attempts to simulate the human brain using layers of interconnected nodes (neurons). Our architecture is a Custom Sequential Model:")
    st.markdown("""
    * **Input Layer:** Receives the 7 scaled environmental features.
    * **Hidden Layers:** Three Dense layers (64, 32, and 16 neurons) using the ReLU activation function to learn complex, non-linear relationships in the gas levels.
    * **Dropout Layer:** A 20% Dropout layer was included to randomly turn off neurons during training, which prevents the model from overfitting.
    * **Output Layer:** A single linear node to predict the continuous CO value.
    """)
    
    st.write("### 3. Development Process")
    st.write("The model was compiled using the Adam optimizer and Mean Squared Error (MSE) as the loss function. It was trained for 50 epochs with a validation split to monitor performance and prevent overfitting.")

# ==========================================
# 5. PAGES 3 & 4: TESTING INTERFACES
# ==========================================
else:
    if page == "3. Test: Ensemble Model":
        st.title("Test the Ensemble Model")
        active_model = ensemble_model
        model_type = "ensemble"
    else:
        st.title("Test the Neural Network")
        active_model = nn_model
        model_type = "nn"

    st.write("Adjust the sensor values below to predict the Carbon Monoxide `CO(GT)` concentration.")

    # Create input sliders for the user
    col1, col2 = st.columns(2)
    with col1:
        pt08_s1 = st.number_input("PT08.S1 (CO Sensor)", value=1000.0)
        pt08_s2 = st.number_input("PT08.S2 (NMHC Sensor)", value=1000.0)
        nox = st.number_input("NOx (Nitrogen Oxides)", value=100.0)
        no2 = st.number_input("NO2 (Nitrogen Dioxide)", value=100.0)
    with col2:
        t = st.number_input("Temperature (°C)", value=20.0)
        rh = st.number_input("Relative Humidity (%)", value=50.0)
        ah = st.number_input("Absolute Humidity", value=1.0)

    # When the user clicks the Predict button
    if st.button("Predict CO Level"):
        # 1. Gather inputs into a numpy array
        input_data = np.array([[pt08_s1, pt08_s2, nox, no2, t, rh, ah]])
        
        # 2. Scale the input using our saved scaler
        scaled_input = scaler.transform(input_data)
        
        # 3. Predict!
        if model_type == "ensemble":
            prediction = active_model.predict(scaled_input)[0]
        else:
            prediction = active_model.predict(scaled_input).flatten()[0]
            
        # 4. Display the result
        st.success(f"Predicted Carbon Monoxide (CO) Level: **{prediction:.2f}**")