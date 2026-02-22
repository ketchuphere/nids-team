import streamlit as st
import pandas as pd
import numpy as np
import joblib
from tensorflow.keras.models import load_model

# --- Page Config ---
st.set_page_config(page_title="NIDS Deep Learning Demo", layout="wide")
st.title("🛡️ Network Intrusion Detection System (DCNN)")
st.write("Upload a CSV file containing network flows to detect malicious traffic.")

# --- Load Model and Scaler ---
# We use st.cache_resource so it only loads these large files once when the app starts
@st.cache_resource
def load_nids_assets():
    model = load_model('models/nids_dcnn_model.h5')
    scaler = joblib.load('models/cicids_scaler.pkl')
    return model, scaler

try:
    model, scaler = load_nids_assets()
    st.success("✅ Model and Scaler loaded successfully!")
except Exception as e:
    st.error(f"Error loading model/scaler: {e}")
    st.stop()

# --- File Uploader ---
uploaded_file = st.file_uploader("Upload Network Traffic (CSV)", type=["csv"])

if uploaded_file is not None:
    st.write("### 🔍 Analyzing Network Traffic...")
    
    # 1. Read the CSV
    df = pd.read_csv(uploaded_file)
    
    # Store original dataframe to display later
    display_df = df.copy()

    # 2. Preprocess the uploaded data (must match training exactly)
    # Drop identifier columns if they exist in the uploaded CSV
    to_drop = ['Flow ID', 'Source IP', 'Destination IP', 'Timestamp', 'Label']
    df_features = df.drop(columns=[col for col in to_drop if col in df.columns])
    
    try:
        # 3. Scale the features using your saved training scaler
        scaled_features = scaler.transform(df_features)
        
        # 4. Reshape for the Conv1D layer: (Samples, Features, 1)
        num_samples = scaled_features.shape[0]
        num_features = scaled_features.shape[1]
        X_input = scaled_features.reshape(num_samples, num_features, 1)
        
        # 5. Make Predictions
        predictions = model.predict(X_input)
        
        # Since we used softmax with 2 units, argmax gets the predicted class (0 or 1)
        predicted_classes = np.argmax(predictions, axis=1)
        
        # 6. Map predictions back to human-readable labels
        # Assuming 0 was BENIGN and 1 was Attack during your preprocessing
        label_mapping = {0: "BENIGN", 1: "🚨 ATTACK DETECTED"}
        display_df['DCNN Prediction'] = [label_mapping[val] for val in predicted_classes]
        
        # 7. Display Results
        st.write(f"Processed **{num_samples}** network flows.")
        
        # Highlight attacks in red
        def color_attacks(val):
            color = 'red' if val == "🚨 ATTACK DETECTED" else 'green'
            return f'color: {color}'
            
        st.dataframe(display_df.style.applymap(color_attacks, subset=['DCNN Prediction']))
        
        # Show a summary chart
        st.write("### Prediction Summary")
        summary_counts = display_df['DCNN Prediction'].value_counts()
        st.bar_chart(summary_counts, color=["#FF0000", "#00FF00"])

    except ValueError as e:
        st.error(f"Feature mismatch! Ensure your CSV has the exact same feature columns as your training data. Error details: {e}")