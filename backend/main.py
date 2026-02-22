from fastapi import FastAPI, UploadFile, File, HTTPException
import pandas as pd
import numpy as np
import joblib
from tensorflow.keras.models import load_model
import io
import uvicorn

app = FastAPI(title="NIDS Prediction API")

# Global variables for model and scaler
model = None
scaler = None

# Load assets when the API starts up
@app.on_event("startup")
def load_assets():
    global model, scaler
    try:
        model = load_model('models/nids_dcnn_model.h5')
        scaler = joblib.load('models/cicids_scaler.pkl')
        print("✅ Model and Scaler loaded successfully!")
    except Exception as e:
        print(f"❌ Error loading model/scaler: {e}")

@app.post("/predict")
async def predict_network_traffic(file: UploadFile = File(...)):
    if not file.filename.endswith('.csv'):
        raise HTTPException(status_code=400, detail="Only CSV files are supported.")
    
    try:
        # 1. Read the uploaded CSV content
        contents = await file.read()
        df = pd.read_csv(io.BytesIO(contents))
        
        # 2. Preprocess
        to_drop = ['Flow ID', 'Source IP', 'Destination IP', 'Timestamp', 'Label']
        df_features = df.drop(columns=[col for col in to_drop if col in df.columns])
        
        # 3. Scale features
        scaled_features = scaler.transform(df_features)
        
        # 4. Reshape for Conv1D: (Samples, Features, 1)
        num_samples = scaled_features.shape[0]
        num_features = scaled_features.shape[1]
        X_input = scaled_features.reshape(num_samples, num_features, 1)
        
        # 5. Predict
        predictions = model.predict(X_input)
        predicted_classes = np.argmax(predictions, axis=1)
        
        # 6. Map to labels
        label_mapping = {0: "BENIGN", 1: "🚨 ATTACK DETECTED"}
        human_readable_preds = [label_mapping[val] for val in predicted_classes]
        
        # Return only the predictions to save bandwidth
        return {"predictions": human_readable_preds}
        
    except ValueError as ve:
        raise HTTPException(status_code=400, detail=f"Feature mismatch error: {ve}")
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Internal server error: {e}")

if __name__ == "__main__":
    uvicorn.run("main:app", host="127.0.0.1", port=8000, reload=True)