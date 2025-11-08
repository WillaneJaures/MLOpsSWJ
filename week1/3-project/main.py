from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
import joblib
import json
import pandas as pd
import uvicorn
import os
import numpy as np

app = FastAPI(title="Churn Predictor API")

# ---- Chargement des artefacts au démarrage ----
MODEL_DIR = './models'
model = joblib.load(os.path.join(MODEL_DIR, 'forest_model.pkl'))
scaler = joblib.load(os.path.join(MODEL_DIR, 'scaler.pkl'))
encoder = joblib.load(os.path.join(MODEL_DIR, 'encoder.pkl'))

with open(os.path.join(MODEL_DIR, 'feature_list.json'), 'r') as f:
    FEATURE_LIST = json.load(f)

with open(os.path.join(MODEL_DIR, 'categorical_cols.json'), 'r') as f:
    CATEGORICAL_COLS = json.load(f)

# ---- schéma d'entrée générique ----
# On accepte un dict de type {"feature_name": value, ...}
class PredictRequest(BaseModel):
    data: dict

def preprocess_input(sample_dict):
    cleaned = {k: (v.lower().strip().replace(" ", "_") if isinstance(v, str) else v)
               for k, v in sample_dict.items()}

    row = {col: cleaned.get(col, np.nan) for col in FEATURE_LIST}
    df = pd.DataFrame([row])

    # ⛔️ on ne refait pas d'encodage : on suppose que FEATURE_LIST est déjà encodé
    df_final = df.fillna(0).astype(float)

    X_scaled = scaler.transform(df_final)
    return X_scaled

@app.post("/predict")
def predict(req: PredictRequest):
    """
    Attends JSON: {"data": {"feature1": value1, "feature2": value2, ...}}
    """
    X_in = req.data
    X_proc = preprocess_input(X_in)

    # prédiction (probabilité + classe)
    probs = model.predict_proba(X_proc)  # shape (1, 2) si binaire
    pred = model.predict(X_proc)         # shape (1,)

    # on renvoie la probabilité de churn=1 et la prédiction binaire
    # Ajuste l'index si ta classe positive n'est pas la seconde colonne
    churn_prob = float(probs[0][1])
    churn_pred = int(pred[0])

    return {
        "prediction": churn_pred,
        "churn_probability": churn_prob
    }

# Optionnel : endpoint healthcheck
@app.get("/health")
def health():
    return {"status": "ok"}

# pour debug / dev : lance avec `python main.py` (ou via uvicorn)
if __name__ == "__main__":
    uvicorn.run("main:app", host="0.0.0.0", port=8000, reload=True)
